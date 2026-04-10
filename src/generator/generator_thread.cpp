#include "generator/generator_thread.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_set>

#include <cuda_runtime.h>

#include "generator/generator.h"
#include "models/tag_dict.h"
#include "tokenizer/tokenizer.hpp"
#include "utils/types.hpp"

namespace generator
{
    namespace
    {
        bool hasClipScoringProfile(const ModelsConfig &modelsConfig)
        {
            return std::any_of(modelsConfig.profiles.begin(), modelsConfig.profiles.end(),
                               [](const ModelProfile &profile)
                               { return profile.usesClipScoring(); });
        }

        bool usesOnlyClipRandomVocabGeneration(const ModelsConfig &modelsConfig)
        {
            return !modelsConfig.profiles.empty() &&
                   std::all_of(modelsConfig.profiles.begin(), modelsConfig.profiles.end(),
                               [](const ModelProfile &profile)
                               { return profile.effectiveGenerationMode() == ModelProfile::GENERATION_MODE_CLIP_RANDOM_VOCAB; });
        }

        std::vector<std::string> collectMissingClipVocabTags(const Tag_dict &dict, const ClipEmbedCache &cache)
        {
            std::unordered_set<std::string> missingSet;
            for (const std::string &rawTag : dict.flatten())
            {
                for (const std::string &subtag : Tokenizer::splitTagString(rawTag))
                {
                    if (!cache.lookupTagId(subtag).has_value())
                    {
                        missingSet.insert(subtag);
                    }
                }
            }

            std::vector<std::string> missing(missingSet.begin(), missingSet.end());
            std::sort(missing.begin(), missing.end());
            return missing;
        }

        std::string formatMissingTagsPreview(const std::vector<std::string> &missing, size_t limit = 8)
        {
            std::ostringstream oss;
            const size_t count = std::min(missing.size(), limit);
            for (size_t i = 0; i < count; ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << "'" << missing[i] << "'";
            }
            if (missing.size() > count)
            {
                oss << ", ...";
            }
            return oss.str();
        }

        int parsePositiveEnvInt(const char *name, int fallback)
        {
            const char *value = std::getenv(name);
            if (value == nullptr || *value == '\0')
                return fallback;
            char *endPtr = nullptr;
            long parsed = std::strtol(value, &endPtr, 10);
            if (endPtr == value || *endPtr != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
                return fallback;
            return static_cast<int>(parsed);
        }

        int deriveScoringThreads(unsigned int cpuThreads, int gpuCount, bool clipEmbedWorkload)
        {
            unsigned int target = static_cast<unsigned int>(std::max(gpuCount, 1) * (clipEmbedWorkload ? 2 : 4));
            unsigned int cpuBudget = std::max(2u, cpuThreads / (clipEmbedWorkload ? 6u : 4u));
            target = std::min(target, cpuBudget);
            target = std::max(target, static_cast<unsigned int>(std::max(gpuCount, 1)));
            target = std::min(target, 32u);
            return static_cast<int>(target);
        }

        int deriveGeneratorThreads(unsigned int cpuThreads, int scoringThreads, bool clipEmbedWorkload)
        {
            unsigned int threads = clipEmbedWorkload
                                       ? std::max(4u, static_cast<unsigned int>(std::max(scoringThreads, 1) * 4))
                                       : std::max(4u, (cpuThreads * 3u) / 4u);
            if (clipEmbedWorkload)
            {
                unsigned int cpuBudget = std::max(4u, cpuThreads / 2u);
                threads = std::min(threads, cpuBudget);
            }
            threads = std::min(threads, 128u);
            return static_cast<int>(threads);
        }
    }


    GeneratorThread::GeneratorThread(const ModelsConfig &modelsConfig, GeneratorRuntimeConfig runtimeConfig)
        : modelsConfig_(modelsConfig), runtimeConfig_(runtimeConfig)
    {
        const bool clipScoringWorkload = hasClipScoringProfile(modelsConfig_);
        const bool lightweightGenerationWorkload = usesOnlyClipRandomVocabGeneration(modelsConfig_);
        const unsigned int cpuThreads = [&]()
        {
            unsigned int count = std::thread::hardware_concurrency();
            if (count == 0)
                count = 32;
            return count;
        }();

        if (runtimeConfig_.numGpus <= 0)
        {
            int gpuCount = 0;
            cudaError_t err = cudaGetDeviceCount(&gpuCount);
            if (err != cudaSuccess || gpuCount <= 0)
                gpuCount = 1;
            runtimeConfig_.numGpus = gpuCount;
        }

        if (runtimeConfig_.numScoringThreads <= 0)
        {
            runtimeConfig_.numScoringThreads = deriveScoringThreads(cpuThreads, runtimeConfig_.numGpus, clipScoringWorkload);
        }

        if (runtimeConfig_.numGeneratorThreads <= 0)
        {
            runtimeConfig_.numGeneratorThreads = deriveGeneratorThreads(
                cpuThreads,
                runtimeConfig_.numScoringThreads,
                lightweightGenerationWorkload);
        }

        if (runtimeConfig_.numScoringThreads > runtimeConfig_.numGeneratorThreads)
        {
            runtimeConfig_.numGeneratorThreads = runtimeConfig_.numScoringThreads;
        }
        if (runtimeConfig_.profileBurstSize <= 0)
        {
            runtimeConfig_.profileBurstSize = modelsConfig_.profiles.size() <= 1
                                                  ? 1
                                                  : parsePositiveEnvInt("SDML_PROFILE_BURST", clipScoringWorkload ? 8 : 1);
        }

        // Build list of profile IDs for round-robin
        for (const auto &profile : modelsConfig_.profiles)
        {
            profileIds_.push_back(profile.profileId);
        }

        // Initialize global Tag_dict singleton with first available dict
        // This is needed because Lora classes depend on Tag_dict::getInstance()
        {
            bool globalDictLoaded = false;
            for (const auto &profile : modelsConfig_.profiles)
            {
                if (!profile.dictPath.empty())
                {
                    if (!Tag_dict::loadFromJson(profile.dictPath))
                    {
                        throw std::runtime_error("Failed to load global Tag_dict from: " + profile.dictPath);
                    }
                    std::cout << "Loaded global Tag_dict from: " << profile.dictPath << std::endl;
                    globalDictLoaded = true;
                    break;
                }
            }
            if (!globalDictLoaded)
            {
                std::cout << "No tag dictionary found in any profile; global Tag_dict will be empty" << std::endl;
            }
        }

        std::cout << "Generator concurrency: " << runtimeConfig_.numGeneratorThreads
                  << " generator thread(s), " << runtimeConfig_.numScoringThreads
                  << " scoring thread(s) across " << runtimeConfig_.numGpus << " GPU(s)." << std::endl;
        std::cout << "Profile burst size: " << runtimeConfig_.profileBurstSize << std::endl;
        std::cout << "Number of model profiles: " << profileIds_.size() << std::endl;

        try
        {
            // Build per-model scoring configurations
            std::vector<ModelScoringConfig> scoringConfigs;
            scoringConfigs.reserve(modelsConfig_.profiles.size());
            for (const auto &profile : modelsConfig_.profiles)
            {
                ModelScoringConfig cfg;
                cfg.profileId = profile.profileId;
                cfg.modelId = profile.modelId;
                cfg.onnxModelPath = profile.onnxModelPath;
                cfg.clipCachePath = profile.usesClipScoring() ? profile.clipCachePath : std::string{};
                cfg.scoreThreshold = profile.scoreThreshold;
                cfg.preferredDeviceId = -1; // Auto-assign based on thread
                cfg.batchSize = (profile.batchSize > 0)
                                    ? static_cast<unsigned>(profile.batchSize)
                                    : 0;
                scoringConfigs.push_back(cfg);
                std::cout << "Model '" << profile.name << "' scoring config: "
                          << profile.onnxModelPath << " (threshold=" << profile.scoreThreshold
                          << ", batch=" << (profile.batchSize > 0 ? std::to_string(profile.batchSize) : "auto")
                          << ")" << std::endl;
            }
            // Create scoring thread pool with per-model configurations
            scoringThreadPool_ = std::make_unique<ScoringThreadPool>(
                scoringConfigs,
                runtimeConfig_.numGeneratorThreads,
                runtimeConfig_.numScoringThreads,
                runtimeConfig_.numGpus);
            std::cout << "Scoring thread pool initialized with " << scoringConfigs.size() << " models" << std::endl;
            // Create one FilteringThread per model profile
            for (const auto &profile : modelsConfig_.profiles)
            {
                FilteringThread::Config filterCfg;
                filterCfg.scoreThreshold = profile.scoreThreshold; // Per-model threshold
                filterCfg.loraCap = runtimeConfig_.loraCap;
                filterCfg.penaltyPerUse = 0.02f;
                filterCfg.useTagsAsLoras = profile.useTagsAsLoras;
                // Score-only mode: no LoRA diversity when both flags are off
                filterCfg.scoreOnly = (!profile.useLoras && !profile.useTagsAsLoras);
                // Apply per-model max output cap if set
                if (profile.maxOutputPrompts > 0)
                {
                    filterCfg.maxAcceptedPrompts = static_cast<size_t>(profile.maxOutputPrompts);
                }
                // Initialize profile components first to get tag dict
                if (!initProfileComponents(profile.profileId))
                {
                    throw std::runtime_error("Failed to initialize components for profile " + profile.name);
                }
                auto &components = profileComponents_[profile.profileId];
                filterCfg.tagDict = components->tagDict;
                auto filterThread = std::make_unique<FilteringThread>(filterCfg);
                filterThread->setTokenizer(Tokenizer(modelsConfig_.tokenizerPath, ',', profile.stripParentheses));
                std::cout << "Filtering thread for profile '" << profile.name << "' (profileId=" << profile.profileId
                          << ", modelId=" << profile.modelId << ") initialized:" << std::endl;
                std::cout << "  - Score threshold: " << filterCfg.scoreThreshold << std::endl;
                std::cout << "  - LoRA cap: " << filterCfg.loraCap << std::endl;
                std::cout << "  - Use tags as LoRAs: " << (filterCfg.useTagsAsLoras ? "yes" : "no") << std::endl;
                std::cout << "  - Score-only mode: " << (filterCfg.scoreOnly ? "yes" : "no") << std::endl;
                std::cout << "  - Max accepted prompts: " << filterCfg.maxAcceptedPrompts << std::endl;
                filteringThreads_[profile.profileId] = std::move(filterThread);
            }
            // Build vector of FilteringThread pointers indexed by profileId
            // (the scoring thread routes prompts using profileId as the vector index)
            int32_t maxProfileId = *std::max_element(profileIds_.begin(), profileIds_.end());
            std::vector<FilteringThread *> filterThreadPtrs(static_cast<size_t>(maxProfileId + 1), nullptr);
            for (const auto &[profileId, filterThread] : filteringThreads_)
            {
                filterThreadPtrs[static_cast<size_t>(profileId)] = filterThread.get();
            }
            // Connect scoring thread pool to filtering threads
            scoringThreadPool_->setFilterThreads(filterThreadPtrs);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to initialize threads: " << e.what() << std::endl;
            throw;
        }

        // Start worker threads
        std::cout << "Starting " << runtimeConfig_.numGeneratorThreads << " generator threads..." << std::endl;
        for (int i = 0; i < runtimeConfig_.numGeneratorThreads; i++)
        {
            generatorThreads_.emplace_back(&GeneratorThread::generatorThreadFunction, this, i);
            std::cout << "Started generator thread " << i << std::endl;
        }
    }

    GeneratorThread::~GeneratorThread()
    {
        stopAndJoin();
    }

    void GeneratorThread::stopAndJoin()
    {
        running_ = false;
        // Unblock generator threads that may be waiting on full scoring input queues.
        if (scoringThreadPool_)
        {
            scoringThreadPool_->closeInputQueues();
        }

        for (auto &thread : generatorThreads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        generatorThreads_.clear();

        // Stop filtering threads first to ensure they no longer access queues
        for (auto &[modelId, filterThread] : filteringThreads_)
        {
            if (filterThread)
            {
                filterThread->stopAndJoin();
            }
        }

        // Now it is safe to stop the scoring thread pool (destroys their queues)
        if (scoringThreadPool_)
        {
            scoringThreadPool_->stopAndJoin();
        }
    }

    size_t GeneratorThread::getFilteredPromptsSize(void) const
    {
        size_t total = 0;
        for (const auto &[modelId, filterThread] : filteringThreads_)
        {
            if (filterThread)
            {
                total += filterThread->acceptedCountAlive();
            }
        }
        return total;
    }

    std::vector<ScoredPrompt> GeneratorThread::getFilteredPrompts()
    {
        std::vector<ScoredPrompt> allPrompts;

        for (auto &[modelId, filterThread] : filteringThreads_)
        {
            if (filterThread)
            {
                auto prompts = filterThread->getAcceptedPrompts();
                allPrompts.insert(allPrompts.end(),
                                  std::make_move_iterator(prompts.begin()),
                                  std::make_move_iterator(prompts.end()));
            }
        }

        return allPrompts;
    }

    std::vector<ScoredPrompt> GeneratorThread::getFilteredPromptsForProfile(int32_t profileId)
    {
        auto it = filteringThreads_.find(profileId);
        if (it != filteringThreads_.end() && it->second)
        {
            return it->second->getAcceptedPrompts();
        }
        return {};
    }

    std::vector<std::string> GeneratorThread::formatPrompts(const std::vector<ScoredPrompt> &prompts) const
    {
        std::vector<std::string> out;
        out.reserve(prompts.size());

        // Group prompts by profileId for efficient formatting
        std::unordered_map<int32_t, std::vector<const ScoredPrompt *>> promptsByProfile;
        for (const auto &p : prompts)
        {
            if (p.prompt)
            {
                promptsByProfile[p.prompt->profileId].push_back(&p);
            }
        }

        // Format each group with the appropriate generator
        for (const auto &[profileId, profilePrompts] : promptsByProfile)
        {
            auto compIt = profileComponents_.find(profileId);
            if (compIt == profileComponents_.end())
            {
                std::cerr << "Warning: No components found for profile ID " << profileId << std::endl;
                continue;
            }

            const auto &components = compIt->second;
            GeneratorModelSettings settings = components->modelSettings;

            Generator generator(*components->loraController,
                                *components->tagList,
                                *components->tokenizer,
                                components->generatorConfigPath,
                                settings,
                                components->clipEmbedCache.get());

            for (const auto *p : profilePrompts)
            {
                out.push_back(generator.formatPrompt(*p));
            }
        }

        return out;
    }

    bool GeneratorThread::initProfileComponents(int32_t profileId)
    {
        std::lock_guard<std::mutex> lock(componentsMutex_);

        if (profileComponents_.find(profileId) != profileComponents_.end())
        {
            return true; // Already initialized
        }

        // Find the profile by profileId
        const ModelProfile *profile = modelsConfig_.getProfileByProfileId(profileId);
        if (!profile)
        {
            std::cerr << "Profile ID " << profileId << " not found in configuration" << std::endl;
            return false;
        }

        try
        {
            auto components = std::make_unique<ProfileComponents>();

            // Store model settings
            components->modelSettings.profileId = profile->profileId;
            components->modelSettings.modelId = profile->modelId;
            components->modelSettings.modelName = profile->name;
            components->modelSettings.modelType = profile->modelType;
            components->modelSettings.generationMode = profile->effectiveGenerationMode();
            components->modelSettings.minSteps = profile->minSteps;
            components->modelSettings.maxSteps = profile->maxSteps;
            components->modelSettings.useLoras = profile->useLoras;
            components->generatorConfigPath = profile->generatorConfigPath;

            // Initialize tokenizer with model-specific strip parentheses setting
            components->tokenizer = std::make_unique<Tokenizer>(
                modelsConfig_.tokenizerPath, ',', profile->stripParentheses);
            std::cout << "Model '" << profile->name << "': Loaded tokenizer (stripParentheses="
                      << (profile->stripParentheses ? "true" : "false") << ")" << std::endl;

            // Load Tag_dict for this model (optional for clip_embed)
            components->tagDict = std::make_shared<Tag_dict>();
            if (!profile->dictPath.empty())
            {
                if (!std::filesystem::exists(profile->dictPath))
                {
                    throw std::runtime_error("Dict file not found: " + profile->dictPath);
                }
                if (!components->tagDict->loadFromJsonInstance(profile->dictPath))
                {
                    throw std::runtime_error("Failed to load tag dictionary: " + profile->dictPath);
                }
                std::cout << "Model '" << profile->name << "': Loaded dictionary from " << profile->dictPath << std::endl;
            }
            else
            {
                std::cout << "Model '" << profile->name << "': No tag dictionary" << std::endl;
            }

            // Load ClipEmbedCache before Tag_list so expert-dict profiles can precompute CLIP tag IDs.
            if (profile->usesClipScoring())
            {
                if (!profile->tagVocabPath.empty() && !std::filesystem::exists(profile->tagVocabPath))
                {
                    throw std::runtime_error("Tag vocab file not found: " + profile->tagVocabPath);
                }
                if (!std::filesystem::exists(profile->clipCachePath))
                {
                    throw std::runtime_error("CLIP cache file not found: " + profile->clipCachePath);
                }
                components->clipEmbedCache = std::make_unique<ClipEmbedCache>(
                    profile->tagVocabPath, profile->clipCachePath);
                std::cout << "Model '" << profile->name << "': Loaded CLIP embed cache ("
                          << components->clipEmbedCache->size() << " embeddings, dim="
                          << components->clipEmbedCache->getEmbedDim() << ")" << std::endl;

                if (components->modelSettings.generationMode == ModelProfile::GENERATION_MODE_EXPERT_DICT)
                {
                    if (profile->dictPath.empty() || !components->tagDict)
                    {
                        throw std::runtime_error(
                            "Profile '" + profile->name + "': generation_mode='expert_dict' requires a non-empty dict");
                    }
                    const std::vector<std::string> missing = collectMissingClipVocabTags(
                        *components->tagDict,
                        *components->clipEmbedCache);
                    if (!missing.empty())
                    {
                        throw std::runtime_error(
                            "Profile '" + profile->name + "': expert dict contains " + std::to_string(missing.size()) +
                            " tag(s) missing from CLIP vocab/cache. First entries: " + formatMissingTagsPreview(missing));
                    }
                }
            }
            // Initialize tag list with tokenizer for precomputed tokens and optional CLIP tag IDs.
            components->tagList = std::make_unique<Tag_list>(
                components->tokenizer.get(),
                components->tagDict.get(),
                components->clipEmbedCache.get());

            // Initialize LoRA controller if this model uses LoRAs
            if (profile->useLoras && !profile->lorasPath.empty())
            {
                if (!std::filesystem::exists(profile->lorasPath))
                {
                    throw std::runtime_error("LoRAs file not found: " + profile->lorasPath);
                }

                std::unique_ptr<Lora_controller> lc{init_lora_controller_from_json(profile->lorasPath)};
                lc->precompute_all_tokens(components->tokenizer.get());
                components->loraController = std::move(lc);
                std::cout << "Model '" << profile->name << "': Loaded LoRAs from " << profile->lorasPath << std::endl;
            }
            else
            {
                // Create empty LoRA controller for models without LoRAs
                components->loraController = std::make_unique<Lora_controller>();
                std::cout << "Model '" << profile->name << "': No LoRAs (useLoras=false)" << std::endl;
            }
            std::cout << "Profile '" << profile->name << "' (profileId=" << profileId
                      << ", modelId=" << profile->modelId << "): Components initialized" << std::endl;
            profileComponents_[profileId] = std::move(components);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to initialize components for profile " << profileId << ": " << e.what() << std::endl;
            return false;
        }
    }

    void GeneratorThread::generatorThreadFunction(int threadId)
    {
        std::cout << "Generator thread " << threadId << " started" << std::endl;

        // Seed the random generator differently for each thread
        srand(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) + threadId);

        // Create per-thread components for each profile
        // Tag_list and Lora_controller have mutable state and are not thread-safe
        std::unordered_map<int32_t, std::unique_ptr<Tag_list>> perThreadTagLists;
        std::unordered_map<int32_t, std::unique_ptr<Lora_controller>> perThreadLoraControllers;

        {
            std::lock_guard<std::mutex> lock(componentsMutex_);
            for (const auto &[profileId, components] : profileComponents_)
            {
                // Each thread needs its own Tag_list instance
                perThreadTagLists[profileId] = std::make_unique<Tag_list>(
                    components->tokenizer.get(),
                    components->tagDict.get(),
                    components->clipEmbedCache.get());

                // Each thread needs its own Lora_controller instance (if profile uses LoRAs)
                const ModelProfile *profile = modelsConfig_.getProfileByProfileId(profileId);
                if (profile && profile->useLoras && !profile->lorasPath.empty())
                {
                    std::unique_ptr<Lora_controller> lc{init_lora_controller_from_json(profile->lorasPath)};
                    lc->precompute_all_tokens(components->tokenizer.get());
                    perThreadLoraControllers[profileId] = std::move(lc);
                }
                else
                {
                    perThreadLoraControllers[profileId] = std::make_unique<Lora_controller>();
                }
            }
        }

        // Build per-profile generators for this thread using per-thread components
        std::unordered_map<int32_t, std::unique_ptr<Generator>> generators;

        {
            std::lock_guard<std::mutex> lock(componentsMutex_);
            for (const auto &[profileId, components] : profileComponents_)
            {
                generators[profileId] = std::make_unique<Generator>(
                    *perThreadLoraControllers[profileId], // Use per-thread Lora_controller
                    *perThreadTagLists[profileId],        // Use per-thread Tag_list
                    *components->tokenizer,
                    components->generatorConfigPath,
                    components->modelSettings,
                    components->clipEmbedCache.get()); // nullptr for one-hot models
            }
        }

        if (generators.empty())
        {
            std::cerr << "Thread " << threadId << ": No generators available, exiting" << std::endl;
            return;
        }

        // Round-robin counter for profile selection
        size_t profileIndex = static_cast<size_t>(threadId) % profileIds_.size();
        int promptsRemainingForProfile = std::max(runtimeConfig_.profileBurstSize, 1);

        while (running_)
        {
            try
            {
                int32_t currentProfileId = profileIds_[profileIndex];
                auto genIt = generators.find(currentProfileId);
                if (genIt == generators.end())
                {
                    profileIndex = (profileIndex + 1) % profileIds_.size();
                    promptsRemainingForProfile = std::max(runtimeConfig_.profileBurstSize, 1);
                    continue;
                }

                std::unique_ptr<Prompt> prompt = genIt->second->generatePrompt();

                if (!prompt)
                    continue;

                scoringThreadPool_->queueBatch(std::move(prompt), threadId);
                promptsRemainingForProfile--;
                if (promptsRemainingForProfile <= 0)
                {
                    profileIndex = (profileIndex + 1) % profileIds_.size();
                    promptsRemainingForProfile = std::max(runtimeConfig_.profileBurstSize, 1);
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Thread " << threadId << ": Error generating prompt: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        std::cout << "Generator thread " << threadId << " exiting" << std::endl;
    }

} // namespace generator