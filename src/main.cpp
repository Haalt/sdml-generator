#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>

#include <cuda_runtime.h>

#include "config/model_profile.hpp"
#include "generator/generator_thread.hpp"

namespace
{
    unsigned int detectCpuThreads()
    {
        unsigned int count = std::thread::hardware_concurrency();
        if (count == 0)
            count = 32;
        return count;
    }

    int detectGpuCount()
    {
        int gpuCount = 0;
        cudaError_t err = cudaGetDeviceCount(&gpuCount);
        if (err != cudaSuccess || gpuCount <= 0)
        {
            std::cerr << "Warning: cudaGetDeviceCount failed (" << cudaGetErrorString(err)
                      << "); defaulting to a single GPU." << std::endl;
            gpuCount = 1;
        }
        return gpuCount;
    }

    bool hasClipScoringProfile(const generator::ModelsConfig &modelsConfig)
    {
        return std::any_of(modelsConfig.profiles.begin(), modelsConfig.profiles.end(),
                           [](const generator::ModelProfile &profile)
                           { return profile.usesClipScoring(); });
    }

    bool usesOnlyClipRandomVocabGeneration(const generator::ModelsConfig &modelsConfig)
    {
        return !modelsConfig.profiles.empty() &&
               std::all_of(modelsConfig.profiles.begin(), modelsConfig.profiles.end(),
                           [](const generator::ModelProfile &profile)
                           { return profile.effectiveGenerationMode() == generator::ModelProfile::GENERATION_MODE_CLIP_RANDOM_VOCAB; });
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
        target = std::min(target, cpuBudget == 0 ? 8u : cpuBudget);
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

    int deriveProfileBurstSize(const generator::ModelsConfig &modelsConfig, bool clipEmbedWorkload)
    {
        if (modelsConfig.profiles.size() <= 1)
            return 1;
        return parsePositiveEnvInt("SDML_PROFILE_BURST", clipEmbedWorkload ? 8 : 1);
    }
} // namespace

// global flag for signal handling
volatile sig_atomic_t running = 1;

// signal handler for graceful shutdown
void signal_handler(int signal)
{
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    running = 0;
}

void printUsage(const char *programName)
{
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Multi-Model Mode (recommended):" << std::endl;
    std::cout << "  --models_config <path>   Path to models configuration JSON (contains all model profiles)" << std::endl;
    std::cout << std::endl;
    std::cout << "Common Options:" << std::endl;
    std::cout << "  --output <path>          Path to output file (default: prompts.txt)" << std::endl;
    std::cout << "  --output-format <fmt>    Output format: 'text' or 'binary' (default: text)" << std::endl;
    std::cout << "  --threads <int>          Number of generator threads (default: auto)" << std::endl;
    std::cout << "  --gpus <int>             Number of CUDA GPUs to use (default: auto)" << std::endl;
    std::cout << "  --scorers <int>          Number of scoring threads (default: auto)" << std::endl;
    std::cout << "  --lora-cap <int>         Maximum prompts per LoRA in filtering (default: 5)" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << programName << " --models_config models.json --output prompts.txt" << std::endl;
}

int main(int argc, char *argv[])
{
    // register signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "SDML Prompt Generator - Multi-Model" << std::endl;
    std::cout << "============================================" << std::endl;

    // CLI options
    std::string models_config_path;
    std::string output_file = "prompts.txt";
    std::string output_format = "text";
    const unsigned int cpu_threads_available = detectCpuThreads();
    const int detected_gpus = detectGpuCount();
    int num_generator_threads = 0; // 0 = auto
    int num_gpus = 0;              // 0 = auto
    int num_scorers = 0;           // 0 = auto
    int lora_cap = 5;

    // parse CLI args
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--models_config" && i + 1 < argc)
        {
            models_config_path = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (arg == "--output-format" && i + 1 < argc)
        {
            output_format = argv[++i];
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            num_generator_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--gpus" && i + 1 < argc)
        {
            num_gpus = std::stoi(argv[++i]);
        }
        else if (arg == "--scorers" && i + 1 < argc)
        {
            num_scorers = std::stoi(argv[++i]);
        }
        else if (arg == "--lora-cap" && i + 1 < argc)
        {
            lora_cap = std::stoi(argv[++i]);
        }
        else if (arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate required options
    if (models_config_path.empty())
    {
        std::cerr << "Error: --models_config is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    if (!std::filesystem::exists(models_config_path))
    {
        std::cerr << "Error: Models config file not found: " << models_config_path << std::endl;
        return 1;
    }

    // Load models configuration
    generator::ModelsConfig modelsConfig;
    try
    {
        modelsConfig = generator::ModelsConfig::loadFromJson(models_config_path);
        std::cout << "Loaded models configuration from: " << models_config_path << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading models config: " << e.what() << std::endl;
        return 1;
    }

    // Validate shared files exist
    if (!std::filesystem::exists(modelsConfig.tokenizerPath))
    {
        std::cerr << "Error: Tokenizer file not found: " << modelsConfig.tokenizerPath << std::endl;
        return 1;
    }

    // Validate profile files exist
    for (const auto &profile : modelsConfig.profiles)
    {
        if (!std::filesystem::exists(profile.onnxModelPath))
        {
            std::cerr << "Error: ONNX model file not found for profile '" << profile.name << "': " << profile.onnxModelPath << std::endl;
            return 1;
        }
        if (!profile.dictPath.empty() && !std::filesystem::exists(profile.dictPath))
        {
            std::cerr << "Error: Dict file not found for profile '" << profile.name << "': " << profile.dictPath << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(profile.generatorConfigPath))
        {
            std::cerr << "Error: Generator config not found for profile '" << profile.name << "': " << profile.generatorConfigPath << std::endl;
            return 1;
        }
        if (profile.useLoras && !profile.lorasPath.empty() && !std::filesystem::exists(profile.lorasPath))
        {
            std::cerr << "Error: LoRAs file not found for profile '" << profile.name << "': " << profile.lorasPath << std::endl;
            return 1;
        }
    }

    // Print configuration summary
    modelsConfig.printSummary();
    const bool clipScoringWorkload = hasClipScoringProfile(modelsConfig);
    const bool lightweightGenerationWorkload = usesOnlyClipRandomVocabGeneration(modelsConfig);

    // Set up runtime configuration
    generator::GeneratorRuntimeConfig runtimeConfig;
    runtimeConfig.numGeneratorThreads = num_generator_threads;
    runtimeConfig.numGpus = num_gpus > 0 ? num_gpus : detected_gpus;
    runtimeConfig.numScoringThreads = num_scorers;
    runtimeConfig.loraCap = lora_cap;
    runtimeConfig.profileBurstSize = deriveProfileBurstSize(modelsConfig, clipScoringWorkload);

    if (runtimeConfig.numScoringThreads <= 0)
        runtimeConfig.numScoringThreads = deriveScoringThreads(cpu_threads_available, runtimeConfig.numGpus, clipScoringWorkload);
    if (runtimeConfig.numGeneratorThreads <= 0)
        runtimeConfig.numGeneratorThreads = deriveGeneratorThreads(cpu_threads_available, runtimeConfig.numScoringThreads, lightweightGenerationWorkload);

    runtimeConfig.numScoringThreads = std::max(runtimeConfig.numScoringThreads, runtimeConfig.numGpus);
    runtimeConfig.numGeneratorThreads = std::max(runtimeConfig.numGeneratorThreads, runtimeConfig.numScoringThreads);

    std::cout << "\n===== Runtime Configuration =====" << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "Output format: " << output_format << std::endl;
    std::cout << "Generator threads: " << runtimeConfig.numGeneratorThreads << std::endl;
    std::cout << "Scoring threads: " << runtimeConfig.numScoringThreads << std::endl;
    std::cout << "GPUs: " << runtimeConfig.numGpus << std::endl;
    std::cout << "LoRA cap: " << runtimeConfig.loraCap << std::endl;
    std::cout << "Profile burst size: " << runtimeConfig.profileBurstSize << std::endl;
    std::cout << "CPU threads detected: " << cpu_threads_available << std::endl;
    std::cout << "Clip scoring auto-tuning: " << (clipScoringWorkload ? "enabled" : "disabled") << std::endl;
    std::cout << "Lightweight generation auto-tuning: " << (lightweightGenerationWorkload ? "enabled" : "disabled") << std::endl;
    std::cout << "=================================\n"
              << std::endl;

    try
    {
        // Create generator thread with multi-model configuration
        std::cout << "Initializing multi-model generator..." << std::endl;
        generator::GeneratorThread generator_thread(modelsConfig, runtimeConfig);

        std::cout << "Generator ready!" << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;

        // track time for status logging
        auto start_time = std::chrono::steady_clock::now();

        // main loop: periodic status logging
        while (running)
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            static int statCounter = 0;
            if (statCounter++ % 10 == 0)
            {
                size_t totalPrompts = generator_thread.getFilteredPromptsSize();
                std::cout << "[Main] Time: " << elapsed << "s, Accepted prompts: " << totalPrompts << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!running)
            {
                break;
            }
        }

        // generation stopped by signal, write final prompts
        std::cout << "\nStopping generator threads..." << std::endl;
        generator_thread.stopAndJoin();

        auto filteredPrompts = generator_thread.getFilteredPrompts();
        std::cout << "Processing " << filteredPrompts.size() << " prompts for output..." << std::endl;

        if (filteredPrompts.empty())
        {
            std::cout << "No prompts generated to save." << std::endl;
            return 0;
        }

        // Apply per-profile maxOutputPrompts limit: keep highest-scoring prompts
        {
            // Group prompts by profileId (each profile has its own limit)
            std::unordered_map<int32_t, std::vector<generator::ScoredPrompt>> byProfile;
            for (auto &p : filteredPrompts)
            {
                if (p.prompt)
                    byProfile[p.prompt->profileId].push_back(std::move(p));
            }
            filteredPrompts.clear();
            for (auto &[profileId, profilePrompts] : byProfile)
            {
                const auto *profile = modelsConfig.getProfileByProfileId(profileId);
                if (profile && profile->maxOutputPrompts > 0 &&
                    profilePrompts.size() > static_cast<size_t>(profile->maxOutputPrompts))
                {
                    // Sort descending by score and keep top N
                    std::partial_sort(profilePrompts.begin(),
                                      profilePrompts.begin() + profile->maxOutputPrompts,
                                      profilePrompts.end(),
                                      [](const generator::ScoredPrompt &a, const generator::ScoredPrompt &b)
                                      { return a.score > b.score; });
                    profilePrompts.resize(static_cast<size_t>(profile->maxOutputPrompts));
                    std::cout << "Profile '" << profile->name << "' (profileId=" << profileId
                              << "): truncated to " << profile->maxOutputPrompts
                              << " prompts (maxOutputPrompts)" << std::endl;
                }
                for (auto &p : profilePrompts)
                    filteredPrompts.push_back(std::move(p));
            }
        }

        // Count prompts per profile
        std::unordered_map<int32_t, size_t> promptsPerProfile;
        for (const auto &p : filteredPrompts)
        {
            if (p.prompt)
            {
                promptsPerProfile[p.prompt->profileId]++;
            }
        }

        std::cout << "\n===== Per-Profile Statistics =====" << std::endl;
        for (const auto &profile : modelsConfig.profiles)
        {
            size_t count = promptsPerProfile[profile.profileId];
            std::cout << "  " << profile.name << " (profileId=" << profile.profileId
                      << ", modelId=" << profile.modelId << "): " << count << " prompts" << std::endl;
        }
        std::cout << "  Total: " << filteredPrompts.size() << " prompts" << std::endl;
        std::cout << "=================================" << std::endl;

        // Shuffle prompts
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(filteredPrompts.begin(), filteredPrompts.end(), gen);

        // Calculate and log min score
        float min_score = 1.0f;
        for (const auto &p : filteredPrompts)
        {
            if (p.score < min_score)
                min_score = p.score;
        }
        std::cout << "Minimum score of saved prompts: " << min_score << std::endl;

        // Write to output
        if (output_format == "binary")
        {
            std::ofstream out_file(output_file, std::ios::out | std::ios::binary);
            if (!out_file.is_open())
            {
                std::cerr << "Error: Failed to open output file: " << output_file << std::endl;
                return 1;
            }

            std::cout << "Writing binary output to " << output_file << std::endl;

            for (const auto &sp : filteredPrompts)
            {
                const auto &p = sp.prompt;
                if (!p)
                    continue;

                out_file.write(reinterpret_cast<const char *>(&sp.score), sizeof(float));

                // Write model ID
                out_file.write(reinterpret_cast<const char *>(&p->modelId), sizeof(int32_t));

                // Helper to write vectors
                auto write_vec_i64 = [&](const std::vector<int64_t> &v)
                {
                    uint64_t sz = v.size();
                    out_file.write(reinterpret_cast<const char *>(&sz), sizeof(uint64_t));
                    if (sz > 0)
                        out_file.write(reinterpret_cast<const char *>(v.data()), sz * sizeof(int64_t));
                };
                auto write_vec_float = [&](const std::vector<float> &v)
                {
                    uint64_t sz = v.size();
                    out_file.write(reinterpret_cast<const char *>(&sz), sizeof(uint64_t));
                    if (sz > 0)
                        out_file.write(reinterpret_cast<const char *>(v.data()), sz * sizeof(float));
                };

                write_vec_i64(p->tokens);
                write_vec_i64(p->loras);
                write_vec_float(p->weights);
                out_file.write(reinterpret_cast<const char *>(&p->cfgScale), sizeof(float));
                out_file.write(reinterpret_cast<const char *>(&p->sequenceLength), sizeof(int64_t));
                out_file.write(reinterpret_cast<const char *>(&p->numLoras), sizeof(int64_t));
                out_file.write(reinterpret_cast<const char *>(&p->sampler), sizeof(int64_t));
                out_file.write(reinterpret_cast<const char *>(&p->steps), sizeof(int64_t));
                out_file.write(reinterpret_cast<const char *>(&p->stepsLog), sizeof(float));
                out_file.write(reinterpret_cast<const char *>(&p->stepsBucket), sizeof(int64_t));
                out_file.write(reinterpret_cast<const char *>(&p->upscaler), sizeof(int64_t));

                uint8_t upEnabled = p->upscalerEnabled ? 1 : 0;
                out_file.write(reinterpret_cast<const char *>(&upEnabled), sizeof(uint8_t));

                out_file.write(reinterpret_cast<const char *>(&p->upscalerSteps), sizeof(int64_t));
                out_file.write(reinterpret_cast<const char *>(&p->denoisingStrength), sizeof(float));
            }
            out_file.close();
        }
        else
        {
            std::ofstream out_file(output_file, std::ios::app);
            if (!out_file.is_open())
            {
                std::cerr << "Error: Failed to open output file: " << output_file << std::endl;
                return 1;
            }

            std::vector<std::string> formattedPrompts = generator_thread.formatPrompts(filteredPrompts);
            std::cout << "Writing " << formattedPrompts.size() << " text prompts to " << output_file << std::endl;

            auto rng = std::default_random_engine{};
            std::shuffle(std::begin(formattedPrompts), std::end(formattedPrompts), rng);

            for (const auto &prompt : formattedPrompts)
            {
                out_file << prompt << std::endl;
            }
            out_file.close();
        }

        std::cout << "\nOutput written to: " << output_file << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Generator stopped successfully" << std::endl;
    return 0;
}
