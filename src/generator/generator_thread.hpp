#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/model_profile.hpp"
#include "evaluation/filtering_thread.hpp"
#include "evaluation/scoring_thread.hpp"
#include "generator/generator.h"
#include "models/Lora_controller.h"
#include "models/Tag_list.h"
#include "models/clip_embed_cache.hpp"
#include "tokenizer/tokenizer.hpp"
#include "utils/types.hpp"

namespace generator
{
    /**
     * @brief Runtime configuration for the generator thread pool
     *
     * Contains threading parameters that are not part of the models config.
     */
    struct GeneratorRuntimeConfig
    {
        int numGeneratorThreads = 0; // 0 -> auto (based on CPU cores)
        int numGpus = 0;             // 0 -> auto (cudaGetDeviceCount)
        int numScoringThreads = 0;   // 0 -> auto (balanced per GPU)
        int loraCap = 5;             // Maximum prompts per LoRA in filtering
        int profileBurstSize = 0;    // 0 -> auto (favor homogeneous scoring batches)
    };

    class GeneratorThread
    {
    public:
        /**
         * @brief Constructs a GeneratorThread with multi-model support
         * @param modelsConfig Configuration containing all model profiles
         * @param runtimeConfig Runtime threading parameters
         */
        GeneratorThread(const ModelsConfig &modelsConfig, GeneratorRuntimeConfig runtimeConfig = {});
        ~GeneratorThread();

        void stopAndJoin();

        // Get processed prompts that meet the threshold and filtering rules (aggregated from all models)
        std::vector<ScoredPrompt> getFilteredPrompts();
        size_t getFilteredPromptsSize(void) const;

        // Get filtered prompts for a specific profile
        std::vector<ScoredPrompt> getFilteredPromptsForProfile(int32_t profileId);

        std::vector<std::string> formatPrompts(const std::vector<ScoredPrompt> &prompts) const;

        // Get the models configuration
        const ModelsConfig &getModelsConfig() const { return modelsConfig_; }

    private:
        // Worker thread function for generation
        void generatorThreadFunction(int threadId);

        // Initialization of generator components for a specific profile
        bool initProfileComponents(int32_t profileId);

        std::vector<std::thread> generatorThreads_;
        std::atomic<bool> running_{true};

        // Scoring thread pool for model-based filtering (shared across all models)
        std::unique_ptr<ScoringThreadPool> scoringThreadPool_;

        // Per-profile filtering threads (key: profileId)
        std::unordered_map<int32_t, std::unique_ptr<FilteringThread>> filteringThreads_;

        ModelsConfig modelsConfig_;
        GeneratorRuntimeConfig runtimeConfig_;

        // Per-profile generator components (key: profileId)
        struct ProfileComponents
        {
            std::unique_ptr<Lora_controller> loraController;
            std::unique_ptr<Tag_list> tagList;
            std::unique_ptr<Tokenizer> tokenizer;
            std::shared_ptr<Tag_dict> tagDict;
            std::unique_ptr<ClipEmbedCache> clipEmbedCache; // Non-null for clip_embed models
            GeneratorModelSettings modelSettings;
            std::string generatorConfigPath;
        };
        std::unordered_map<int32_t, std::unique_ptr<ProfileComponents>> profileComponents_;
        std::mutex componentsMutex_;

        // List of profile IDs for round-robin iteration
        std::vector<int32_t> profileIds_;
    };

} // namespace generator