#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "evaluation/filtering_thread.hpp"
#include "evaluation/scoring_thread.hpp"
#include "models/Lora_controller.h"
#include "models/Tag_list.h"
#include "tokenizer/Tokenizer.hpp"
#include "utils/types.hpp"

namespace generator
{

    struct GeneratorConfig
    {
        std::string modelPath;           // Path to ONNX model
        std::string tokenizerPath;       // Path to tokenizer file for scoring
        std::string lorasJsonPath;       // Path to LoRAs JSON file
        std::string generatorConfigPath; // Path to generator config JSON
        float scoreThreshold = 0.8f;     // Minimum score to keep a prompt
        int numGeneratorThreads = 4;     // Number of generator threads to use
    };

    class GeneratorThread
    {
    public:
        GeneratorThread(GeneratorConfig config);
        ~GeneratorThread();

        void stopAndJoin();

        // get processed prompts that meet the threshold and filtering rules
        std::vector<ScoredPrompt> getFilteredPrompts();
        size_t getFilteredPromptsSize(void) const;

        std::vector<std::string> formatPrompts(const std::vector<ScoredPrompt> &prompts) const;

    private:
        // worker thread function for generation
        void generatorThreadFunction(int threadId);

        // initialization of your actual generator components
        bool initGeneratorComponents(int threadId);

        std::vector<std::thread> generatorThreads_;
        std::atomic<bool> running_{true};

        // scoring thread pool for model-based filtering
        std::unique_ptr<ScoringThreadPool> scoringThreadPool_;

        std::unique_ptr<FilteringThread> filteringThread_;

        GeneratorConfig config_;

        // thread local generator components
        struct GeneratorComponents
        {
            std::unique_ptr<Lora_controller> loraController;
            std::unique_ptr<Tag_list> tagList;
            std::unique_ptr<Tokenizer> tokenizer;
        };
        std::vector<std::unique_ptr<GeneratorComponents>> generatorComponents_;
        std::mutex componentsMutex_;
    };

} // namespace generator