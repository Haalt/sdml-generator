#include "generator/generator_thread.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include "generator/generator.h"
#include "utils/types.hpp"

namespace generator
{

    GeneratorThread::GeneratorThread(GeneratorConfig config)
        : config_(std::move(config))
    {
        try
        {
            // create scoring thread pool with 2 threads
            scoringThreadPool_ = std::make_unique<ScoringThreadPool>(
                config_.modelPath,
                config_.scoreThreshold,
                config_.numGeneratorThreads,
                2); // 2 scoring thread. TODO: move somewhere else
            std::cout << "Scoring thread pool initialized" << std::endl;

            // create filtering thread and connect
            FilteringThread::Config filterCfg;
            filterCfg.scoreThreshold = config_.scoreThreshold;
            filterCfg.loraCap = 5; // Maximum prompts per individual LoRA. TODO: move somewhere else
            filterCfg.penaltyPerUse = 0.02f;

            filteringThread_ = std::make_unique<FilteringThread>(filterCfg);

            std::cout << "Filtering thread initialized and linked with: " << std::endl;
            std::cout << "  - Score threshold: " << filterCfg.scoreThreshold << std::endl;
            std::cout << "  - LoRA cap per LoRA: " << filterCfg.loraCap << std::endl;

            // connect scoring thread pool to filtering thread
            scoringThreadPool_->setFilterThread(filteringThread_.get());
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to initialize threads: " << e.what() << std::endl;
            throw;
        }

        generatorComponents_.resize(config_.numGeneratorThreads);

        // start worker threads
        std::cout << "Starting " << config_.numGeneratorThreads << " generator threads..." << std::endl;
        for (int i = 0; i < config_.numGeneratorThreads; i++)
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

        for (auto &thread : generatorThreads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        generatorThreads_.clear();

        // stop filtering thread first to ensure it no longer accesses queues
        if (filteringThread_)
        {
            filteringThread_->stopAndJoin();
        }

        // Now it is safe to stop the scoring thread pool (destroys their queues)
        if (scoringThreadPool_)
        {
            scoringThreadPool_->stopAndJoin();
        }
    }

    size_t GeneratorThread::getFilteredPromptsSize(void) const
    {
        if (filteringThread_)
            return filteringThread_->acceptedCountAlive();
        return 0;
    }

    std::vector<ScoredPrompt> GeneratorThread::getFilteredPrompts()
    {
        if (filteringThread_)
        {
            return filteringThread_->getAcceptedPrompts();
        }

        return std::vector<ScoredPrompt>();
    }

    std::vector<std::string> GeneratorThread::formatPrompts(const std::vector<ScoredPrompt> &prompts) const
    {
        std::vector<std::string> out;
        out.reserve(prompts.size());

        // cheap: assumes we have at least 1 generator components if prompts were generated
        auto &components = generatorComponents_[0u];

        Generator generator(*components->loraController,
                            *components->tagList,
                            *components->tokenizer,
                            config_.generatorConfigPath);

        for (const auto &p : prompts)
        {
            out.push_back(generator.formatPrompt(p));
        }

        return out;
    }

    bool GeneratorThread::initGeneratorComponents(int threadId)
    {
        std::lock_guard<std::mutex> lock(componentsMutex_);

        if (!generatorComponents_[threadId])
        {
            try
            {
                generatorComponents_[threadId] = std::make_unique<GeneratorComponents>();

                // Initialize tokenizer
                generatorComponents_[threadId]->tokenizer = std::make_unique<Tokenizer>(config_.tokenizerPath);
                std::cout << "Thread " << threadId << ": Loaded tokenizer from "
                          << config_.tokenizerPath << std::endl;

                // init tag list with tokenizer for precomputed tokens
                generatorComponents_[threadId]->tagList = std::make_unique<Tag_list>(generatorComponents_[threadId]->tokenizer.get());

                std::unique_ptr<Lora_controller> lc{init_lora_controller_from_json(config_.lorasJsonPath)};

                // Initialize precomputed tokens for LoRAs
                lc->precompute_all_tokens(generatorComponents_[threadId]->tokenizer.get());

                generatorComponents_[threadId]
                    ->loraController = std::move(lc);

                std::cout << "Thread " << threadId << ": Generator components initialized" << std::endl;
                return true;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Thread " << threadId << ": Failed to initialize generator components: " << e.what() << std::endl;
                generatorComponents_[threadId].reset();
                return false;
            }
        }

        return true;
    }

    void GeneratorThread::generatorThreadFunction(int threadId)
    {
        std::cout << "Generator thread " << threadId << " started" << std::endl;

        // initialize the generator components for THIS thread
        if (!initGeneratorComponents(threadId))
        {
            std::cerr << "Thread " << threadId << ": Failed to initialize generator components, exiting" << std::endl;
            return;
        }

        // seed the random generator differently for each thread
        // TODO: investigate
        srand(std::chrono::system_clock::now().time_since_epoch().count() + threadId);

        auto &components = generatorComponents_[threadId];

        Generator gen(*components->loraController,
                      *components->tagList,
                      *components->tokenizer,
                      config_.generatorConfigPath);

        while (running_)
        {
            try
            {
                std::unique_ptr<Prompt> prompt = gen.generatePrompt();

                if (!prompt)
                    continue;

                scoringThreadPool_->queueBatch(std::move(prompt), threadId);
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