#include "evaluation/scoring_thread.hpp"
#include "utils/types.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace generator
{

    inline void brief_backoff(unsigned &spins)
    {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
        if (++spins > 256)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            spins = 0;
        }
    }

    ScoringThreadPool::ScoringThreadPool(const std::string &modelPath, float scoreThreshold, int numGeneratorThreads, int numThreads)
        : numThreads_(numThreads), numGeneratorThreads_(numGeneratorThreads)
    {
        std::cout << "Initializing scoring thread pool with " << numThreads_ << " threads" << "..." << std::endl;

        queues_.reserve(numGeneratorThreads);

        for (int i = 0; i < numGeneratorThreads; ++i)
        {
            queues_.emplace_back(std::make_unique<PromptQueue>(1 << 14));
        }

        // create the scoring threads
        scoringThreads_.reserve(numThreads_);
        for (int i = 0; i < numThreads_; ++i)
        {
            // Build the list of queues assigned to this scoring thread
            std::vector<PromptQueue *> assignedQueues;
            for (int genId = 0; genId < numGeneratorThreads_; ++genId)
            {
                if (genId % numThreads_ == i)
                {
                    assignedQueues.push_back(queues_[genId].get());
                }
            }

            try
            {
                auto thread = std::make_unique<ScoringThread>(modelPath, assignedQueues, scoreThreshold, i);
                scoringThreads_.push_back(std::move(thread));
                std::cout << "Scoring thread " << i << " initialized successfully with " << assignedQueues.size() << " queues" << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to initialize scoring thread " << i << ": " << e.what() << std::endl;
                throw;
            }
        }

        std::cout << "Scoring thread pool initialization completed" << std::endl;
    }

    ScoringThreadPool::~ScoringThreadPool()
    {
        stopAndJoin();
    }

    bool ScoringThreadPool::queueBatch(std::unique_ptr<Prompt> prompt, int generatorThreadId)
    {
        static thread_local unsigned backoff_spins = 0;

        while (!queues_[generatorThreadId]->try_enqueue(std::move(prompt)))
        {
            brief_backoff(backoff_spins);
        }
        return true;
    }

    void ScoringThreadPool::setFilterThread(FilteringThread *filterThread)
    {
        for (auto &thread : scoringThreads_)
            thread->setFilterThread(filterThread);
    }

    void ScoringThreadPool::stopAndJoin()
    {
        std::cout << "Stopping " << numThreads_ << " scoring threads..." << std::endl;
        for (auto &thread : scoringThreads_)
        {
            if (thread)
            {
                thread->stopAndJoin();
            }
        }
        scoringThreads_.clear();
        std::cout << "All scoring threads stopped" << std::endl;
    }

    ScoringThread::ScoringThread(const std::string &modelPath, std::vector<PromptQueue *> assignedQueues, float scoreThreshold, int threadId)
        : scoreThreshold_(scoreThreshold),
          threadId_(threadId),
          assignedQueues_(assignedQueues)
    {
        std::cout << "Initializing scoring thread " << threadId_ << "..." << std::endl;

        try
        {
            engine_ = std::make_unique<ScoringEngine>(modelPath, threadId);
            std::cout << "Scoring engine initialized successfully for thread " << threadId_ << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to load scoring engine for thread " << threadId_ << ": " << e.what() << std::endl;
            throw;
        }

        std::cout << "Starting scoring thread " << threadId_ << " worker..." << std::endl;
        workerThread_ = std::thread(&ScoringThread::processLoop, this);

        // create output queue
        outQueue_ = new CandidateQueue(1 << 16);
    }

    ScoringThread::~ScoringThread()
    {
        stopAndJoin();
        delete outQueue_;
    }

    void ScoringThread::setFilterThread(FilteringThread *filterThread)
    {
        filterThread_ = filterThread;
        if (filterThread_ && outQueue_)
        {
            filterThread_->registerQueue(outQueue_);
        }
    }

    void ScoringThread::stopAndJoin()
    {
        // signals the processing loop to stop
        running_ = false;

        if (workerThread_.joinable())
        {
            workerThread_.join();
        }
    }

    ScoringThread::BatchItem ScoringThread::processBatchIfReady(std::vector<std::unique_ptr<Prompt>> &prompts)
    {
        std::vector<std::unique_ptr<Prompt>> batchPrompts;

        batchPrompts.swap(prompts);

        PromptBatch promptBatch;
        promptBatch.batch_size = batchPrompts.size();
        promptBatch.sequence_length = MAX_SEQUENCE_LENGTH;

        // reserve space for all tensors
        size_t tokenTensorSize = batchPrompts.size() * MAX_SEQUENCE_LENGTH;
        size_t loraTensorSize = batchPrompts.size() * MAX_LORAS;

        // Reserve once to avoid re-allocations under growth
        if (promptBatch.token_indices.capacity() < tokenTensorSize)
            promptBatch.token_indices.reserve(tokenTensorSize);
        if (promptBatch.token_mask.capacity() < tokenTensorSize)
            promptBatch.token_mask.reserve(tokenTensorSize);
        if (promptBatch.lora_ids.capacity() < loraTensorSize)
            promptBatch.lora_ids.reserve(loraTensorSize);
        if (promptBatch.lora_weights.capacity() < loraTensorSize)
            promptBatch.lora_weights.reserve(loraTensorSize);
        if (promptBatch.cfg_scales.capacity() < batchPrompts.size())
            promptBatch.cfg_scales.reserve(batchPrompts.size());
        if (promptBatch.num_loras.capacity() < batchPrompts.size())
            promptBatch.num_loras.reserve(batchPrompts.size());
        if (promptBatch.sampler_ids.capacity() < batchPrompts.size())
            promptBatch.sampler_ids.reserve(batchPrompts.size());
        if (promptBatch.steps_log.capacity() < batchPrompts.size())
            promptBatch.steps_log.reserve(batchPrompts.size());
        if (promptBatch.steps_bucket.capacity() < batchPrompts.size())
            promptBatch.steps_bucket.reserve(batchPrompts.size());
        if (promptBatch.upscaler_ids.capacity() < batchPrompts.size())
            promptBatch.upscaler_ids.reserve(batchPrompts.size());
        if (promptBatch.up_has.capacity() < batchPrompts.size())
            promptBatch.up_has.reserve(batchPrompts.size());
        if (promptBatch.up_steps.capacity() < batchPrompts.size())
            promptBatch.up_steps.reserve(batchPrompts.size());
        if (promptBatch.denoise.capacity() < batchPrompts.size())
            promptBatch.denoise.reserve(batchPrompts.size());

        // Resize to logical size without releasing capacity
        promptBatch.token_indices.resize(tokenTensorSize);
        promptBatch.token_mask.resize(tokenTensorSize);
        promptBatch.lora_ids.resize(loraTensorSize);
        promptBatch.lora_weights.resize(loraTensorSize);
        promptBatch.cfg_scales.resize(batchPrompts.size());
        promptBatch.num_loras.resize(batchPrompts.size());
        promptBatch.sampler_ids.resize(batchPrompts.size());
        promptBatch.steps_log.resize(batchPrompts.size());
        promptBatch.steps_bucket.resize(batchPrompts.size());
        promptBatch.upscaler_ids.resize(batchPrompts.size());
        promptBatch.up_has.resize(batchPrompts.size());
        promptBatch.up_steps.resize(batchPrompts.size());
        promptBatch.denoise.resize(batchPrompts.size());

        for (size_t i = 0; i < batchPrompts.size(); ++i)
        {
            const auto &prompt = batchPrompts[i];

            for (size_t j = 0; j < prompt->tokens.size() && j < MAX_SEQUENCE_LENGTH; ++j)
            {
                size_t idx = i * MAX_SEQUENCE_LENGTH + j;
                promptBatch.token_indices[idx] = prompt->tokens[j];
                promptBatch.token_mask[idx] = 1.0f; // set mask to 1 for valid tokens
            }

            promptBatch.cfg_scales[i] = prompt->cfgScale / MAX_CFG_SCALE;                                    // normalize CFG scale
            promptBatch.num_loras[i] = static_cast<float>(prompt->numLoras) / static_cast<float>(MAX_LORAS); // normalize num LoRAs

            for (size_t j = 0; j < static_cast<size_t>(prompt->numLoras) && j < MAX_LORAS; ++j)
            {
                size_t idx = i * MAX_LORAS + j;
                promptBatch.lora_ids[idx] = prompt->loras[j];
                promptBatch.lora_weights[idx] = prompt->weights[j];
            }

            promptBatch.sampler_ids[i] = prompt->sampler;
            promptBatch.steps_log[i] = prompt->stepsLog;
            promptBatch.steps_bucket[i] = prompt->stepsBucket;
            promptBatch.upscaler_ids[i] = prompt->upscaler;
            promptBatch.up_has[i] = prompt->upscalerEnabled ? 1.0f : 0.0f;
            promptBatch.up_steps[i] = static_cast<float>(prompt->upscalerSteps) / 25.0f; // normalize to [0-1]
            promptBatch.denoise[i] = prompt->denoisingStrength;
        }

        BatchItem ret = {promptBatch, std::move(batchPrompts)};

        return ret;
    }

    void ScoringThread::processLoop()
    {
        std::cout << "Scoring thread " << threadId_ << " started" << std::endl;

        // stats tracking
        size_t totalProcessed = 0;
        size_t totalAccepted = 0;
        auto startTime = std::chrono::steady_clock::now();
        auto lastPrint = startTime;

        // Upper bound, actual batch may be smaller
        // constexpr unsigned max_batch_size = 4096; // TODO: size dynamicly based on VRAM usage
        constexpr unsigned max_batch_size = 8192;

        std::vector<std::unique_ptr<Prompt>> prompts_buffer;
        prompts_buffer.reserve(max_batch_size);

        while (running_)
        {
            static thread_local unsigned backoff_spins = 0;

            // collect prompts up to max_batch_size, but don't wait longer than 5 ms for new data
            auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);

            while (prompts_buffer.size() < max_batch_size)
            {
                bool gotItem = false;
                std::unique_ptr<Prompt> item;

                for (PromptQueue *q : assignedQueues_)
                {
                    if (q && q->try_dequeue(item))
                    {
                        gotItem = true;
                        prompts_buffer.emplace_back(std::move(item));
                        if (prompts_buffer.size() >= max_batch_size)
                            break; // buffer full
                    }
                }

                if (!gotItem)
                {
                    if (std::chrono::steady_clock::now() >= wait_deadline)
                    {
                        break; // timeout reached, process what we have
                    }
                    // pause to avoid busy waiting
                    brief_backoff(backoff_spins);
                }
            }

            // if we didn't collect anything, sleep a little and try again
            if (prompts_buffer.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            BatchItem batch = processBatchIfReady(prompts_buffer);

            if (!running_ && batch.originalPrompts.empty())
            {
                break;
            }

            std::vector<float> scores = engine_->scoreBatch(batch.batch);

            // send scored prompts to filtering thread
            for (size_t i = 0; i < scores.size(); ++i)
            {
                ScoredPrompt cand;
                cand.prompt = std::unique_ptr<Prompt>(std::move(batch.originalPrompts[i]));

                cand.score = scores[i];

                // try to enqueue; if full, busy wait briefly
                static thread_local unsigned backoff_spins_out = 0;
                while (!outQueue_ || !outQueue_->try_enqueue(std::move(cand)))
                {
                    if (!running_)
                    {
                        break;
                    }
                    brief_backoff(backoff_spins_out);
                }
                if (!running_)
                {
                    break;
                }
            }

            if (!running_)
            {
                break;
            }

            totalProcessed += scores.size();
            totalAccepted += std::count_if(scores.begin(), scores.end(),
                                           [this](float score)
                                           { return score >= scoreThreshold_; });

            // log periodic stats
            auto now = std::chrono::steady_clock::now();
            auto elapsedSecondsLastPrint = std::chrono::duration_cast<std::chrono::seconds>(now - lastPrint).count();
            auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();

            if (elapsedSecondsLastPrint > 1)
            {
                double acceptanceRate = totalProcessed > 0 ? (double)totalAccepted / totalProcessed * 100.0 : 0.0;
                double promptsPerSecond = elapsedSeconds > 0 ? (double)totalProcessed / elapsedSeconds : 0.0;

                std::cout << "Scoring thread " << threadId_ << " stats: "
                          << totalProcessed << " processed, "
                          << totalAccepted << " accepted ("
                          << std::fixed << std::setprecision(1) << acceptanceRate << "%), "
                          << std::fixed << std::setprecision(1) << promptsPerSecond << " prompts/sec"
                          << std::endl;

                lastPrint = now;
            }
        }

        std::cout << "Scoring thread " << threadId_ << " exiting" << std::endl;
    }

} // namespace generator
