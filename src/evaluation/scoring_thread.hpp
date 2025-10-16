#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "evaluation/filtering_thread.hpp"
#include "evaluation/scoring_engine.hpp"
#include "utils/readerwriterqueue.h"
#include "utils/types.hpp"

namespace generator
{
    using PromptQueue = moodycamel::ReaderWriterQueue<std::unique_ptr<Prompt>>;

    // FD
    class FilteringThread;

    // individual scoring thread
    class ScoringThread
    {
    public:
        using CandidateQueue = moodycamel::ReaderWriterQueue<ScoredPrompt>;
        /**
         * @brief Constructs a new ScoringThread instance
         * @param modelPath Path to the ONNX model file
         * @param scoreThreshold Minimum score threshold for accepting prompts
         * @param threadId Unique thread identifier
         */
        ScoringThread(const std::string &modelPath, std::vector<PromptQueue *> assignedQueues, float scoreThreshold, int threadId = 0);

        ~ScoringThread();

        /**
         * @brief Sets the filtering thread that will receive scored prompts
         */
        void setFilterThread(FilteringThread *filterThread);

        void stopAndJoin();

    private:
        struct BatchItem
        {
            PromptBatch batch;
            std::vector<std::unique_ptr<Prompt>> originalPrompts;
        };

        // main processing loop
        void processLoop();

        std::unique_ptr<ScoringEngine> engine_;
        std::thread workerThread_;
        std::atomic<bool> running_{true};
        float scoreThreshold_;
        int threadId_; // Thread identifier for logging
        std::vector<PromptQueue *> assignedQueues_;

        FilteringThread *filterThread_{nullptr};

        // pointer to externally owned output queue (SPSC)
        CandidateQueue *outQueue_{nullptr};

        BatchItem processBatchIfReady(std::vector<std::unique_ptr<Prompt>> &prompts);
    };

    // thread pool manager for multiple scoring threads
    class ScoringThreadPool
    {
    public:
        /**
         * @brief Constructs a new ScoringThreadPool with multiple scoring threads
         * @param modelPath Path to the ONNX model file
         * @param scoreThreshold Minimum score threshold for accepting prompts
         * @param numThreads Number of scoring threads to create (default: 3)
         */
        ScoringThreadPool(const std::string &modelPath, float scoreThreshold, int numGeneratorThreads, int numThreads = 3);

        ~ScoringThreadPool();

        /**
         * @brief Queues a preprocessed batch for scoring, assigned by generator thread ID
         * @param batch The preprocessed batch to score
         * @param originalPrompts The original prompts corresponding to the batch
         * @param generatorThreadId ID of the generator thread (for deterministic assignment)
         * @return true if the batch was queued successfully
         */
        bool queueBatch(std::unique_ptr<Prompt> prompt, int generatorThreadId);

        /**
         * @brief Sets the filtering thread that will receive scored prompts
         */
        void setFilterThread(FilteringThread *filterThread);

        void stopAndJoin();

        int getNumThreads() const { return numThreads_; }

    private:
        int numThreads_;
        int numGeneratorThreads_;
        std::vector<std::unique_ptr<ScoringThread>> scoringThreads_;

        std::vector<std::unique_ptr<PromptQueue>> queues_;
    };

} // namespace generator