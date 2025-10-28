#pragma once

#include <chrono>
#include <condition_variable>
#include <atomic>
#include <array>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "evaluation/filtering_thread.hpp"
#include "evaluation/scoring_engine.hpp"
#include "utils/blocking_bounded_queue.hpp"
#include "utils/types.hpp"

namespace generator
{
    using PromptQueue = BlockingBoundedQueue<std::unique_ptr<Prompt>>;

    // FD
    class FilteringThread;

    /**
     * @brief Configuration for scoring a specific SD model
     */
    struct ModelScoringConfig
    {
        int32_t profileId{0};      // Unique profile index (for internal routing/keying)
        int32_t modelId{0};        // SD model ID (for ONNX model input, can be shared)
        std::string onnxModelPath;
        std::string clipCachePath;   // non-empty for clip_embed profiles
        float scoreThreshold{0.8f};
        int preferredDeviceId{-1}; // -1 = auto-assign based on thread
        unsigned batchSize{0};     // 0 = use default
    };

    /**
     * @brief Individual scoring thread with multi-model support
     *
     * Each ScoringThread owns one ScoringEngine per profile, allowing it to
     * score prompts for any profile. Prompts are grouped by profileId and scored
     * with the appropriate engine.
     */
    class ScoringThread
    {
    public:
        using CandidateQueue = BlockingBoundedQueue<ScoredPrompt>;
        /**
         * @brief Constructs a new ScoringThread with multi-model support
         * @param modelConfigs Per-model scoring configurations
         * @param assignedQueues Input queues from generator threads
         * @param threadId Unique thread identifier
         * @param deviceId CUDA device identifier (used when model config has preferredDeviceId=-1)
         */
        ScoringThread(const std::vector<ModelScoringConfig> &modelConfigs,
                      std::vector<PromptQueue *> assignedQueues,
                      int threadId = 0,
                      int deviceId = 0);
        ~ScoringThread();
        /**
         * @brief Sets the filtering thread that will receive scored prompts (single filter thread, legacy)
         */
        void setFilterThread(FilteringThread *filterThread);
        /**
         * @brief Sets multiple filtering threads for multi-profile routing
         * @param filterThreads Vector of filtering threads, indexed by profileId
         */
        void setFilterThreads(const std::vector<FilteringThread *> &filterThreads);
        void stopAndJoin();
        void notifyInputAvailable();

    private:
        struct BatchItem
        {
            PromptBatch batch;
            std::vector<std::unique_ptr<Prompt>> originalPrompts;
        };
        // main processing loop
        void processLoop();
        // Per-profile scoring engines and thresholds (keyed by profileId)
        std::unordered_map<int32_t, std::unique_ptr<ScoringEngine>> engines_;
        std::unordered_map<int32_t, float> scoreThresholds_;
        std::unordered_map<int32_t, unsigned> maxBatchSizes_;
        std::thread workerThread_;
        std::atomic<bool> running_{true};
        int threadId_;
        int deviceId_;
        std::vector<PromptQueue *> assignedQueues_;
        std::condition_variable inputCv_;
        std::mutex inputMutex_;
        std::atomic<int64_t> pendingPromptCount_{0};
        // Per-profile filtering threads and output queues (SPSC)
        std::vector<FilteringThread *> filterThreads_; // indexed by profileId
        std::vector<CandidateQueue *> outQueues_;      // indexed by profileId
        unsigned globalMaxBatchSize_{DEFAULT_SCORING_BATCH_SIZE};
        std::chrono::milliseconds batchFillWindow_{5};
        std::chrono::milliseconds metricsLogInterval_{2000};
        bool metricsEnabled_{false};
        struct RuntimeMetrics
        {
            uint64_t totalProcessed{0};
            uint64_t totalAccepted{0};
            uint64_t batchesProcessed{0};
            uint64_t preProfileBatchSamples{0};
            uint64_t postProfileBatchSamples{0};
            uint64_t batchFillWaitCount{0};
            uint64_t batchFillWaitNs{0};
            uint64_t inputWaitCount{0};
            uint64_t inputWaitNs{0};
            uint64_t inputWakeups{0};
            uint64_t inputWakeNoWork{0};
            size_t maxInputQueueDepth{0};
            size_t maxOutputQueueDepth{0};
            std::array<uint64_t, 8> preProfileBatchHistogram{};
            std::array<uint64_t, 8> postProfileBatchHistogram{};
        } metrics_{};
        // Group prompts by profileId and process each group
        void processPromptsByProfile(std::vector<std::unique_ptr<Prompt>> &prompts);
        BatchItem prepareBatch(std::vector<std::unique_ptr<Prompt>> &prompts);
        void recordBatchHistogram(std::array<uint64_t, 8> &histogram, size_t size);
        void logMetrics(const std::unordered_map<int32_t, size_t> &processedByModel,
                        const std::chrono::steady_clock::time_point &startTime,
                        std::chrono::steady_clock::time_point &lastPrint);
    };

    /**
     * @brief Thread pool manager for multiple scoring threads with multi-model support
     */
    class ScoringThreadPool
    {
    public:
        /**
         * @brief Constructs a new ScoringThreadPool with multi-model support
         * @param modelConfigs Per-model scoring configurations
         * @param numGeneratorThreads Number of generator threads (determines queue count)
         * @param numThreads Number of scoring threads to create (default: 3)
         * @param numGpus Number of GPUs to distribute threads across (default: 1)
         */
        ScoringThreadPool(const std::vector<ModelScoringConfig> &modelConfigs,
                          int numGeneratorThreads,
                          int numThreads = 3,
                          int numGpus = 1);
        ~ScoringThreadPool();
        /**
         * @brief Queues a prompt for scoring, assigned by generator thread ID
         * @param prompt The prompt to score
         * @param generatorThreadId ID of the generator thread (for deterministic assignment)
         * @return true if the prompt was queued successfully
         */
        bool queueBatch(std::unique_ptr<Prompt> prompt, int generatorThreadId);
        /**
         * @brief Sets the filtering thread that will receive scored prompts (legacy single thread)
         */
        void setFilterThread(FilteringThread *filterThread);
        /**
         * @brief Sets multiple filtering threads for multi-profile routing
         * @param filterThreads Vector of filtering threads, indexed by profileId
         */
        void setFilterThreads(const std::vector<FilteringThread *> &filterThreads);
        void stopAndJoin();
        void closeInputQueues();
        int getNumThreads() const { return numThreads_; }
        int getScoringThreadIndex(int generatorThreadId) const;

    private:
        int numThreads_;
        int numGeneratorThreads_;
        int numGpus_;
        std::vector<ModelScoringConfig> modelConfigs_;
        std::vector<std::unique_ptr<ScoringThread>> scoringThreads_;
        std::vector<std::unique_ptr<PromptQueue>> queues_;
    };

} // namespace generator