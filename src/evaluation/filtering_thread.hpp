#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/readerwriterqueue.h"
#include "utils/types.hpp"

namespace generator
{

    /**
     * @brief Thread-safe, asynchronous filter applying LoRA-diversity logic.
     *
     * Incoming prompts (with scores) are pushed via registerQueue() and fetched internally from those queues.
     * The internal worker thread consumes them, applies the refined acceptance / replacement algorithm and maintains an "accepted" list
     * The accepted prompts can be retrieved (typically when generation is stopped) with getAcceptedPrompts().
     */
    class FilteringThread
    {
    public:
        using CandidateQueue = moodycamel::ReaderWriterQueue<ScoredPrompt>;

        struct Config
        {
            float scoreThreshold{0.8f};       // Minimum effective score to accept
            int loraCap{3};                   // Maximum prompts per individual LoRA
            float penaltyPerUse{0.02f};       // Penalty applied per prior usage of a LoRA - legacy
            size_t maxAcceptedPrompts{50000}; // Hard cap on number of accepted prompts held in memory
        };

        explicit FilteringThread(const Config &cfg);
        ~FilteringThread();

        // Non copyable, non movable
        FilteringThread(const FilteringThread &) = delete;
        FilteringThread &operator=(const FilteringThread &) = delete;

        // register a new input queue (to be called once per scoring thread)
        void registerQueue(CandidateQueue *queue);

        std::vector<ScoredPrompt> getAcceptedPrompts() const;
        size_t acceptedCountAlive() const;

        void stopAndJoin();

        std::string stats() const;

    private:
        // Internal representation of an accepted prompt
        struct AcceptedPrompt
        {
            ScoredPrompt scored;
            std::vector<int64_t> loras; // LoRAs used by the prompt
        };

        // vector of SPSC queues (one per scoring thread)
        std::vector<CandidateQueue *> candidateQueues_;
        mutable std::mutex queuesMutex_;
        std::thread workerThread_;
        std::atomic<bool> running_{true};

        // state protected by mutex
        mutable std::mutex stateMutex_;
        // Prompt slots managed via stable IDs. loraUsage_ tracks counts among live slots
        std::unordered_map<int64_t, int> loraUsage_;       // LoRA -> count across live prompts
        std::unordered_map<int64_t, float> loraBestScore_; // LoRA -> best raw score observed in acceptedPrompts_

        // replace old auxiliary indices with stable ID system and support structures
        using PromptId = uint32_t;

        struct Slot
        {
            AcceptedPrompt ap;
            bool alive{false};
        };

        struct HeapEntry
        {
            float score;
            PromptId id;
        };

        struct HeapCmp
        {
            bool operator()(const HeapEntry &a, const HeapEntry &b) const { return a.score > b.score; }
        };

        // slab allocator of prompt slots (stable IDs)
        std::vector<Slot> slots_;
        std::vector<PromptId> freeList_;
        std::deque<PromptId> fifo_; // acceptance order for FIFO eviction

        // LoRA -> set of prompt IDs that currently contain it
        std::unordered_map<int64_t, std::unordered_set<PromptId>> loraToIds_;
        // LoRA -> min heap of weakest prompts (lazy deletion)
        std::unordered_map<int64_t, std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp>> loraMin_;

        // Helper allocation / indexing routines (stateMutex_ held)
        PromptId allocSlot();
        void markDead(PromptId id);
        void indexPrompt(PromptId id, const AcceptedPrompt &ap);

        Config config_;

        // worker loop
        void processLoop();

        bool isExceptional(const ScoredPrompt &cand, const std::vector<int64_t> &loras);
        std::vector<PromptId> selectReplacements(const ScoredPrompt &cand, const std::vector<int64_t> &loras);
    };

} // namespace generator
