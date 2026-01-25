#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "models/tag_dict.h"
#include "tokenizer/tokenizer.hpp"
#include "utils/blocking_bounded_queue.hpp"
#include "utils/types.hpp"

namespace generator
{
    struct TagIndex
    {
        uint64_t unique_mask_all{0};                        // which category bits are "unique categories"
        std::unordered_map<std::string, uint64_t> tag_mask; // tag -> category bit(s)
        std::array<int64_t, 64> bit_to_lora_id{};           // bit -> synthetic LoRA id (>0)
        // Per-subkey synthetic ids: each subkey under a unique category gets a stable id.
        // A single tag can belong to multiple synthetic families.
        std::unordered_map<std::string, std::vector<int64_t>> tag_to_lora_ids; // tag string -> synthetic family ids (>0)
        // Reverse index used for overlap-based disambiguation scoring.
        std::unordered_map<int64_t, std::unordered_set<std::string>> lora_id_to_tags; // synthetic family id -> full family tag set
        // Stable insertion rank for deterministic tie-breaking.
        std::unordered_map<int64_t, uint32_t> lora_rank;
        TagIndex() { bit_to_lora_id.fill(0); }
    };

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
        using CandidateQueue = BlockingBoundedQueue<ScoredPrompt>;

        struct Config
        {
            float scoreThreshold{0.8f};         // Minimum effective score to accept
            int loraCap{3};                     // Maximum prompts per individual LoRA
            float penaltyPerUse{0.02f};         // Penalty applied per prior usage of a LoRA - legacy
            size_t maxAcceptedPrompts{5000000}; // Hard cap on number of accepted prompts held in memory
            std::shared_ptr<Tag_dict> tagDict = nullptr; // null -> legacy LoRA mode
            bool useTagsAsLoras = true;
            bool scoreOnly{false};              // When true, skip LoRA diversity logic; accept all above threshold
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
        void notifyWorkAvailable();

        std::string stats() const;

        void setTokenizer(Tokenizer tok) { tokenizer_ = std::move(tok); }
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
        size_t nextQueueIndex_{0};
        std::thread workerThread_;
        std::atomic<bool> running_{true};
        std::condition_variable workCv_;
        mutable std::mutex workMutex_;
        std::atomic<uint64_t> workSignalCount_{0};
        bool metricsEnabled_{false};
        std::chrono::milliseconds metricsLogInterval_{2000};
        struct RuntimeMetrics
        {
            uint64_t totalProcessed{0};
            uint64_t totalAccepted{0};
            uint64_t wakeCount{0};
            uint64_t emptyWakeCount{0};
            uint64_t waitCount{0};
            uint64_t waitNs{0};
            size_t maxQueueDepth{0};
        } metrics_{};

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

        std::optional<TagIndex> tagIndex_;
        std::optional<Tokenizer> tokenizer_;
    };

} // namespace generator
