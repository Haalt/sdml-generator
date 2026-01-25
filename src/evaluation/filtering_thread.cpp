#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>

#include "evaluation/filtering_thread.hpp"
#include "tokenizer/tokenizer.hpp"

namespace generator
{
    namespace
    {
        int parsePositiveEnvInt(const char *name, int fallback)
        {
            const char *env = std::getenv(name);
            if (env == nullptr || *env == '\0')
                return fallback;
            char *endPtr = nullptr;
            long parsed = std::strtol(env, &endPtr, 10);
            if (endPtr == env || *endPtr != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
                return fallback;
            return static_cast<int>(parsed);
        }

        bool parseEnvBool(const char *name, bool fallback)
        {
            const char *env = std::getenv(name);
            if (env == nullptr || *env == '\0')
                return fallback;
            return std::string(env) != "0";
        }
    }

    inline void for_each_set_bit(uint64_t mask, const std::function<void(uint8_t)> &fn)
    {
        while (mask)
        {
#if defined(_MSC_VER) && defined(__AVX2__)
            unsigned long bit;
            _BitScanForward64(&bit, mask);
            uint8_t b = static_cast<uint8_t>(bit);
#elif defined(__GNUC__) || defined(__clang__)
            uint8_t b = static_cast<uint8_t>(__builtin_ctzll(mask));
#else
            // portable fallback
            uint8_t b = 0;
            uint64_t t = mask;
            while ((t & 1ull) == 0ull)
            {
                ++b;
                t >>= 1;
            }
#endif
            fn(b);
            mask &= (mask - 1); // clear lowest set bit
        }
    }

    // Recursively collect leaf tags and attach the given category mask
    static void collect_tags_for_category(const std::shared_ptr<Node> &node,
                                          uint64_t cat_mask,
                                          std::unordered_map<std::string, uint64_t> &out)
    {
        if (!node)
            return;

        if (auto s = std::dynamic_pointer_cast<StringNode>(node))
        {
            for (const auto &tag : Tokenizer::splitTagString(s->value))
            {
                if (!tag.empty())
                {
                    out[tag] |= cat_mask;
                }
            }
            return;
        }
        if (auto l = std::dynamic_pointer_cast<ListNode>(node))
        {
            for (const auto &v : l->values)
            {
                for (const auto &tag : Tokenizer::splitTagString(v))
                {
                    if (!tag.empty())
                    {
                        out[tag] |= cat_mask;
                    }
                }
            }
            return;
        }
        if (auto m = std::dynamic_pointer_cast<MapNode>(node))
        {
            for (const auto &kv : m->children)
            {
                collect_tags_for_category(kv.second, cat_mask, out);
            }
            return;
        }
    }

    static bool has_family_id(const std::vector<int64_t> &values, int64_t familyId)
    {
        return std::find(values.begin(), values.end(), familyId) != values.end();
    }

    static void append_unique_family(std::vector<int64_t> &families, std::unordered_set<int64_t> &seen, int64_t familyId)
    {
        if (familyId != 0 && seen.insert(familyId).second)
        {
            families.push_back(familyId);
        }
    }

    static std::vector<int64_t> intersect_families(const std::vector<int64_t> &candidates, const std::unordered_set<int64_t> &contextFamilies)
    {
        std::vector<int64_t> narrowed;
        for (int64_t familyId : candidates)
        {
            if (contextFamilies.count(familyId))
            {
                narrowed.push_back(familyId);
            }
        }
        return narrowed;
    }

    static uint32_t get_family_rank(const TagIndex &tagIndex, int64_t familyId)
    {
        const auto it = tagIndex.lora_rank.find(familyId);
        if (it == tagIndex.lora_rank.end())
        {
            return std::numeric_limits<uint32_t>::max();
        }
        return it->second;
    }

    static double compute_overlap_score(const TagIndex &tagIndex, int64_t familyId, const std::vector<int64_t> &candidateFamilies)
    {
        const auto itFamilyTags = tagIndex.lora_id_to_tags.find(familyId);
        if (itFamilyTags == tagIndex.lora_id_to_tags.end() || itFamilyTags->second.empty())
        {
            return 0.0;
        }
        const std::unordered_set<std::string> &familyTags = itFamilyTags->second;
        std::unordered_set<std::string> otherTags;
        for (int64_t otherId : candidateFamilies)
        {
            if (otherId == familyId)
            {
                continue;
            }
            const auto itOther = tagIndex.lora_id_to_tags.find(otherId);
            if (itOther == tagIndex.lora_id_to_tags.end())
            {
                continue;
            }
            for (const auto &tag : itOther->second)
            {
                otherTags.insert(tag);
            }
        }
        if (otherTags.empty())
        {
            return 0.0;
        }
        size_t sharedCount = 0;
        for (const auto &tag : familyTags)
        {
            if (otherTags.count(tag))
            {
                ++sharedCount;
            }
        }
        return static_cast<double>(sharedCount) / static_cast<double>(familyTags.size());
    }

    static int64_t choose_family_by_overlap(const TagIndex &tagIndex, const std::vector<int64_t> &candidateFamilies)
    {
        if (candidateFamilies.empty())
        {
            return 0;
        }
        int64_t selected = candidateFamilies.front();
        double bestScore = -1.0;
        uint32_t bestRank = std::numeric_limits<uint32_t>::max();
        for (int64_t familyId : candidateFamilies)
        {
            const double score = compute_overlap_score(tagIndex, familyId, candidateFamilies);
            const uint32_t rank = get_family_rank(tagIndex, familyId);
            if (score > bestScore || (score == bestScore && (rank < bestRank || (rank == bestRank && familyId < selected))))
            {
                bestScore = score;
                bestRank = rank;
                selected = familyId;
            }
        }
        return selected;
    }

    struct TagResolutionResult
    {
        std::vector<int64_t> families;
        bool hadSubkeyMatches{false};
    };

    static TagResolutionResult resolve_prompt_families(const std::shared_ptr<Prompt> &prompt,
                                                       const TagIndex &tagIndex,
                                                       std::optional<Tokenizer> &tok)
    {
        TagResolutionResult result;
        if (!prompt || !tok.has_value())
        {
            return result;
        }
        struct AmbiguousTag
        {
            std::vector<int64_t> candidates;
            bool resolved{false};
        };
        std::unordered_set<std::string> seenTags;
        std::unordered_set<int64_t> seenFamilies;
        std::unordered_set<int64_t> contextFamilies;
        std::vector<AmbiguousTag> ambiguousTags;
        for (int64_t tokenId : prompt->tokens)
        {
            const auto tag = tok->tokenToText(tokenId);
            if (!tag || !seenTags.insert(*tag).second)
            {
                continue;
            }
            const auto itCandidates = tagIndex.tag_to_lora_ids.find(*tag);
            if (itCandidates == tagIndex.tag_to_lora_ids.end() || itCandidates->second.empty())
            {
                continue;
            }
            result.hadSubkeyMatches = true;
            const std::vector<int64_t> &candidates = itCandidates->second;
            if (candidates.size() == 1)
            {
                const int64_t familyId = candidates.front();
                append_unique_family(result.families, seenFamilies, familyId);
                contextFamilies.insert(familyId);
                continue;
            }
            ambiguousTags.push_back({candidates, false});
        }

        bool hasProgress = true;
        while (hasProgress)
        {
            hasProgress = false;
            for (auto &ambiguousTag : ambiguousTags)
            {
                if (ambiguousTag.resolved)
                {
                    continue;
                }
                std::vector<int64_t> narrowed = intersect_families(ambiguousTag.candidates, contextFamilies);
                if (narrowed.size() == 1)
                {
                    const int64_t selectedFamily = narrowed.front();
                    append_unique_family(result.families, seenFamilies, selectedFamily);
                    contextFamilies.insert(selectedFamily);
                    ambiguousTag.resolved = true;
                    hasProgress = true;
                }
            }
        }

        for (const auto &ambiguousTag : ambiguousTags)
        {
            if (ambiguousTag.resolved)
            {
                continue;
            }
            std::vector<int64_t> narrowed = intersect_families(ambiguousTag.candidates, contextFamilies);
            const std::vector<int64_t> &selectionPool = narrowed.empty() ? ambiguousTag.candidates : narrowed;
            const int64_t selectedFamily = choose_family_by_overlap(tagIndex, selectionPool);
            append_unique_family(result.families, seenFamilies, selectedFamily);
            contextFamilies.insert(selectedFamily);
        }
        return result;
    }

    // Build an immutable index from Tag_dict.
    // Filtering uses rules.filter_categories when present, otherwise it falls back to unique_categories.
    // - Legacy: filter categories map to category bits (tag_mask + bit_to_lora_id)
    // - New: each subkey under a filter category becomes its own synthetic LoRA id (tag_to_lora_ids)
    static generator::TagIndex build_tag_index(const std::shared_ptr<Tag_dict> &dict)
    {
        generator::TagIndex idx;
        if (!dict)
            return idx;

        const auto &reg = dict->getRegistry(); // CategoryRegistry
        const auto &rules = dict->getRules();
        const auto &filterCategories = rules.has_filter_categories ? rules.filter_categories : rules.unique_categories;

        // Stable, non-colliding base for synthetic subkey ids
        constexpr int64_t SUBKEY_ID_BASE = 1000000000LL; // 1e9 offset to avoid collisions with real LoRA ids
        int64_t subkeyCounter = 0;

        for (const auto &cat : filterCategories)
        {
            const std::shared_ptr<Node> categoryNode = dict->getNode(cat);
            if (!categoryNode)
                continue;

            auto itBit = reg.bit_by_category.find(cat);
            if (itBit == reg.bit_by_category.end())
                continue;

            const uint8_t bit = itBit->second;
            const uint64_t cat_mask = (uint64_t(1) << bit);
            idx.unique_mask_all |= cat_mask;

            // Collect every leaf tag under this category and map it to this category's bit (legacy category-based path)
            collect_tags_for_category(categoryNode, cat_mask, idx.tag_mask);

            auto registerSyntheticFamily = [&](const std::shared_ptr<Node> &familyNode)
            {
                const int64_t subkeyId = SUBKEY_ID_BASE + subkeyCounter;
                idx.lora_rank[subkeyId] = static_cast<uint32_t>(subkeyCounter);
                ++subkeyCounter;

                std::unordered_map<std::string, uint64_t> familyTags;
                collect_tags_for_category(familyNode, 0 /*unused in this map*/, familyTags);
                for (const auto &familyTag : familyTags)
                {
                    std::vector<int64_t> &ids = idx.tag_to_lora_ids[familyTag.first];
                    if (!has_family_id(ids, subkeyId))
                    {
                        ids.push_back(subkeyId);
                    }
                    idx.lora_id_to_tags[subkeyId].insert(familyTag.first);
                }
            };

            // New per-subkey path: assign a unique synthetic id for each immediate subkey under this category,
            // and map ALL leaf tags under that subkey to the same id (groups synonyms together)
            if (auto catMap = std::dynamic_pointer_cast<MapNode>(categoryNode))
            {
                for (const auto &kv : catMap->children)
                {
                    const std::string &subkey = kv.first;
                    registerSyntheticFamily(kv.second);
                    const int64_t subkeyId = SUBKEY_ID_BASE + (subkeyCounter - 1);

                    // Also map the subkey itself if it appears as a tag in the tokenizer vocabulary
                    if (!subkey.empty())
                    {
                        std::vector<int64_t> &ids = idx.tag_to_lora_ids[subkey];
                        if (!has_family_id(ids, subkeyId))
                        {
                            ids.push_back(subkeyId);
                        }
                        idx.lora_id_to_tags[subkeyId].insert(subkey);
                    }
                }
            }
            else if (auto catList = std::dynamic_pointer_cast<ListNode>(categoryNode))
            {
                for (const auto &value : catList->values)
                {
                    registerSyntheticFamily(std::make_shared<StringNode>(value));
                }
            }
            else if (auto catString = std::dynamic_pointer_cast<StringNode>(categoryNode))
            {
                registerSyntheticFamily(std::make_shared<StringNode>(catString->value));
            }

            // Deterministic synthetic "LoRA id" for this category bit. (+1 keeps > 0, distinct from legacy 0)
            idx.bit_to_lora_id[bit] = static_cast<int64_t>(bit) + 1;
        }

        return idx;
    }

    // returns vector of LoRA ids contained in prompt
    static std::vector<int64_t> extract_loras(const std::shared_ptr<Prompt> &prompt, const std::optional<generator::TagIndex> &tagIndexOpt, std::optional<Tokenizer> &tok)
    {
        std::vector<int64_t> loras;
        std::unordered_set<int64_t> seen;
        if (!prompt)
            return loras;

        if (!tagIndexOpt.has_value())
        {
            // Legacy path: use prompt->loras directly
            for (int64_t id : prompt->loras)
            {
                if (id != 0 && seen.insert(id).second)
                    loras.push_back(id);
            }
            if (loras.empty())
            {
                constexpr int64_t NO_LORA_ID = -1; // synthetic LoRA for "plain" prompts
                loras.push_back(NO_LORA_ID);
            }
            return loras;
        }
        else
        {
            const auto &tagIndex = *tagIndexOpt;
            bool usedSubkeyIds = false;

            // Preferred: per-subkey mapping
            if (!tagIndex.tag_to_lora_ids.empty() && tok.has_value())
            {
                TagResolutionResult resolution = resolve_prompt_families(prompt, tagIndex, tok);
                loras = std::move(resolution.families);
                usedSubkeyIds = resolution.hadSubkeyMatches;
            }

            if (!usedSubkeyIds)
            {
                // Fallback: category-bit mapping (legacy unique-category buckets)
                uint64_t mask = 0;
                if (tok.has_value())
                {
                    for (const auto &t : prompt->tokens)
                    {
                        const auto tag = tok->tokenToText(t);
                        if (tag)
                        {
                            auto it = tagIndex.tag_mask.find(*tag);
                            if (it != tagIndex.tag_mask.end())
                                mask |= it->second;
                        }
                    }
                }
                mask &= tagIndex.unique_mask_all;
                for_each_set_bit(mask, [&](uint8_t bit)
                                 {
                    int64_t id = tagIndex.bit_to_lora_id[bit];
                    if (id != 0 && seen.insert(id).second)
                        loras.push_back(id); });
            }
        }

        if (loras.empty())
        {
            constexpr int64_t NO_LORA_ID = -1; // synthetic LoRA for "plain" prompts
            loras.push_back(NO_LORA_ID);
        }
        return loras;
    }

    FilteringThread::FilteringThread(const Config &cfg) : config_(cfg)
    {
        std::cout << "Filtering thread initializing..." << std::endl;
        metricsEnabled_ = parseEnvBool("SDML_PIPELINE_METRICS", false);
        metricsLogInterval_ = std::chrono::milliseconds(parsePositiveEnvInt("SDML_PIPELINE_METRICS_INTERVAL_MS", 2000));

        if (config_.useTagsAsLoras && config_.tagDict)
        {
            tagIndex_.emplace(build_tag_index(config_.tagDict));
            const auto &rules = config_.tagDict->getRules();
            const bool intentionallyEmptyFilterCategories = rules.has_filter_categories && rules.filter_categories.empty();
            if (tagIndex_->tag_mask.empty() && !intentionallyEmptyFilterCategories)
            {
                std::cerr << "[Tags][WARN] Built tag index is empty; "
                             "check that filter_categories or unique_categories exist and contain mappable tag subtrees."
                          << std::endl;
            }
            if (tagIndex_->unique_mask_all == 0 && !intentionallyEmptyFilterCategories)
            {
                std::cerr << "[Tags][WARN] unique_mask_all is 0; rules.filter_categories or rules.unique_categories may be empty or unmapped."
                          << std::endl;
            }
        }

        workerThread_ = std::thread(&FilteringThread::processLoop, this);
    }

    FilteringThread::~FilteringThread()
    {
        stopAndJoin();
    }

    void FilteringThread::registerQueue(CandidateQueue *queue)
    {
        if (queue)
        {
            std::lock_guard<std::mutex> lock(queuesMutex_);
            candidateQueues_.push_back(queue);
            workCv_.notify_one();
        }
    }

    std::vector<ScoredPrompt> FilteringThread::getAcceptedPrompts() const
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        std::vector<ScoredPrompt> output;
        output.reserve(slots_.size());
        for (const auto &slot : slots_)
        {
            if (slot.alive)
                output.push_back(slot.ap.scored);
        }
        return output;
    }

    void FilteringThread::stopAndJoin()
    {
        running_ = false;
        workCv_.notify_all();
        if (workerThread_.joinable())
        {
            workerThread_.join();
        }
    }

    void FilteringThread::notifyWorkAvailable()
    {
        workSignalCount_.fetch_add(1, std::memory_order_relaxed);
        workCv_.notify_one();
    }

    std::string FilteringThread::stats() const
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        std::ostringstream oss;
        oss << "Accepted=" << acceptedCountAlive() << ", Distinct LoRAs=" << loraUsage_.size();
        if (metricsEnabled_)
        {
            const double avgWaitMs = metrics_.waitCount > 0
                                         ? (static_cast<double>(metrics_.waitNs) / 1000000.0) / static_cast<double>(metrics_.waitCount)
                                         : 0.0;
            oss << ", Processed=" << metrics_.totalProcessed
                << ", Wakeups=" << metrics_.wakeCount
                << ", EmptyWakeups=" << metrics_.emptyWakeCount
                << ", MaxQueueDepth=" << metrics_.maxQueueDepth
                << ", AvgWaitMs=" << std::fixed << std::setprecision(3) << avgWaitMs;
        }
        return oss.str();
    }

    bool FilteringThread::isExceptional(const ScoredPrompt &cand, const std::vector<int64_t> &loras)
    {
        for (auto l : loras)
        {
            auto itHeap = loraMin_.find(l);
            if (itHeap == loraMin_.end())
                return true;

            // find current weakest live prompt for this LoRA (non-destructive scan)
            auto heap = itHeap->second;
            bool found = false;
            float currentWorst = std::numeric_limits<float>::infinity();
            while (!heap.empty())
            {
                HeapEntry top = heap.top();
                heap.pop();
                if (top.id >= slots_.size())
                    continue;
                const Slot &s = slots_[top.id];
                if (!s.alive || !loraToIds_[l].count(top.id))
                    continue;
                currentWorst = s.ap.scored.score; // equals top.score when valid
                found = true;
                break;
            }

            if (!found)
                return true; // no valid entries yet

            if (cand.score > currentWorst)
                return true;
        }
        return false;
    }

    size_t FilteringThread::acceptedCountAlive() const
    {
        // std::lock_guard<std::mutex> lock(stateMutex_);
        size_t cnt = 0;
        for (const auto &s : slots_)
            if (s.alive)
                ++cnt;
        return cnt;
    }

    FilteringThread::PromptId FilteringThread::allocSlot()
    {
        if (!freeList_.empty())
        {
            PromptId id = freeList_.back();
            freeList_.pop_back();
            slots_[id].alive = true;
            return id;
        }
        slots_.push_back(Slot{});
        slots_.back().alive = true;
        return static_cast<PromptId>(slots_.size() - 1);
    }

    void FilteringThread::markDead(PromptId id)
    {
        if (id >= slots_.size())
            return;
        Slot &slot = slots_[id];
        if (!slot.alive)
            return;

        slot.alive = false;

        // update LoRA usage and indices
        for (auto l : slot.ap.loras)
        {
            auto itU = loraUsage_.find(l);
            if (itU != loraUsage_.end() && --(itU->second) <= 0)
            {
                loraUsage_.erase(itU);
            }

            auto itSet = loraToIds_.find(l);
            if (itSet != loraToIds_.end())
            {
                itSet->second.erase(id);
                if (itSet->second.empty())
                    loraToIds_.erase(itSet);
            }
        }

        freeList_.push_back(id);
    }

    void FilteringThread::indexPrompt(PromptId id, const AcceptedPrompt &ap)
    {
        for (auto l : ap.loras)
        {
            loraToIds_[l].insert(id);
            loraMin_[l].push({ap.scored.score, id});
        }
    }

    std::vector<FilteringThread::PromptId> FilteringThread::selectReplacements(const ScoredPrompt &cand, const std::vector<int64_t> &loras)
    {
        std::vector<PromptId> out;
        for (auto l : loras)
        {
            int usage = loraUsage_.count(l) ? loraUsage_[l] : 0;
            if (usage < config_.loraCap)
                continue;

            int need = 1 + usage - config_.loraCap;
            auto itHeap = loraMin_.find(l);
            if (itHeap == loraMin_.end())
                continue;

            // Use a local copy to avoid mutating the real heap during planning
            auto heap = itHeap->second;
            while (need > 0 && !heap.empty())
            {
                HeapEntry top = heap.top();
                heap.pop();
                if (top.id >= slots_.size())
                    continue;
                const Slot &s = slots_[top.id];
                // lazy skip dead or no-longer-relevant entries
                if (!s.alive || !loraToIds_[l].count(top.id))
                    continue;
                if (cand.score > s.ap.scored.score)
                {
                    out.push_back(top.id);
                    --need;
                }
                else
                {
                    break; // remaining are stronger
                }
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    void FilteringThread::processLoop()
    {
        auto lastLog = std::chrono::steady_clock::now();
        auto drainQueues = [&](ScoredPrompt &cand) -> bool
        {
            std::lock_guard<std::mutex> lock(queuesMutex_);
            size_t queueCount = candidateQueues_.size();
            if (queueCount == 0)
                return false;
            if (nextQueueIndex_ >= queueCount)
                nextQueueIndex_ %= queueCount;
            size_t startIndex = nextQueueIndex_;
            for (size_t offset = 0; offset < queueCount; ++offset)
            {
                size_t idx = (startIndex + offset) % queueCount;
                CandidateQueue *q = candidateQueues_[idx];
                if (q != nullptr && q->tryDequeue(cand))
                {
                    nextQueueIndex_ = (idx + 1) % queueCount;
                    if (metricsEnabled_)
                    {
                        metrics_.maxQueueDepth = std::max(metrics_.maxQueueDepth, q->sizeApprox());
                    }
                    return true;
                }
            }
            nextQueueIndex_ = (startIndex + 1) % queueCount;
            return false;
        };

        while (running_)
        {
            ScoredPrompt cand;
            const uint64_t observedSignal = workSignalCount_.load(std::memory_order_relaxed);
            if (!drainQueues(cand))
            {
                auto waitStart = std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lock(workMutex_);
                metrics_.waitCount++;
                workCv_.wait(lock, [this, observedSignal]()
                             { return !running_ || workSignalCount_.load(std::memory_order_relaxed) != observedSignal; });
                lock.unlock();
                metrics_.waitNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count());
                metrics_.wakeCount++;
                if (!running_)
                    break;
                if (!drainQueues(cand))
                {
                    metrics_.emptyWakeCount++;
                    continue;
                }
            }
            ++metrics_.totalProcessed;

            std::lock_guard<std::mutex> lock(stateMutex_);

            if (cand.score < config_.scoreThreshold)
                continue;

            // ======= Score-only mode: accept all above threshold =======
            if (config_.scoreOnly)
            {
                PromptId id = allocSlot();
                Slot &slot = slots_[id];
                slot.ap.scored = std::move(cand);
                slot.ap.loras.clear();
                fifo_.push_back(id);
                // global cap FIFO
                while (acceptedCountAlive() > config_.maxAcceptedPrompts && !fifo_.empty())
                {
                    PromptId old = fifo_.front();
                    fifo_.pop_front();
                    markDead(old);
                }
                // periodic log
                auto now = std::chrono::steady_clock::now();
                metrics_.totalAccepted++;
                if (metrics_.totalProcessed % 100 == 0 || (now - lastLog) >= metricsLogInterval_)
                {
                    lastLog = now;
                    std::cout << "[Filter score-only] Accepted=" << acceptedCountAlive()
                              << ", Processed=" << metrics_.totalProcessed
                              << ", Wakeups=" << metrics_.wakeCount
                              << ", EmptyWakeups=" << metrics_.emptyWakeCount
                              << ", MaxQueueDepth=" << metrics_.maxQueueDepth << std::endl;
                }
                continue;
            }

            // ======= LoRA-diversity mode =======
            // Use tags as LoRAs (via tagIndex) or use actual prompt LoRAs based on config
            std::vector<int64_t> loras;
            if (config_.useTagsAsLoras && tagIndex_.has_value())
            {
                loras = extract_loras(cand.prompt, tagIndex_, tokenizer_);
            }
            else
            {
                // Direct LoRA extraction from prompt
                std::optional<generator::TagIndex> noTagIndex;
                std::optional<generator::Tokenizer> noTokenizer;
                loras = extract_loras(cand.prompt, noTagIndex, noTokenizer);
            }

            bool exceptional = isExceptional(cand, loras);

            bool reject = false;
            std::vector<PromptId> toRemove;

            for (auto l : loras)
            {
                int usage = loraUsage_.count(l) ? loraUsage_[l] : 0;
                if (usage >= config_.loraCap)
                {
                    if (exceptional)
                    {
                        int need = 1 + usage - config_.loraCap;
                        // plan replacements strictly for this SPECIFIC LoRA
                        auto repl = selectReplacements(cand, std::vector<int64_t>{l});
                        if (static_cast<int>(repl.size()) < need)
                        {
                            reject = true;
                            break;
                        }
                        toRemove.insert(toRemove.end(), repl.begin(), repl.end());
                    }
                    else
                    {
                        reject = true;
                        break;
                    }
                }
            }

            if (reject)
                continue;

            // deduplicate the accumulated removal list across all LoRAs
            if (!toRemove.empty())
            {
                std::sort(toRemove.begin(), toRemove.end());
                toRemove.erase(std::unique(toRemove.begin(), toRemove.end()), toRemove.end());
            }

            // mark removals dead
            for (PromptId id : toRemove)
                markDead(id);

            // accept new candidate
            PromptId id = allocSlot();
            Slot &slot = slots_[id];
            slot.ap.scored = std::move(cand);
            slot.ap.loras = loras;

            // update usage & bestscore
            for (auto l : loras)
            {
                ++loraUsage_[l];
                auto &best = loraBestScore_[l];
                if (cand.score > best)
                    best = cand.score;
            }

            indexPrompt(id, slot.ap);
            fifo_.push_back(id);
            metrics_.totalAccepted++;

            // global cap FIFO
            while (acceptedCountAlive() > config_.maxAcceptedPrompts && !fifo_.empty())
            {
                PromptId old = fifo_.front();
                fifo_.pop_front();
                markDead(old);
            }

            // periodic log
            auto now = std::chrono::steady_clock::now();
            if (metrics_.totalProcessed % 100 == 0 || (now - lastLog) >= metricsLogInterval_)
            {
                lastLog = now;
                std::cout << "[Filter] Accepted=" << acceptedCountAlive()
                          << ", Distinct LoRAs=" << loraUsage_.size()
                          << ", Processed=" << metrics_.totalProcessed
                          << ", Wakeups=" << metrics_.wakeCount
                          << ", EmptyWakeups=" << metrics_.emptyWakeCount
                          << ", MaxQueueDepth=" << metrics_.maxQueueDepth << std::endl;
            }
        }

        std::cout << "Filtering thread exiting. Final accepted prompts: " << acceptedCountAlive() << std::endl;
    }

} // namespace generator
