#include <algorithm>
#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>

#include "evaluation/filtering_thread.hpp"

namespace generator
{

    // returns vector of LoRA ids contained in prompt
    static std::vector<int64_t> extract_loras(const std::shared_ptr<Prompt> &prompt)
    {
        std::vector<int64_t> loras;
        std::unordered_set<int64_t> seen; // guarantees uniqueness
        if (!prompt)
            return loras;

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

    FilteringThread::FilteringThread(const Config &cfg) : config_(cfg)
    {
        std::cout << "Filtering thread initializing..." << std::endl;
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
        if (workerThread_.joinable())
        {
            workerThread_.join();
        }
    }

    std::string FilteringThread::stats() const
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        std::ostringstream oss;
        oss << "Accepted=" << acceptedCountAlive() << ", Distinct LoRAs=" << loraUsage_.size();
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
        size_t totalProcessed = 0;
        auto lastLog = std::chrono::steady_clock::now();

        while (running_)
        {
            bool gotItem = false;
            ScoredPrompt cand;

            {
                // std::lock_guard<std::mutex> lock(queuesMutex_);
                for (CandidateQueue *q : candidateQueues_)
                {
                    if (q && q->try_dequeue(cand))
                    {
                        gotItem = true;
                        break;
                    }
                }
            }

            if (!gotItem)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (!running_)
                {
                    bool anyNotEmpty = false;
                    // std::lock_guard<std::mutex> lock(queuesMutex_);
                    for (CandidateQueue *q : candidateQueues_)
                    {
                        if (q && q->size_approx() > 0)
                        {
                            anyNotEmpty = true;
                            break;
                        }
                    }
                    if (!anyNotEmpty)
                        break;
                }
                continue;
            }
            ++totalProcessed;

            const std::vector<int64_t> loras = extract_loras(cand.prompt);

            std::lock_guard<std::mutex> lock(stateMutex_);

            if (cand.score < config_.scoreThreshold)
                continue;

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

            // global cap FIFO
            while (acceptedCountAlive() > config_.maxAcceptedPrompts && !fifo_.empty())
            {
                PromptId old = fifo_.front();
                fifo_.pop_front();
                markDead(old);
            }

            // periodic log
            auto now = std::chrono::steady_clock::now();
            if (totalProcessed % 100 == 0 || std::chrono::duration_cast<std::chrono::seconds>(now - lastLog).count() >= 2)
            {
                lastLog = now;
                std::cout << "[Filter] Processed=" << totalProcessed << ", Accepted=" << acceptedCountAlive()
                          << ", Queue=" << 0 << ", Unique LoRAs=" << loraUsage_.size() << std::endl;
            }
        }

        std::cout << "Filtering thread exiting. Final accepted prompts: " << acceptedCountAlive() << std::endl;
    }

} // namespace generator
