#include "evaluation/scoring_thread.hpp"
#include "utils/types.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

namespace generator
{
    namespace
    {
        unsigned resolveMaxBatchSize()
        {
            const char *env = std::getenv("SDML_SCORING_BATCH_MAX");
            if (env == nullptr || *env == '\0')
                return DEFAULT_SCORING_BATCH_SIZE;
            char *endPtr = nullptr;
            unsigned long parsed = std::strtoul(env, &endPtr, 10);
            if (endPtr == env || *endPtr != '\0')
            {
                std::cerr << "Ignoring invalid SDML_SCORING_BATCH_MAX value \"" << env << "\"" << std::endl;
                return DEFAULT_SCORING_BATCH_SIZE;
            }
            if (parsed == 0)
                parsed = DEFAULT_SCORING_BATCH_SIZE;
            if (parsed > static_cast<unsigned long>(std::numeric_limits<unsigned>::max()))
                parsed = std::numeric_limits<unsigned>::max();
            if (parsed < MIN_SCORING_BATCH_SIZE)
                parsed = MIN_SCORING_BATCH_SIZE;
            return static_cast<unsigned>(parsed);
        }

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

        std::chrono::milliseconds resolveBatchFillWindow()
        {
            return std::chrono::milliseconds(parsePositiveEnvInt("SDML_SCORING_BATCH_FILL_MS", 5));
        }

        std::chrono::milliseconds resolveMetricsLogInterval()
        {
            return std::chrono::milliseconds(parsePositiveEnvInt("SDML_PIPELINE_METRICS_INTERVAL_MS", 2000));
        }

        size_t batchHistogramIndex(size_t size)
        {
            if (size <= 1)
                return 0;
            if (size <= 8)
                return 1;
            if (size <= 32)
                return 2;
            if (size <= 128)
                return 3;
            if (size <= 512)
                return 4;
            if (size <= 1024)
                return 5;
            if (size <= 2048)
                return 6;
            return 7;
        }

        const char *batchHistogramLabel(size_t index)
        {
            static constexpr const char *labels[] = {"1", "2-8", "9-32", "33-128", "129-512", "513-1024", "1025-2048", "2049+"};
            return labels[index];
        }
    }

    ScoringThreadPool::ScoringThreadPool(const std::vector<ModelScoringConfig> &modelConfigs,
                                         int numGeneratorThreads,
                                         int numThreads,
                                         int numGpus)
        : numThreads_(numThreads),
          numGeneratorThreads_(numGeneratorThreads),
          numGpus_(numGpus),
          modelConfigs_(modelConfigs)
    {
        std::cout << "Initializing scoring thread pool with " << numThreads_ << " threads across "
                  << numGpus_ << " GPUs for " << modelConfigs_.size() << " models..." << std::endl;
        queues_.reserve(numGeneratorThreads);
        for (int i = 0; i < numGeneratorThreads; ++i)
        {
            queues_.emplace_back(std::make_unique<PromptQueue>(1 << 14));
        }
        // Create the scoring threads
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
            int deviceId = 0;
            if (numGpus_ > 0)
                deviceId = i % numGpus_;
            try
            {
                auto thread = std::make_unique<ScoringThread>(modelConfigs_, assignedQueues, i, deviceId);
                scoringThreads_.push_back(std::move(thread));
                std::cout << "Scoring thread " << i << " initialized with " << assignedQueues.size()
                          << " queues on GPU " << deviceId << std::endl;
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
        bool ok = queues_[generatorThreadId]->enqueue(std::move(prompt));
        if (ok)
        {
            int threadIndex = getScoringThreadIndex(generatorThreadId);
            if (threadIndex >= 0 && threadIndex < static_cast<int>(scoringThreads_.size()) && scoringThreads_[threadIndex])
            {
                scoringThreads_[threadIndex]->notifyInputAvailable();
            }
        }
        return ok;
    }

    int ScoringThreadPool::getScoringThreadIndex(int generatorThreadId) const
    {
        if (numThreads_ <= 0)
            return 0;
        int index = generatorThreadId % numThreads_;
        if (index < 0)
            index += numThreads_;
        return index;
    }

    void ScoringThreadPool::setFilterThread(FilteringThread *filterThread)
    {
        for (auto &thread : scoringThreads_)
            thread->setFilterThread(filterThread);
    }

    void ScoringThreadPool::setFilterThreads(const std::vector<FilteringThread *> &filterThreads)
    {
        for (auto &thread : scoringThreads_)
            thread->setFilterThreads(filterThreads);
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

    void ScoringThreadPool::closeInputQueues()
    {
        for (auto &queue : queues_)
        {
            if (queue)
            {
                queue->close();
            }
        }
    }

    ScoringThread::ScoringThread(const std::vector<ModelScoringConfig> &modelConfigs,
                                 std::vector<PromptQueue *> assignedQueues,
                                 int threadId,
                                 int deviceId)
        : threadId_(threadId),
          deviceId_(deviceId),
          assignedQueues_(std::move(assignedQueues))
    {
        std::cout << "Initializing scoring thread " << threadId_ << " on device " << deviceId_
                  << " with " << modelConfigs.size() << " models..." << std::endl;
        globalMaxBatchSize_ = resolveMaxBatchSize();
        batchFillWindow_ = resolveBatchFillWindow();
        metricsEnabled_ = parseEnvBool("SDML_PIPELINE_METRICS", false);
        metricsLogInterval_ = resolveMetricsLogInterval();
        // Create one ScoringEngine per model
        for (const auto &config : modelConfigs)
        {
            int actualDeviceId = (config.preferredDeviceId >= 0) ? config.preferredDeviceId : deviceId_;
            try
            {
                auto engine = std::make_unique<ScoringEngine>(
                    config.onnxModelPath, threadId, actualDeviceId, config.clipCachePath);
                engines_[config.profileId] = std::move(engine);
                scoreThresholds_[config.profileId] = config.scoreThreshold;
                maxBatchSizes_[config.profileId] = (config.batchSize > 0)
                                                       ? config.batchSize
                                                       : globalMaxBatchSize_;
                std::cout << "  Profile " << config.profileId << " (SD model " << config.modelId << "): "
                          << config.onnxModelPath
                          << " (threshold=" << config.scoreThreshold
                          << ", batch=" << maxBatchSizes_[config.profileId]
                          << ", device=" << actualDeviceId << ")" << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Failed to load scoring engine for profile " << config.profileId
                          << " (" << config.onnxModelPath << "): " << e.what() << std::endl;
                throw;
            }
        }
        std::cout << "Starting scoring thread " << threadId_ << " worker..." << std::endl;
        workerThread_ = std::thread(&ScoringThread::processLoop, this);
    }

    void ScoringThread::notifyInputAvailable()
    {
        pendingPromptCount_.fetch_add(1, std::memory_order_relaxed);
        inputCv_.notify_one();
    }

    ScoringThread::~ScoringThread()
    {
        stopAndJoin();
        for (auto *queue : outQueues_)
        {
            delete queue;
        }
        outQueues_.clear();
    }

    void ScoringThread::setFilterThread(FilteringThread *filterThread)
    {
        // Legacy single filter thread support - creates a single-element vector
        filterThreads_.clear();
        filterThreads_.push_back(filterThread);
        // Create output queue for this single filter thread
        for (auto *q : outQueues_)
            delete q;
        outQueues_.clear();
        auto *queue = new CandidateQueue(1 << 16);
        outQueues_.push_back(queue);
        if (filterThread)
        {
            filterThread->registerQueue(queue);
        }
    }

    void ScoringThread::setFilterThreads(const std::vector<FilteringThread *> &filterThreads)
    {
        filterThreads_ = filterThreads;
        // Clean up old queues
        for (auto *q : outQueues_)
            delete q;
        outQueues_.clear();
        // Create output queues indexed by profileId (same size as filterThreads_)
        // Null entries are possible for gaps in profile IDs
        outQueues_.resize(filterThreads_.size(), nullptr);
        size_t activeCount = 0;
        for (size_t i = 0; i < filterThreads_.size(); ++i)
        {
            if (filterThreads_[i])
            {
                auto *queue = new CandidateQueue(1 << 16);
                outQueues_[i] = queue;
                filterThreads_[i]->registerQueue(queue);
                ++activeCount;
            }
        }
        std::cout << "Scoring thread " << threadId_ << ": Configured " << activeCount
                  << " filter threads across " << outQueues_.size() << " model ID slots" << std::endl;
    }

    void ScoringThread::stopAndJoin()
    {
        running_ = false;
        inputCv_.notify_all();
        for (PromptQueue *q : assignedQueues_)
        {
            if (q)
                q->close();
        }
        for (CandidateQueue *q : outQueues_)
        {
            if (q)
                q->close();
        }
        if (workerThread_.joinable())
        {
            workerThread_.join();
        }
    }

    void ScoringThread::recordBatchHistogram(std::array<uint64_t, 8> &histogram, size_t size)
    {
        histogram[batchHistogramIndex(size)]++;
    }

    void ScoringThread::logMetrics(const std::unordered_map<int32_t, size_t> &processedByModel,
                                   const std::chrono::steady_clock::time_point &startTime,
                                   std::chrono::steady_clock::time_point &lastPrint)
    {
        auto now = std::chrono::steady_clock::now();
        if ((now - lastPrint) < metricsLogInterval_)
            return;
        double elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(now - startTime).count();
        double promptsPerSecond = elapsedSeconds > 0.0 ? static_cast<double>(metrics_.totalProcessed) / elapsedSeconds : 0.0;
        std::ostringstream oss;
        oss << "Scoring thread " << threadId_ << " stats: "
            << metrics_.totalProcessed << " processed, "
            << metrics_.totalAccepted << " accepted, "
            << std::fixed << std::setprecision(1) << promptsPerSecond << " prompts/sec";
        for (const auto &[profileId, count] : processedByModel)
        {
            oss << " [P" << profileId << ":" << count << "]";
        }
        if (metricsEnabled_)
        {
            const double avgCollectBatch = metrics_.preProfileBatchSamples > 0
                                               ? static_cast<double>(metrics_.totalProcessed) / static_cast<double>(metrics_.preProfileBatchSamples)
                                               : 0.0;
            const double avgProfileBatch = metrics_.postProfileBatchSamples > 0
                                               ? static_cast<double>(metrics_.totalProcessed) / static_cast<double>(metrics_.postProfileBatchSamples)
                                               : 0.0;
            const double avgFillWaitMs = metrics_.batchFillWaitCount > 0
                                             ? (static_cast<double>(metrics_.batchFillWaitNs) / 1000000.0) / static_cast<double>(metrics_.batchFillWaitCount)
                                             : 0.0;
            const double avgInputWaitMs = metrics_.inputWaitCount > 0
                                              ? (static_cast<double>(metrics_.inputWaitNs) / 1000000.0) / static_cast<double>(metrics_.inputWaitCount)
                                              : 0.0;
            oss << " | wait(fill/input)=" << std::setprecision(3) << avgFillWaitMs << "/" << avgInputWaitMs << " ms"
                << " | depth(in/out)=" << metrics_.maxInputQueueDepth << "/" << metrics_.maxOutputQueueDepth
                << " | batch(avg pre/post)=" << std::setprecision(1) << avgCollectBatch << "/" << avgProfileBatch
                << " | wakes=" << metrics_.inputWakeups << " empty=" << metrics_.inputWakeNoWork
                << " | pre=[";
            for (size_t i = 0; i < metrics_.preProfileBatchHistogram.size(); ++i)
            {
                if (i > 0)
                    oss << ",";
                oss << batchHistogramLabel(i) << ":" << metrics_.preProfileBatchHistogram[i];
            }
            oss << "] post=[";
            for (size_t i = 0; i < metrics_.postProfileBatchHistogram.size(); ++i)
            {
                if (i > 0)
                    oss << ",";
                oss << batchHistogramLabel(i) << ":" << metrics_.postProfileBatchHistogram[i];
            }
            oss << "]";
        }
        std::cout << oss.str() << std::endl;
        lastPrint = now;
    }

    ScoringThread::BatchItem ScoringThread::prepareBatch(std::vector<std::unique_ptr<Prompt>> &prompts)
    {
        std::vector<std::unique_ptr<Prompt>> batchPrompts;
        batchPrompts.swap(prompts);
        PromptBatch promptBatch;
        promptBatch.batch_size = batchPrompts.size();
        promptBatch.sequence_length = MAX_SEQUENCE_LENGTH;
        const size_t batchSize = batchPrompts.size();
        const size_t loraTensorSize = batchSize * MAX_LORAS;
        // Detect clip_embed mode via IDs/embedding data.
        bool isClipEmbed = !batchPrompts.empty() &&
                           (!batchPrompts[0]->clipEmb.empty() || !batchPrompts[0]->clipTagIds.empty() ||
                            batchPrompts[0]->clipDim > 0);
        if (isClipEmbed)
        {
            size_t clipDim = batchPrompts[0]->clipEmb.empty()
                                 ? static_cast<size_t>(batchPrompts[0]->clipDim)
                                 : batchPrompts[0]->clipEmb.size();
            promptBatch.clip_dim = clipDim;
            bool hasHostPooledEmb = false;
            for (size_t i = 0; i < batchSize; ++i)
            {
                if (!batchPrompts[i]->clipEmb.empty())
                {
                    hasHostPooledEmb = true;
                    break;
                }
            }
            if (hasHostPooledEmb)
            {
                promptBatch.clip_emb.resize(batchSize * clipDim, 0);
            }
            promptBatch.clip_tag_offsets.resize(batchSize + 1, 0);
            size_t totalTagCount = 0;
            for (size_t i = 0; i < batchSize; ++i)
            {
                totalTagCount += batchPrompts[i]->clipTagIds.size();
            }
            promptBatch.clip_tag_ids_flat.reserve(totalTagCount);
            // No token/mask tensors needed
            promptBatch.token_indices.clear();
            promptBatch.token_mask.clear();
        }
        else
        {
            promptBatch.clip_dim = 0;
            size_t tokenTensorSize = batchSize * MAX_SEQUENCE_LENGTH;
            promptBatch.token_indices.resize(tokenTensorSize, 0);
            promptBatch.token_mask.resize(tokenTensorSize, 0.0f);
        }
        promptBatch.lora_ids.resize(loraTensorSize, 0);
        promptBatch.lora_weights.resize(loraTensorSize, 0.0f);
        promptBatch.cfg_scales.resize(batchSize);
        promptBatch.num_loras.resize(batchSize);
        promptBatch.sampler_ids.resize(batchSize);
        promptBatch.steps_log.resize(batchSize);
        promptBatch.steps_bucket.resize(batchSize);
        promptBatch.upscaler_ids.resize(batchSize);
        promptBatch.up_has.resize(batchSize);
        promptBatch.up_steps.resize(batchSize);
        promptBatch.denoise.resize(batchSize);
        promptBatch.model_ids.resize(batchSize);
        for (size_t i = 0; i < batchSize; ++i)
        {
            const auto &prompt = batchPrompts[i];
            if (isClipEmbed)
            {
                // Copy pooled CLIP embedding
                size_t clipDim = promptBatch.clip_dim;
                if (!prompt->clipEmb.empty() && !promptBatch.clip_emb.empty())
                {
                    size_t copyLen = std::min(prompt->clipEmb.size(), clipDim);
                    std::memcpy(&promptBatch.clip_emb[i * clipDim], prompt->clipEmb.data(),
                                copyLen * sizeof(uint16_t));
                }
                promptBatch.clip_tag_offsets[i] = static_cast<uint32_t>(promptBatch.clip_tag_ids_flat.size());
                promptBatch.clip_tag_ids_flat.insert(
                    promptBatch.clip_tag_ids_flat.end(),
                    prompt->clipTagIds.begin(),
                    prompt->clipTagIds.end());
            }
            else
            {
                // Copy token indices and mask
                for (size_t j = 0; j < prompt->tokens.size() && j < MAX_SEQUENCE_LENGTH; ++j)
                {
                    size_t idx = i * MAX_SEQUENCE_LENGTH + j;
                    promptBatch.token_indices[idx] = prompt->tokens[j];
                    promptBatch.token_mask[idx] = prompt->tokens[j] > 0 ? 1.0f : 0.0f;
                }
            }
            promptBatch.cfg_scales[i] = prompt->cfgScale / MAX_CFG_SCALE;
            promptBatch.num_loras[i] = static_cast<float>(prompt->numLoras) / static_cast<float>(MAX_LORAS);
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
            promptBatch.up_steps[i] = static_cast<float>(prompt->upscalerSteps) / 25.0f;
            promptBatch.denoise[i] = prompt->denoisingStrength;
            promptBatch.model_ids[i] = prompt->modelId;
        }
        if (isClipEmbed)
        {
            promptBatch.clip_tag_offsets[batchSize] = static_cast<uint32_t>(promptBatch.clip_tag_ids_flat.size());
        }
        return {promptBatch, std::move(batchPrompts)};
    }

    void ScoringThread::processPromptsByProfile(std::vector<std::unique_ptr<Prompt>> &prompts)
    {
        if (prompts.empty())
            return;
        // Group prompts by profileId (each profile has its own engine/threshold)
        std::unordered_map<int32_t, std::vector<std::unique_ptr<Prompt>>> promptsByProfile;
        for (auto &prompt : prompts)
        {
            int32_t profileId = prompt->profileId;
            promptsByProfile[profileId].push_back(std::move(prompt));
        }
        prompts.clear();
        std::vector<std::pair<int32_t, size_t>> profileOrder;
        profileOrder.reserve(promptsByProfile.size());
        for (const auto &[profileId, profilePrompts] : promptsByProfile)
        {
            profileOrder.emplace_back(profileId, profilePrompts.size());
        }
        std::sort(profileOrder.begin(), profileOrder.end(),
                  [](const auto &lhs, const auto &rhs)
                  { return lhs.second > rhs.second; });
        // Process each profile's prompts with the appropriate engine
        for (const auto &[profileId, ignoredSize] : profileOrder)
        {
            (void)ignoredSize;
            auto promptsIt = promptsByProfile.find(profileId);
            if (promptsIt == promptsByProfile.end())
                continue;
            auto &profilePrompts = promptsIt->second;
            auto engineIt = engines_.find(profileId);
            if (engineIt == engines_.end())
            {
                std::cerr << "Scoring thread " << threadId_ << ": No engine for profile " << profileId
                          << ", skipping " << profilePrompts.size() << " prompts" << std::endl;
                continue;
            }
            float threshold = scoreThresholds_.count(profileId) ? scoreThresholds_[profileId] : 0.8f;
            // Process in batches up to this profile's max batch size
            unsigned profileBatchSize = maxBatchSizes_.count(profileId)
                                            ? maxBatchSizes_[profileId]
                                            : globalMaxBatchSize_;
            size_t offset = 0;
            while (offset < profilePrompts.size())
            {
                size_t batchEnd = std::min(offset + profileBatchSize, profilePrompts.size());
                size_t effectiveBatchSize = batchEnd - offset;
                metrics_.postProfileBatchSamples++;
                recordBatchHistogram(metrics_.postProfileBatchHistogram, effectiveBatchSize);
                std::vector<std::unique_ptr<Prompt>> batchPrompts;
                batchPrompts.reserve(effectiveBatchSize);
                for (size_t i = offset; i < batchEnd; ++i)
                {
                    batchPrompts.push_back(std::move(profilePrompts[i]));
                }
                BatchItem batch = prepareBatch(batchPrompts);
                std::vector<float> scores;
                try
                {
                    scores = engineIt->second->scoreBatch(batch.batch);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Scoring thread " << threadId_ << ": scoreBatch failed for profile "
                              << profileId << ": " << e.what() << std::endl;
                    scores.assign(batch.originalPrompts.size(), 0.0f);
                }
                if (scores.size() != batch.originalPrompts.size())
                {
                    std::cerr << "Scoring thread " << threadId_ << ": score count mismatch for profile "
                              << profileId << std::endl;
                    scores.resize(std::min(scores.size(), batch.originalPrompts.size()));
                }
                // Send scored prompts to filtering thread
                size_t acceptedForBatch = 0;
                for (size_t i = 0; i < scores.size(); ++i)
                {
                    if (scores[i] < threshold)
                        continue;
                    ScoredPrompt cand;
                    cand.prompt = std::shared_ptr<Prompt>(std::move(batch.originalPrompts[i]));
                    cand.score = scores[i];
                    // Select output queue based on profileId
                    CandidateQueue *targetQueue = nullptr;
                    if (!outQueues_.empty() && profileId >= 0 &&
                        static_cast<size_t>(profileId) < outQueues_.size())
                    {
                        targetQueue = outQueues_[static_cast<size_t>(profileId)];
                    }
                    if (!targetQueue)
                        continue; // no filter thread for this profile
                    if (!targetQueue->enqueue(std::move(cand)) || !running_)
                        break;
                    ++acceptedForBatch;
                    if (static_cast<size_t>(profileId) < filterThreads_.size() && filterThreads_[static_cast<size_t>(profileId)])
                    {
                        filterThreads_[static_cast<size_t>(profileId)]->notifyWorkAvailable();
                    }
                    if (metricsEnabled_)
                    {
                        metrics_.maxOutputQueueDepth = std::max(metrics_.maxOutputQueueDepth, targetQueue->sizeApprox());
                    }
                }
                metrics_.totalAccepted += acceptedForBatch;
                offset = batchEnd;
                if (!running_)
                    break;
            }
            if (!running_)
                break;
        }
    }

    void ScoringThread::processLoop()
    {
        std::cout << "Scoring thread " << threadId_ << " started with " << engines_.size() << " models" << std::endl;
        // Stats tracking per model
        std::unordered_map<int32_t, size_t> processedByModel;
        auto startTime = std::chrono::steady_clock::now();
        auto lastPrint = startTime;
        std::vector<std::unique_ptr<Prompt>> prompts_buffer;
        prompts_buffer.reserve(globalMaxBatchSize_);
        auto drainAssignedQueues = [&](size_t maxItems) -> size_t
        {
            size_t drained = 0;
            std::unique_ptr<Prompt> item;
            for (PromptQueue *q : assignedQueues_)
            {
                while (q != nullptr && prompts_buffer.size() < maxItems && q->tryDequeue(item))
                {
                    drained++;
                    prompts_buffer.emplace_back(std::move(item));
                    int64_t previous = pendingPromptCount_.fetch_sub(1, std::memory_order_relaxed);
                    if (previous <= 0)
                        pendingPromptCount_.store(0, std::memory_order_relaxed);
                    if (metricsEnabled_)
                    {
                        metrics_.maxInputQueueDepth = std::max(metrics_.maxInputQueueDepth, q->sizeApprox());
                    }
                }
            }
            return drained;
        };
        while (running_)
        {
            if (prompts_buffer.empty())
            {
                size_t drainedNow = drainAssignedQueues(1);
                if (drainedNow == 0)
                {
                    auto waitStart = std::chrono::steady_clock::now();
                    std::unique_lock<std::mutex> lock(inputMutex_);
                    metrics_.inputWaitCount++;
                    inputCv_.wait(lock, [this]()
                                  { return !running_ || pendingPromptCount_.load(std::memory_order_relaxed) > 0; });
                    lock.unlock();
                    metrics_.inputWaitNs += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count());
                    if (!running_)
                        break;
                    metrics_.inputWakeups++;
                    if (drainAssignedQueues(1) == 0)
                    {
                        metrics_.inputWakeNoWork++;
                        continue;
                    }
                }
            }
            auto waitDeadline = std::chrono::steady_clock::now() + batchFillWindow_;
            while (prompts_buffer.size() < globalMaxBatchSize_)
            {
                if (drainAssignedQueues(globalMaxBatchSize_) > 0)
                    continue;
                auto now = std::chrono::steady_clock::now();
                if (now >= waitDeadline)
                    break;
                auto waitStart = now;
                std::unique_lock<std::mutex> lock(inputMutex_);
                metrics_.batchFillWaitCount++;
                inputCv_.wait_until(lock, waitDeadline, [this]()
                                    { return !running_ || pendingPromptCount_.load(std::memory_order_relaxed) > 0; });
                lock.unlock();
                metrics_.batchFillWaitNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count());
                if (!running_)
                {
                    break;
                }
            }
            if (prompts_buffer.empty())
            {
                continue;
            }
            // Count prompts before processing for stats
            size_t batchSize = prompts_buffer.size();
            metrics_.preProfileBatchSamples++;
            recordBatchHistogram(metrics_.preProfileBatchHistogram, batchSize);
            std::unordered_map<int32_t, size_t> batchCountByProfile;
            for (const auto &p : prompts_buffer)
            {
                batchCountByProfile[p->profileId]++;
            }
            // Process prompts grouped by profile
            processPromptsByProfile(prompts_buffer);
            if (!running_)
                break;
            // Update stats
            metrics_.totalProcessed += batchSize;
            metrics_.batchesProcessed++;
            for (const auto &[profileId, count] : batchCountByProfile)
            {
                processedByModel[profileId] += count;
            }
            logMetrics(processedByModel, startTime, lastPrint);
        }
        logMetrics(processedByModel, startTime, lastPrint);
        std::cout << "Scoring thread " << threadId_ << " exiting" << std::endl;
    }

} // namespace generator
