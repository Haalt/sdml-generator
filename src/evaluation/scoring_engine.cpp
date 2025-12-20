#include "evaluation/scoring_engine.hpp"
#include "evaluation/clip_pooling_kernel.hpp"
#include "onnxruntime_cxx_api.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <onnxruntime_c_api.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace generator
{
    namespace
    {
        int parsePositiveEnvInt(const char *name, int fallback)
        {
            const char *value = std::getenv(name);
            if (value == nullptr || *value == '\0')
                return fallback;
            char *endPtr = nullptr;
            long parsed = std::strtol(value, &endPtr, 10);
            if (endPtr == value || *endPtr != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
                return fallback;
            return static_cast<int>(parsed);
        }

        bool parseEnvBool(const char *name, bool fallback)
        {
            const char *value = std::getenv(name);
            if (value == nullptr || *value == '\0')
                return fallback;
            return std::string(value) != "0";
        }
    }

    // global ONNX Runtime environment (single instance per process)
    Ort::Env g_env{ORT_LOGGING_LEVEL_WARNING, "ScoringEngine"};
    // Ort::Env g_env{ORT_LOGGING_LEVEL_INFO, "ScoringEngine"};

    CudaStreamManager::CudaStreamManager() : initialized_(false)
    {
        try
        {
            for (int i = 0; i < MAX_STREAMS; ++i)
            {
                cudaError_t err = cudaStreamCreate(&streams_[i]);
                if (err != cudaSuccess)
                {
                    std::cerr << "Failed to create CUDA stream " << i << ": " << cudaGetErrorString(err) << std::endl;
                    // clean up previously created streams ()
                    for (int j = 0; j < i; ++j)
                    {
                        cudaStreamDestroy(streams_[j]);
                    }
                    throw std::runtime_error("Failed to create CUDA streams");
                }
            }
            initialized_ = true;
            std::cout << "Created " << MAX_STREAMS << " CUDA streams for parallel execution" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "CUDA stream creation failed, falling back to default stream: " << e.what() << std::endl;
            initialized_ = false;
        }
    }

    void ScoringEngine::rebuildBindingIfNeeded()
    {
        if (!io_binding_)
            return;
        if (!binding_needs_rebuild_)
            return;

        io_binding_->ClearBoundInputs();
        io_binding_->ClearBoundOutputs();

        for (size_t i = 0; i < numInputs_; ++i)
        {
            io_binding_->BindInput(input_names_[i], d_inputs_[i].tensor);
        }
        // Bind outputs by device memory info so ORT allocates with the correct dtype on CUDA
        io_binding_->BindOutput(output_names_[0], memory_info_cuda_.GetConst());

        binding_needs_rebuild_ = false;
    }

    CudaStreamManager::~CudaStreamManager()
    {
        if (initialized_)
        {
            for (int i = 0; i < MAX_STREAMS; ++i)
            {
                cudaStreamDestroy(streams_[i]);
            }
        }
    }

    CudaStreamManager &CudaStreamManager::getInstance()
    {
        static CudaStreamManager instance;
        return instance;
    }

    cudaStream_t CudaStreamManager::getStream(int threadId)
    {
        if (!initialized_)
        {
            return 0; // default stream
        }
        return streams_[threadId % MAX_STREAMS];
    }

    ScoringEngine::ScoringEngine(const std::string &modelPath, int threadId, int deviceId, const std::string &clipCachePath)
        : memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)),
          memory_info_cuda_(Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, deviceId, OrtMemTypeDefault)),
          thread_id_(threadId),
          device_id_(deviceId),
          clip_cache_path_(clipCachePath)
    {
        try
        {
            metricsEnabled_ = parseEnvBool("SDML_PIPELINE_METRICS", false);
            metricsLogInterval_ = std::chrono::milliseconds(parsePositiveEnvInt("SDML_PIPELINE_METRICS_INTERVAL_MS", 2000));
            lastMetricsLog_ = std::chrono::steady_clock::now();
            // Set device for this thread/session; create stream on THIS device.
            // Shared CudaStreamManager streams are created on device 0, causing
            // "invalid resource handle" when engines use other GPUs.
            cudaSetDevice(device_id_);
            cudaError_t streamErr = cudaStreamCreate(&cuda_stream_);
            if (streamErr != cudaSuccess)
            {
                std::cerr << "Thread " << threadId << ": failed to create CUDA stream on device "
                          << deviceId << " (" << cudaGetErrorString(streamErr)
                          << "); falling back to default stream." << std::endl;
                cuda_stream_ = nullptr;
                owns_stream_ = false;
            }
            else
            {
                owns_stream_ = true;
            }
            cudaError_t eventErr = cudaEventCreateWithFlags(&outputReadyEvent_, cudaEventDisableTiming);
            if (eventErr != cudaSuccess)
            {
                outputReadyEvent_ = nullptr;
                std::cerr << "Thread " << threadId << ": failed to create output event on device "
                          << deviceId << " (" << cudaGetErrorString(eventErr) << ")" << std::endl;
            }

            initSession(modelPath);
            const char *poolingEnv = std::getenv("SDML_CLIP_POOLING_CUDA");
            useCudaClipPooling_ = (poolingEnv != nullptr && std::string(poolingEnv) == "1" &&
                                   !clip_cache_path_.empty());
            std::cout << "Thread " << thread_id_ << ": clip pooling backend="
                      << (useCudaClipPooling_ ? "cuda" : "cpu") << std::endl;
            std::cout << "ONNX model loaded successfully for thread " << threadId << " on device " << deviceId << " (CUDA stream: " << cuda_stream_ << ")" << std::endl;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            throw;
        }
    }

    ScoringEngine::~ScoringEngine()
    {
        cudaSetDevice(device_id_);

        // Explicitly release ORT objects that might depend on device buffers
        // to prevent double-free or use-after-free during member destruction
        if (io_binding_)
        {
            io_binding_->ClearBoundInputs();
            io_binding_->ClearBoundOutputs();
            io_binding_.reset();
        }
        session_.reset();

        // free device buffers
        for (size_t i = 0; i < numInputs_; ++i)
            d_inputs_[i].free();
        d_clip_embedding_table_.free();
        d_clip_tag_ids_.free();
        d_clip_tag_offsets_.free();
        if (outputReadyEvent_ != nullptr)
        {
            cudaEventDestroy(outputReadyEvent_);
            outputReadyEvent_ = nullptr;
        }
        if (owns_stream_ && cuda_stream_ != nullptr)
        {
            cudaStreamDestroy(cuda_stream_);
        }
    }

    void ScoringEngine::initSession(const std::string &modelPath)
    {
        cudaSetDevice(device_id_);

        // session options
        // Ort::SessionOptions session_options;
        // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // session_options.SetExecutionMode(ORT_PARALLEL); // Allow inter-op overlap
        // session_options.EnableCpuMemArena();

        Ort::SessionOptions so;
        so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);

        auto parseEnvInt = [](const char *name, int fallback) -> int
        {
            const char *value = std::getenv(name);
            if (value == nullptr || *value == '\0')
                return fallback;
            char *endPtr = nullptr;
            long parsed = std::strtol(value, &endPtr, 10);
            if (endPtr == value || *endPtr != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max())
                return fallback;
            return static_cast<int>(parsed);
        };
        // Conservative defaults for CUDA-dominant sessions.
        int interThreads = parseEnvInt("SDML_ORT_INTEROP_THREADS", 2);
        int intraThreads = parseEnvInt("SDML_ORT_INTRAOP_THREADS", 1);

        so.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        so.SetInterOpNumThreads(interThreads);
        so.SetIntraOpNumThreads(intraThreads);

        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.EnableCpuMemArena();

        const std::string optimizedPath = "optimized_from_cpp_thread_" + std::to_string(thread_id_) + ".onnx";
        so.SetOptimizedModelFilePath(optimizedPath.c_str());

        OrtCUDAProviderOptionsV2 *cuda_v2 = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_v2));

        // ORT 1.23+ expects bulk update via UpdateCUDAProviderOptions(keys, values, count)
        std::vector<const char *> cuda_opt_keys;
        std::vector<const char *> cuda_opt_values;
        auto set_opt = [&](const char *k, const char *v)
        {
            cuda_opt_keys.push_back(k);
            cuda_opt_values.push_back(v);
        };
        // Keep storage for dynamically formatted values
        std::vector<std::string> cuda_opt_values_owned;
        // keep helper for future options; avoid unused warning by using it below
        auto set_opt_owned = [&](const char *k, std::string v)
        {
            cuda_opt_keys.push_back(k);
            cuda_opt_values_owned.push_back(std::move(v));
            cuda_opt_values.push_back(cuda_opt_values_owned.back().c_str());
        };

        set_opt_owned("device_id", std::to_string(device_id_));
        // ensure lambda is referenced so compilers don't warn if config is minimal
        if (false)
            set_opt_owned("noop", "0");
        // Keep CUDA graph disabled for now due to capture errors observed
        set_opt("enable_cuda_graph", "0");
        set_opt("tunable_op_enable", "1");
        set_opt("tunable_op_tuning_enable", "1");
        set_opt("use_tf32", "1");
        // if any convs exist (unlikely)
        set_opt("cudnn_conv_algo_search", "DEFAULT");
        set_opt("do_copy_in_default_stream", "0");
        set_opt("use_ep_level_unified_stream", "1");
        set_opt("arena_extend_strategy", "kSameAsRequested");
        set_opt("cudnn_conv_use_max_workspace", "0");
        // set_opt("enable_blaslt", "1"); // TODO: investigate why it breaks the graph

        if (cuda_stream_ != nullptr)
        {
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptionsWithValue(
                cuda_v2, "user_compute_stream", cuda_stream_));
        }

        // apply collected options, then append provider
        if (!cuda_opt_keys.empty())
        {
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
                cuda_v2,
                cuda_opt_keys.data(),
                cuda_opt_values.data(),
                static_cast<size_t>(cuda_opt_keys.size())));
        }
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda_v2));
        Ort::GetApi().ReleaseCUDAProviderOptions(cuda_v2);

        // try to use CUDA with a dedicated stream (requires ORT CUDA Provider V2)
        // TODO: check compatibility with preivous ORT versions
        try
        {
#if defined(ORT_API_VERSION) && (ORT_API_VERSION >= 12)
            // use legacy CUDA options struct but fill in user stream fields (present in ORT >= 1.12)
            // OrtCUDAProviderOptions cuda_options{};
            // cuda_options.device_id = 0;
            // // following available starting 1.12
            // cuda_options.has_user_compute_stream = 1;
            // cuda_options.user_compute_stream = reinterpret_cast<void *>(cuda_stream_);
            // session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA provider added for thread " << thread_id_ << " with custom stream=" << cuda_stream_ << std::endl;

            // Keep only CUDA EP for parity with Python benchmark

#else
            // fallback when ORT < 1.12: custom stream fields not available
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = device_id_;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "CUDA provider (no custom stream) added for thread " << thread_id_ << " (ORT API < 12)" << std::endl;
#endif
        }
        catch (const std::exception &e)
        {
            std::cout << "CUDA provider not available for thread " << thread_id_ << ", using CPU: " << e.what() << std::endl;
        }

        // create session using global environment
        session_ = std::make_unique<Ort::Session>(g_env, modelPath.c_str(), so);
        io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
        binding_needs_rebuild_ = true;

        // Dynamically discover input names and types from the ONNX model
        try
        {
            Ort::AllocatorWithDefaultOptions allocator;
            numInputs_ = session_->GetInputCount();
            if (numInputs_ > MAX_INPUTS)
            {
                throw std::runtime_error("Model has " + std::to_string(numInputs_) +
                                         " inputs, exceeding MAX_INPUTS=" + std::to_string(MAX_INPUTS));
            }
            input_names_owned_.resize(numInputs_);
            input_names_.resize(numInputs_);
            for (size_t i = 0; i < numInputs_; ++i)
            {
                input_names_owned_[i] = session_->GetInputNameAllocated(i, allocator).get();
                input_names_[i] = input_names_owned_[i].c_str();
                inputNameToIndex_[input_names_owned_[i]] = i;
                Ort::TypeInfo typeInfo = session_->GetInputTypeInfo(i);
                auto ti = typeInfo.GetTensorTypeAndShapeInfo();
                input_elem_types_[i] = ti.GetElementType();
            }
            // Discover output names
            size_t numOutputs = session_->GetOutputCount();
            output_names_.resize(numOutputs);
            // Store output name strings so pointers remain valid
            static thread_local std::vector<std::string> outputNamesOwned;
            outputNamesOwned.resize(numOutputs);
            for (size_t i = 0; i < numOutputs; ++i)
            {
                outputNamesOwned[i] = session_->GetOutputNameAllocated(i, allocator).get();
                output_names_[i] = outputNamesOwned[i].c_str();
            }
            std::cout << "Thread " << thread_id_ << ": Model has " << numInputs_ << " inputs: ";
            for (size_t i = 0; i < numInputs_; ++i)
            {
                std::cout << input_names_owned_[i];
                if (i + 1 < numInputs_)
                    std::cout << ", ";
            }
            std::cout << std::endl;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "ORT Exception during input discovery: " << e.what() << std::endl;
            throw;
        }
    }

    void ScoringEngine::logMetricsIfNeeded()
    {
        if (!metricsEnabled_)
            return;
        auto now = std::chrono::steady_clock::now();
        if ((now - lastMetricsLog_) < metricsLogInterval_)
            return;
        std::ostringstream oss;
        const double batches = metrics_.batches > 0 ? static_cast<double>(metrics_.batches) : 1.0;
        oss << "[ScoringEngine T" << thread_id_ << "] batches=" << metrics_.batches
            << " copy/run/d2h/wait/sigmoid(ms)="
            << std::fixed << std::setprecision(3)
            << (static_cast<double>(metrics_.h2dCopyNs) / 1000000.0) / batches << "/"
            << (static_cast<double>(metrics_.sessionRunNs) / 1000000.0) / batches << "/"
            << (static_cast<double>(metrics_.d2hCopyNs) / 1000000.0) / batches << "/"
            << (static_cast<double>(metrics_.outputWaitNs) / 1000000.0) / batches << "/"
            << (static_cast<double>(metrics_.sigmoidNs) / 1000000.0) / batches
            << " bytes(h2d/d2h)=" << metrics_.h2dBytes << "/" << metrics_.d2hBytes;
        std::cout << oss.str() << std::endl;
        lastMetricsLog_ = now;
    }

    std::vector<float> ScoringEngine::scoreBatch(const PromptBatch &batch)
    {
        if (batch.batch_size == 0)
            return {};

        try
        {
            // ensure device buffers are sized
            ensureDeviceCapacity(batch.batch_size, batch);
            auto copyStart = std::chrono::steady_clock::now();
            copyBatchToDevice(batch);
            metrics_.h2dCopyNs += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - copyStart).count());

            // bind once (if needed) and run
            auto runStart = std::chrono::steady_clock::now();
            rebuildBindingIfNeeded();
            session_->Run(Ort::RunOptions{nullptr}, *io_binding_);
            metrics_.sessionRunNs += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - runStart).count());

            // retrieve output
            auto output_vals = io_binding_->GetOutputValues();
            auto &out_val = output_vals.front();

            std::vector<float> scores(batch.batch_size);

            auto elem_type = out_val.GetTensorTypeAndShapeInfo().GetElementType();

            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                float *device_out = out_val.GetTensorMutableData<float>();
                auto d2hStart = std::chrono::steady_clock::now();
                cudaMemcpyAsync(scores.data(), device_out, batch.batch_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_);
                metrics_.d2hCopyNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - d2hStart).count());
                metrics_.d2hBytes += static_cast<uint64_t>(batch.batch_size * sizeof(float));
            }
            else // assumes fp16
            {
                uint16_t *device_out = out_val.GetTensorMutableData<uint16_t>();
                std::vector<uint16_t> tmp(batch.batch_size);
                auto d2hStart = std::chrono::steady_clock::now();
                cudaMemcpyAsync(tmp.data(), device_out, batch.batch_size * sizeof(uint16_t), cudaMemcpyDeviceToHost, cuda_stream_);
                metrics_.d2hCopyNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - d2hStart).count());
                metrics_.d2hBytes += static_cast<uint64_t>(batch.batch_size * sizeof(uint16_t));
                auto waitStart = std::chrono::steady_clock::now();
                if (outputReadyEvent_ != nullptr)
                {
                    cudaEventRecord(outputReadyEvent_, cuda_stream_);
                    cudaEventSynchronize(outputReadyEvent_);
                }
                else
                {
                    cudaStreamSynchronize(cuda_stream_);
                }
                metrics_.outputWaitNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count());
                for (size_t i = 0; i < batch.batch_size; ++i)
                    scores[i] = half_to_float(tmp[i]);
                auto sigmoidStart = std::chrono::steady_clock::now();
                for (float &s : scores)
                    s = 1.f / (1.f + std::exp(-s));
                metrics_.sigmoidNs += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sigmoidStart).count());
                metrics_.batches++;
                logMetricsIfNeeded();
                return scores;
            }

            auto waitStart = std::chrono::steady_clock::now();
            if (outputReadyEvent_ != nullptr)
            {
                cudaEventRecord(outputReadyEvent_, cuda_stream_);
                cudaEventSynchronize(outputReadyEvent_);
            }
            else
            {
                cudaStreamSynchronize(cuda_stream_);
            }
            metrics_.outputWaitNs += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count());

            // sigmoid
            auto sigmoidStart = std::chrono::steady_clock::now();
            for (float &s : scores)
                s = 1.f / (1.f + std::exp(-s));
            metrics_.sigmoidNs += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sigmoidStart).count());
            metrics_.batches++;
            logMetricsIfNeeded();

            return scores;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "ONNX Runtime inference error (thread " << thread_id_ << "): " << e.what() << std::endl;
            throw;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Inference error (thread " << thread_id_ << "): " << e.what() << std::endl;
            throw;
        }
    }

    void ScoringEngine::runBatch(const PromptBatch &batch)
    {
        if (batch.batch_size == 0)
            return;
        ensureDeviceCapacity(batch.batch_size, batch);
        copyBatchToDevice(batch);
        rebuildBindingIfNeeded();
        session_->Run(Ort::RunOptions{nullptr}, *io_binding_);
    }

    void ScoringEngine::ensureDeviceCapacity(size_t batch, const PromptBatch &batchRef)
    {
        cudaSetDevice(device_id_);

        // Helper to get element size from ONNX type
        auto getElementSize = [](ONNXTensorElementDataType type) -> size_t
        {
            switch (type)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return 4;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                return 2;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return 8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                return 4;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                return 1;
            default:
                return 4; // Fallback
            }
        };

        const int64_t B = static_cast<int64_t>(batch);
        const int64_t clipDim = static_cast<int64_t>(batchRef.clip_dim);

        // Temporary storage for shapes (kept alive until tensor creation is done)
        struct ShapeSpec
        {
            int64_t dims[2];
            size_t rank;
            size_t elemCount;
        };
        std::array<ShapeSpec, MAX_INPUTS> specs{};

        for (size_t i = 0; i < numInputs_; ++i)
        {
            const std::string &name = input_names_owned_[i];
            auto &spec = specs[i];
            // Determine shape and element count by input name
            if (name == "tokens" || name == "clip_emb")
            {
                // clip_embed models reuse the "tokens" input name for the pooled
                // CLIP embedding (float, B×d_clip) instead of one-hot indices
                // (int64, B×MAX_SEQUENCE_LENGTH).  Detect via batch clip_dim.
                if (clipDim > 0)
                {
                    // The ONNX export may have typed this input as INT64 (export
                    // dummy used z_long).  Override to FLOAT16 so that copyFloat
                    // works correctly and the model's internal Cast(INT64→FP16)
                    // becomes a no-op Cast(FP16→FP16).
                    auto &elemType = input_elem_types_[i];
                    if (elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                        elemType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                    {
                        std::cout << "[ScoringEngine] Overriding '" << name
                                  << "' input type from INT to FLOAT16 for clip_embed model"
                                  << std::endl;
                        elemType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
                    }
                    spec = {{B, clipDim}, 2, batch * static_cast<size_t>(clipDim)};
                }
                else
                {
                    spec = {{B, static_cast<int64_t>(MAX_SEQUENCE_LENGTH)}, 2, batch * MAX_SEQUENCE_LENGTH};
                }
            }
            else if (name == "token_mask")
            {
                spec = {{B, static_cast<int64_t>(MAX_SEQUENCE_LENGTH)}, 2, batch * MAX_SEQUENCE_LENGTH};
            }
            else if (name == "lora_ids" || name == "lora_w")
            {
                spec = {{B, MAX_LORAS}, 2, batch * MAX_LORAS};
            }
            else
            {
                // Scalar-per-sample inputs (cfg, n_loras, sampler_id, etc.)
                spec = {{B, 1}, 2, batch};
            }
        }

        bool shapeChanged = false;
        for (size_t i = 0; i < numInputs_; ++i)
        {
            auto &spec = specs[i];
            size_t elemSize = getElementSize(input_elem_types_[i]);
            size_t bytes = spec.elemCount * elemSize;
            if (bytes > d_inputs_[i].bytes)
            {
                d_inputs_[i].free();
                cudaMalloc(&d_inputs_[i].ptr, bytes);
                d_inputs_[i].bytes = bytes;
            }
            // recreate tensor if shape changed or not yet created
            std::vector<int64_t> shapeVec(spec.dims, spec.dims + spec.rank);
            if (d_inputs_[i].shape != shapeVec)
            {
                d_inputs_[i].shape = shapeVec;
                d_inputs_[i].tensor = Ort::Value::CreateTensor(
                    memory_info_cuda_, d_inputs_[i].ptr, bytes,
                    d_inputs_[i].shape.data(), spec.rank,
                    input_elem_types_[i]);
                shapeChanged = true;
            }
        }
        if (shapeChanged)
            binding_needs_rebuild_ = true;
    }

    void ScoringEngine::initClipEmbeddingTableIfNeeded(const PromptBatch &batch)
    {
        if (!useCudaClipPooling_ || clipEmbeddingTableReady_ || batch.clip_dim == 0 || clip_cache_path_.empty())
            return;
        std::ifstream file(clip_cache_path_, std::ios::binary);
        if (!file.is_open())
        {
            const char *err = std::strerror(errno);
            cudaClipPoolingFailureReason_ = "Failed to open clip cache '" + clip_cache_path_ + "': " + (err ? err : "unknown");
            std::cerr << "[ScoringEngine] " << cudaClipPoolingFailureReason_ << std::endl;
            useCudaClipPooling_ = false;
            return;
        }
        uint32_t numEntries = 0;
        uint32_t embedDim = 0;
        file.read(reinterpret_cast<char *>(&numEntries), sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&embedDim), sizeof(uint32_t));
        if (!file)
        {
            cudaClipPoolingFailureReason_ = "Failed to read clip cache header from '" + clip_cache_path_ + "'";
            std::cerr << "[ScoringEngine] " << cudaClipPoolingFailureReason_ << std::endl;
            useCudaClipPooling_ = false;
            return;
        }
        if (embedDim != batch.clip_dim)
        {
            cudaClipPoolingFailureReason_ = "Clip cache dim mismatch: cache=" + std::to_string(embedDim) +
                                           ", batch=" + std::to_string(batch.clip_dim);
            std::cerr << "[ScoringEngine] " << cudaClipPoolingFailureReason_ << std::endl;
            useCudaClipPooling_ = false;
            return;
        }
        std::vector<uint16_t> hostFp16(static_cast<size_t>(numEntries) * static_cast<size_t>(embedDim), 0);
        for (uint32_t i = 0; i < numEntries; ++i)
        {
            uint32_t nameLen = 0;
            file.read(reinterpret_cast<char *>(&nameLen), sizeof(uint32_t));
            if (!file)
            {
                cudaClipPoolingFailureReason_ = "Failed to read clip cache entry " + std::to_string(i) + " header";
                useCudaClipPooling_ = false;
                return;
            }
            std::string tagName(nameLen, '\0');
            file.read(tagName.data(), nameLen);
            if (!file)
            {
                cudaClipPoolingFailureReason_ = "Failed to read clip cache entry " + std::to_string(i) + " tag name";
                useCudaClipPooling_ = false;
                return;
            }
            std::vector<float> emb(embedDim);
            file.read(reinterpret_cast<char *>(emb.data()), static_cast<size_t>(embedDim) * sizeof(float));
            if (!file)
            {
                cudaClipPoolingFailureReason_ = "Failed to read clip cache entry " + std::to_string(i) + " embedding";
                useCudaClipPooling_ = false;
                return;
            }
            const size_t rowBase = static_cast<size_t>(i) * static_cast<size_t>(embedDim);
            fp32_to_fp16_f16c(emb.data(), hostFp16.data() + rowBase, static_cast<size_t>(embedDim));
        }
        d_clip_embedding_table_.free();
        const size_t bytes = hostFp16.size() * sizeof(uint16_t);
        cudaMalloc(&d_clip_embedding_table_.ptr, bytes);
        cudaMemcpyAsync(d_clip_embedding_table_.ptr, hostFp16.data(), bytes, cudaMemcpyHostToDevice, cuda_stream_);
        metrics_.h2dBytes += static_cast<uint64_t>(bytes);
        d_clip_embedding_table_.bytes = bytes;
        clipEmbeddingRows_ = static_cast<size_t>(numEntries);
        clipEmbeddingDim_ = static_cast<size_t>(embedDim);
        clipEmbeddingTableReady_ = true;
    }

    bool ScoringEngine::tryRunCudaClipPooling(size_t clipInputIdx, const PromptBatch &batch)
    {
        if (!useCudaClipPooling_ || !clipEmbeddingTableReady_ || batch.clip_dim == 0 ||
            batch.clip_tag_offsets.size() != (batch.batch_size + 1))
        {
            if (useCudaClipPooling_ && !clipEmbeddingTableReady_ && cudaClipPoolingFailureReason_.empty())
                cudaClipPoolingFailureReason_ = "Clip embedding table not initialized (init failed earlier)";
            return false;
        }
        const size_t tagIdBytes = batch.clip_tag_ids_flat.size() * sizeof(uint32_t);
        const size_t offsetBytes = batch.clip_tag_offsets.size() * sizeof(uint32_t);
        if (tagIdBytes > d_clip_tag_ids_.bytes)
        {
            d_clip_tag_ids_.free();
            cudaMalloc(&d_clip_tag_ids_.ptr, tagIdBytes);
            d_clip_tag_ids_.bytes = tagIdBytes;
        }
        if (offsetBytes > d_clip_tag_offsets_.bytes)
        {
            d_clip_tag_offsets_.free();
            cudaMalloc(&d_clip_tag_offsets_.ptr, offsetBytes);
            d_clip_tag_offsets_.bytes = offsetBytes;
        }
        if (tagIdBytes > 0)
        {
            cudaMemcpyAsync(d_clip_tag_ids_.ptr, batch.clip_tag_ids_flat.data(), tagIdBytes,
                            cudaMemcpyHostToDevice, cuda_stream_);
            metrics_.h2dBytes += static_cast<uint64_t>(tagIdBytes);
        }
        cudaMemcpyAsync(d_clip_tag_offsets_.ptr, batch.clip_tag_offsets.data(), offsetBytes,
                        cudaMemcpyHostToDevice, cuda_stream_);
        metrics_.h2dBytes += static_cast<uint64_t>(offsetBytes);
        if (input_elem_types_[clipInputIdx] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            cudaClipPoolingFailureReason_ = "Model clip input is not FP16 (unsupported for CUDA pooling)";
            return false;
        }
        std::string kernelError;
        const bool ok = launchClipPoolingKernel(
            static_cast<const uint32_t *>(d_clip_tag_ids_.ptr),
            static_cast<const uint32_t *>(d_clip_tag_offsets_.ptr),
            batch.batch_size,
            static_cast<const uint16_t *>(d_clip_embedding_table_.ptr),
            clipEmbeddingRows_,
            clipEmbeddingDim_,
            static_cast<uint16_t *>(d_inputs_[clipInputIdx].ptr),
            cuda_stream_,
            &kernelError);
        if (!ok)
            cudaClipPoolingFailureReason_ = kernelError.empty()
                                               ? "CUDA kernel launch failed (check nvcc arch vs GPU)"
                                               : std::string("CUDA kernel launch failed: ") + kernelError;
        return ok;
    }

    void ScoringEngine::copyBatchToDevice(const PromptBatch &batch)
    {
        // Helper for int64/int32 inputs
        auto copyInt = [&](size_t idx, const std::vector<int64_t> &src)
        {
            if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                cudaMemcpyAsync(d_inputs_[idx].ptr, src.data(), d_inputs_[idx].bytes,
                                cudaMemcpyHostToDevice, cuda_stream_);
                metrics_.h2dBytes += static_cast<uint64_t>(d_inputs_[idx].bytes);
            }
            else if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
            {
                auto &tmp = h_staging_int32_[idx];
                tmp.resize(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                    tmp[i] = static_cast<int32_t>(src[i]);
                cudaMemcpyAsync(d_inputs_[idx].ptr, tmp.data(), tmp.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice, cuda_stream_);
                metrics_.h2dBytes += static_cast<uint64_t>(tmp.size() * sizeof(int32_t));
            }
        };

        // Helper for float/float16 inputs
        auto copyFloat = [&](size_t idx, const std::vector<float> &src)
        {
            size_t elems = src.size();
            if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                cudaMemcpyAsync(d_inputs_[idx].ptr, src.data(), elems * sizeof(float),
                                cudaMemcpyHostToDevice, cuda_stream_);
                metrics_.h2dBytes += static_cast<uint64_t>(elems * sizeof(float));
            }
            else if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            {
                auto &staging = h_staging_fp16_[idx];
                staging.ensure(elems);
                fp32_to_fp16_f16c(src.data(), staging.p, elems);
                cudaMemcpyAsync(d_inputs_[idx].ptr, staging.p, d_inputs_[idx].bytes,
                                cudaMemcpyHostToDevice, cuda_stream_);
                metrics_.h2dBytes += static_cast<uint64_t>(d_inputs_[idx].bytes);
            }
        };
        // Helper for host-FP16 clip embeddings (FP16-first host path)
        auto copyHalf = [&](size_t idx, const std::vector<uint16_t> &src)
        {
            size_t elems = src.size();
            if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            {
                cudaMemcpyAsync(d_inputs_[idx].ptr, src.data(), elems * sizeof(uint16_t),
                                cudaMemcpyHostToDevice, cuda_stream_);
                metrics_.h2dBytes += static_cast<uint64_t>(elems * sizeof(uint16_t));
            }
            else if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                auto &staging = h_staging_fp32_[idx];
                staging.ensure(elems);
                for (size_t i = 0; i < elems; ++i)
                    staging.p[i] = half_to_float(src[i]);
                cudaMemcpyAsync(d_inputs_[idx].ptr, staging.p, elems * sizeof(float),
                                cudaMemcpyHostToDevice, cuda_stream_);
                metrics_.h2dBytes += static_cast<uint64_t>(elems * sizeof(float));
            }
        };

        // Helper: find index by name, returns SIZE_MAX if not found
        auto findIdx = [&](const char *name) -> size_t
        {
            auto it = inputNameToIndex_.find(name);
            return it != inputNameToIndex_.end() ? it->second : SIZE_MAX;
        };

        // Dispatch by name
        size_t idx;
        bool clipInputHandledByCudaPooling = false;
        if (batch.clip_dim > 0 && useCudaClipPooling_)
        {
            initClipEmbeddingTableIfNeeded(batch);
        }
        // tokens / clip_emb: the ONNX input may be called "tokens" in both
        // the one-hot and clip_embed models, or "clip_emb" in clip_embed only.
        if ((idx = findIdx("tokens")) != SIZE_MAX)
        {
            if (batch.clip_dim > 0)
            {
                if (tryRunCudaClipPooling(idx, batch))
                    clipInputHandledByCudaPooling = true;
                else
                {
                    if (batch.clip_emb.empty())
                    {
                        std::string msg = "CUDA clip pooling unavailable and no host clip embedding fallback present";
                        if (!cudaClipPoolingFailureReason_.empty())
                            msg += " (" + cudaClipPoolingFailureReason_ + ")";
                        throw std::runtime_error(msg);
                    }
                    copyHalf(idx, batch.clip_emb); // clip_embed: FP16 host embedding
                }
            }
            else
                copyInt(idx, batch.token_indices); // one-hot: int64 token IDs
        }
        if ((idx = findIdx("clip_emb")) != SIZE_MAX)
        {
            if (batch.clip_dim > 0 && clipInputHandledByCudaPooling)
            {
                // Already written via pooled "tokens" input.
            }
            else if (batch.clip_dim > 0)
            {
                if (!tryRunCudaClipPooling(idx, batch))
                {
                    if (batch.clip_emb.empty())
                    {
                        std::string msg = "CUDA clip pooling unavailable and no host clip embedding fallback present";
                        if (!cudaClipPoolingFailureReason_.empty())
                            msg += " (" + cudaClipPoolingFailureReason_ + ")";
                        throw std::runtime_error(msg);
                    }
                    copyHalf(idx, batch.clip_emb);
                }
            }
            else
            {
                copyHalf(idx, batch.clip_emb);
            }
        }
        if ((idx = findIdx("token_mask")) != SIZE_MAX)
            copyFloat(idx, batch.token_mask);
        // shared inputs
        if ((idx = findIdx("lora_ids")) != SIZE_MAX)
            copyInt(idx, batch.lora_ids);
        if ((idx = findIdx("lora_w")) != SIZE_MAX)
            copyFloat(idx, batch.lora_weights);
        if ((idx = findIdx("cfg")) != SIZE_MAX)
            copyFloat(idx, batch.cfg_scales);
        if ((idx = findIdx("n_loras")) != SIZE_MAX)
            copyFloat(idx, batch.num_loras);
        if ((idx = findIdx("sampler_id")) != SIZE_MAX)
            copyInt(idx, batch.sampler_ids);
        if ((idx = findIdx("steps_log")) != SIZE_MAX)
            copyFloat(idx, batch.steps_log);
        if ((idx = findIdx("steps_bucket")) != SIZE_MAX)
            copyInt(idx, batch.steps_bucket);
        if ((idx = findIdx("upscaler_id")) != SIZE_MAX)
            copyInt(idx, batch.upscaler_ids);
        if ((idx = findIdx("up_has")) != SIZE_MAX)
            copyFloat(idx, batch.up_has);
        if ((idx = findIdx("up_steps")) != SIZE_MAX)
            copyFloat(idx, batch.up_steps);
        if ((idx = findIdx("denoise")) != SIZE_MAX)
            copyFloat(idx, batch.denoise);
        if ((idx = findIdx("model_id")) != SIZE_MAX)
            copyInt(idx, batch.model_ids);
    }

} // namespace generator
