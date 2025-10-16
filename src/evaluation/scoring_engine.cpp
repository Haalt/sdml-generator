#include "evaluation/scoring_engine.hpp"
#include "onnxruntime_cxx_api.h"
#include <cmath>
#include <iostream>
#include <onnxruntime_c_api.h>
#include <stdexcept>
#include <vector>

namespace generator
{
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

        for (int i = 0; i < 13; ++i)
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

    ScoringEngine::ScoringEngine(const std::string &modelPath, int threadId)
        : memory_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)),
          memory_info_cuda_(Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault)),
          thread_id_(threadId)
    {
        try
        {
            // get dedicated CUDA stream for this thread
            cuda_stream_ = CudaStreamManager::getInstance().getStream(threadId);

            initSession(modelPath);
            std::cout << "ONNX model loaded successfully for thread " << threadId << " (CUDA stream: " << cuda_stream_ << ")" << std::endl;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            throw;
        }
    }

    ScoringEngine::~ScoringEngine()
    {
        // free device buffers
        for (auto &buf : d_inputs_)
            buf.free();
    }

    void ScoringEngine::initSession(const std::string &modelPath)
    {
        // session options
        // Ort::SessionOptions session_options;
        // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // session_options.SetExecutionMode(ORT_PARALLEL); // Allow inter-op overlap
        // session_options.EnableCpuMemArena();

        Ort::SessionOptions so;
        so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
        // so.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetInterOpNumThreads(1);
        so.SetIntraOpNumThreads(1);

        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.EnableCpuMemArena();

        so.SetOptimizedModelFilePath("optimized_from_cpp.onnx");

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

        set_opt("device_id", "0");
        // ensure lambda is referenced so compilers don't warn if config is minimal
        if (false)
            set_opt_owned("noop", "0");
        // Keep CUDA graph disabled for now due to capture errors observed
        set_opt("enable_cuda_graph", "0");
        set_opt("tunable_op_enable", "1");
        set_opt("tunable_op_tuning_enable", "1");
        set_opt("use_tf32", "1");
        // if any convs exist (unlikely)
        set_opt("cudnn_conv_algo_search", "EXHAUSTIVE");
        set_opt("do_copy_in_default_stream", "0");
        set_opt("use_ep_level_unified_stream", "1");
        // set_opt("enable_blaslt", "1"); // TODO: investigate why it breaks the graph

        cudaStream_t s = CudaStreamManager::getInstance().getStream(thread_id_);
        (void)Ort::GetApi().UpdateCUDAProviderOptionsWithValue(cuda_v2, "user_compute_stream", s);

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
            cuda_options.device_id = 0;
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

        // define input/output names
        input_names_ = {
            "tokens",
            "token_mask",
            "lora_ids",
            "lora_w",
            "cfg",
            "n_loras",
            "sampler_id",
            "steps_log",
            "steps_bucket",
            "upscaler_id",
            "up_has",
            "up_steps",
            "denoise"};

        output_names_ = {"output"};

        // outputs will be device-allocated by ORT (dtype/shape inferred from model)

        // query expected input element types from model to avoid implicit ORT casts
        try
        {
            size_t n_in = session_->GetInputCount();
            Ort::AllocatorWithDefaultOptions a;
            for (size_t i = 0; i < n_in && i < input_names_.size(); ++i)
            {
                auto ti = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
                input_elem_types_[i] = ti.GetElementType();
            }
        }
        catch (...)
        {
            // leave defaults
        }
    }

    std::vector<float> ScoringEngine::scoreBatch(const PromptBatch &batch)
    {
        if (batch.batch_size == 0)
            return {};

        try
        {
            // ensure device buffers are sized
            ensureDeviceCapacity(batch.batch_size);

            copyBatchToDevice(batch);

            // bind once (if needed) and run
            rebuildBindingIfNeeded();
            session_->Run(Ort::RunOptions{nullptr}, *io_binding_);

            // retrieve output
            auto output_vals = io_binding_->GetOutputValues();
            auto &out_val = output_vals.front();

            std::vector<float> scores(batch.batch_size);

            auto elem_type = out_val.GetTensorTypeAndShapeInfo().GetElementType();

            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                float *device_out = out_val.GetTensorMutableData<float>();
                cudaMemcpyAsync(scores.data(), device_out, batch.batch_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_);
                cudaStreamSynchronize(cuda_stream_);
            }
            else // assumes fp16
            {
                uint16_t *device_out = out_val.GetTensorMutableData<uint16_t>();
                std::vector<uint16_t> tmp(batch.batch_size);
                cudaMemcpyAsync(tmp.data(), device_out, batch.batch_size * sizeof(uint16_t), cudaMemcpyDeviceToHost, cuda_stream_);
                cudaStreamSynchronize(cuda_stream_);
                for (size_t i = 0; i < batch.batch_size; ++i)
                    scores[i] = half_to_float(tmp[i]);
            }

            // sigmoid
            // custom CUDA kernel to compute on device ?
            for (float &s : scores)
                s = 1.f / (1.f + std::exp(-s));

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
        ensureDeviceCapacity(batch.batch_size);
        copyBatchToDevice(batch);
        rebuildBindingIfNeeded();
        session_->Run(Ort::RunOptions{nullptr}, *io_binding_);
    }

    void ScoringEngine::ensureDeviceCapacity(size_t batch)
    {
        // helper lambda to compute bytes
        auto need_bytes = [](size_t elems, size_t element_size)
        { return elems * element_size; };

        const size_t tokElems = batch * MAX_SEQUENCE_LENGTH;
        const size_t loraElems = batch * MAX_LORAS;
        const size_t vecElems = batch;

        // mapping per input index to element count and sizes/types
        size_t elemCounts[13] = {
            tokElems,  // tokens int64
            tokElems,  // mask fp16
            loraElems, // lora_ids int64
            loraElems, // lora_w fp16
            vecElems,  // cfg fp16
            vecElems,  // n_loras fp16
            vecElems,  // sampler_id int64
            vecElems,  // steps_log fp16
            vecElems,  // steps_bucket int64
            vecElems,  // upscaler_id int64
            vecElems,  // up_has fp16
            vecElems,  // up_steps fp16
            vecElems   // denoise fp16
        };

        size_t elemSizes[13] = {
            sizeof(int64_t),  // tokens
            sizeof(uint16_t), // mask (fp16)
            sizeof(int64_t),  // lora_ids
            sizeof(uint16_t), // lora_w (fp16)
            sizeof(uint16_t), // cfg (fp16)
            sizeof(uint16_t), // n_loras (fp16)
            sizeof(int64_t),  // sampler
            sizeof(uint16_t), // steps_log (fp16)
            sizeof(int64_t),  // steps_bucket
            sizeof(int64_t),  // upscaler
            sizeof(uint16_t), // up_has (fp16)
            sizeof(uint16_t), // up_steps (fp16)
            sizeof(uint16_t)  // denoise (fp16)
        };

        ONNXTensorElementDataType elemTypes[13] = {
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // tokens
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // mask
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // lora_ids
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // lora_w
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // cfg
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // n_loras
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // sampler
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // steps_log
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // steps_bucket
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   // upscaler
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // up_has
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, // up_steps
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16  // denoise
        };

        // shapes per input
        const int64_t B = static_cast<int64_t>(batch);
        int64_t tokShape[2] = {B, MAX_SEQUENCE_LENGTH};
        int64_t loraShape[2] = {B, MAX_LORAS};
        int64_t vecShape[2] = {B, 1};

        const int64_t *shapes[13] = {
            tokShape, tokShape, loraShape, loraShape,
            vecShape, vecShape, vecShape, vecShape,
            vecShape, vecShape, vecShape, vecShape, vecShape};
        size_t rank[13] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

        bool shape_changed = false;
        for (int i = 0; i < 13; ++i)
        {
            size_t bytes = need_bytes(elemCounts[i], elemSizes[i]);
            if (bytes > d_inputs_[i].bytes)
            {
                d_inputs_[i].free();
                cudaMalloc(&d_inputs_[i].ptr, bytes);
                d_inputs_[i].bytes = bytes;
            }

            // recreate tensor if shape changed or not yet created
            if (d_inputs_[i].shape.empty() || d_inputs_[i].shape[0] != B)
            {
                d_inputs_[i].shape.assign(shapes[i], shapes[i] + rank[i]);
                d_inputs_[i].tensor = Ort::Value::CreateTensor(
                    memory_info_cuda_, d_inputs_[i].ptr, bytes, d_inputs_[i].shape.data(), rank[i],
                    elemTypes[i]);
                shape_changed = true;
            }
        }

        if (shape_changed)
            binding_needs_rebuild_ = true;
    }

    void ScoringEngine::copyBatchToDevice(const PromptBatch &batch)
    {
        // int64 inputs: direct copy (match model dtype, handle INT32 fallback if ever used)
        auto copy_i64 = [&](int idx, const std::vector<int64_t> &src)
        {
            if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            {
                cudaMemcpyAsync(d_inputs_[idx].ptr, src.data(), d_inputs_[idx].bytes, cudaMemcpyHostToDevice, cuda_stream_);
            }
            // else if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
            // {
            //     // pack into staging int32
            //     size_t elems = src.size();
            //     if (h_staging_fp16_[idx].size() < elems)
            //         h_staging_fp16_[idx].resize(elems); // reuse the container memory; we won't read as fp16
            //     std::vector<int32_t> tmp;
            //     tmp.resize(elems);
            //     for (size_t i = 0; i < elems; ++i)
            //         tmp[i] = static_cast<int32_t>(src[i]);
            //     cudaMemcpyAsync(d_inputs_[idx].ptr, tmp.data(), elems * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream_);
            // }
            else // assume FP16
            {
                cudaMemcpyAsync(d_inputs_[idx].ptr, src.data(), d_inputs_[idx].bytes, cudaMemcpyHostToDevice, cuda_stream_);
            }
        };

        copy_i64(0, batch.token_indices);
        copy_i64(2, batch.lora_ids);
        copy_i64(6, batch.sampler_ids);
        copy_i64(8, batch.steps_bucket);
        copy_i64(9, batch.upscaler_ids);

        // fp16 inputs: convert then copy (reuse pinned staging buffers)
        auto convert_and_copy = [&](int idx, const std::vector<float> &src)
        {
            size_t elems = src.size();
            if (input_elem_types_[idx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                // send fp32 directly
                cudaMemcpyAsync(d_inputs_[idx].ptr, src.data(), elems * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_);
            }
            else
            {
                // convert to fp16 using pinned staging
                // auto &staging = h_staging_fp16_[idx];
                // if (staging.size() < elems)
                //     staging.resize(elems);
                // fp32_to_fp16_f16c(src.data(), staging.data(), elems);
                // cudaMemcpyAsync(d_inputs_[idx].ptr, staging.data(), d_inputs_[idx].bytes, cudaMemcpyHostToDevice, cuda_stream_);
                auto &staging = h_staging_fp16_[idx];
                staging.ensure(elems);
                fp32_to_fp16_f16c(src.data(), staging.p, elems);
                cudaMemcpyAsync(d_inputs_[idx].ptr, staging.p, d_inputs_[idx].bytes,
                                cudaMemcpyHostToDevice, cuda_stream_);
            }
        };

        convert_and_copy(1, batch.token_mask);
        convert_and_copy(3, batch.lora_weights);
        convert_and_copy(4, batch.cfg_scales);
        convert_and_copy(5, batch.num_loras);
        convert_and_copy(7, batch.steps_log);
        convert_and_copy(10, batch.up_has);
        convert_and_copy(11, batch.up_steps);
        convert_and_copy(12, batch.denoise);
    }

} // namespace generator
