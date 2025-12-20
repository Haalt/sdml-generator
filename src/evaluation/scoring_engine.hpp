#pragma once

#include <chrono>
#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/types.hpp"

inline float half_to_float(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp == 0)
    {
        if (mant == 0)
        {
            uint32_t val = sign;
            float result;
            std::memcpy(&result, &val, sizeof(result));
            return result;
        }
        // subnormals
        while ((mant & 0x0400) == 0)
        {
            mant <<= 1;
            exp -= 1;
        }
        exp += 1;
        mant &= ~0x0400;
    }
    else if (exp == 31)
    {
        uint32_t val = sign | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &val, sizeof(result));
        return result;
    }

    exp = exp + (127 - 15);
    uint32_t val = sign | (exp << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &val, sizeof(result));
    return result;
}

inline void fp32_to_fp16_f16c(const float *src, uint16_t *dst, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(dst + i), h);
    }
    for (; i < n; ++i)
    {
        __m128 f = _mm_set_ss(src[i]);
        __m128i h = _mm_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        dst[i] = (uint16_t)_mm_extract_epi16(h, 0);
    }
}

namespace generator
{

    struct Pinned16
    {
        uint16_t *p = nullptr;
        size_t cap = 0;
        ~Pinned16()
        {
            if (p)
                cudaFreeHost(p);
        }
        void ensure(size_t n)
        {
            if (n > cap)
            {
                if (p)
                    cudaFreeHost(p);
                cudaHostAlloc(&p, n * sizeof(uint16_t), 0);
                cap = n;
            }
        }
    };
    struct Pinned32
    {
        float *p = nullptr;
        size_t cap = 0;
        ~Pinned32()
        {
            if (p)
                cudaFreeHost(p);
        }
        void ensure(size_t n)
        {
            if (n > cap)
            {
                if (p)
                    cudaFreeHost(p);
                cudaHostAlloc(&p, n * sizeof(float), 0);
                cap = n;
            }
        }
    };
    // global ONNX Runtime environment (shared across all sessions)
    extern Ort::Env g_env;

    // CUDA stream manager for parallel GPU execution
    class CudaStreamManager
    {
    public:
        static CudaStreamManager &getInstance();
        cudaStream_t getStream(int threadId);
        ~CudaStreamManager();

    private:
        CudaStreamManager();
        static constexpr int MAX_STREAMS = 16;
        cudaStream_t streams_[MAX_STREAMS];
        bool initialized_;
    };

    // device buffer holding GPU memory and its Ort::Value wrapper
    struct DeviceBuffer
    {
        void *ptr = nullptr;        // cudaMalloc pointer
        size_t bytes = 0;           // allocated size in bytes
        std::vector<int64_t> shape; // cached tensor shape
        Ort::Value tensor{};        // Ort wrapper (does not own ptr)

        void free()
        {
            if (ptr)
            {
                cudaFree(ptr);
                ptr = nullptr;
                bytes = 0;
            }
        }
    };

    /**
     * @brief Engine for scoring prompts using the ONNX model
     *
     * Supports both one-hot (14 inputs) and clip_embed (13 inputs) models.
     * Input layout is discovered dynamically from the ONNX model at init time.
     */
    class ScoringEngine
    {
    public:
        /**
         * @brief Constructs a new ScoringEngine instance
         * @param modelPath Path to the ONNX model file
         * @param threadId Thread ID for CUDA stream assignment
         * @param deviceId CUDA device ID
         */
        explicit ScoringEngine(const std::string &modelPath, int threadId = 0, int deviceId = 0,
                               const std::string &clipCachePath = "");

        ~ScoringEngine();

        /**
         * @brief Scores a batch of prompts
         * @param batch The batch of prompts to score
         * @return Vector of scores for each prompt in the batch
         */
        std::vector<float> scoreBatch(const PromptBatch &batch);

        /**
         * @brief Runs inference for a batch without fetching outputs (for pure throughput timing)
         * @param batch The batch of prompts to score
         */
        void runBatch(const PromptBatch &batch);

        static constexpr size_t MAX_INPUTS = 16;

    private:
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_;
        Ort::MemoryInfo memory_info_cuda_;

        // Dynamic input layout (discovered from ONNX model)
        size_t numInputs_{0};
        std::vector<std::string> input_names_owned_; // owns the strings
        std::vector<const char *> input_names_;      // C-string pointers for ORT
        std::vector<const char *> output_names_;

        // Name-to-index map for dispatch in copyBatchToDevice
        std::unordered_map<std::string, size_t> inputNameToIndex_;

        std::unique_ptr<Ort::IoBinding> io_binding_;
        bool binding_needs_rebuild_ = true;
        cudaStream_t cuda_stream_;
        bool owns_stream_{false};
        int thread_id_;
        int device_id_;
        std::string clip_cache_path_;
        bool useCudaClipPooling_{false};
        bool clipEmbeddingTableReady_{false};
        size_t clipEmbeddingRows_{0};
        size_t clipEmbeddingDim_{0};
        std::string cudaClipPoolingFailureReason_;

        // device buffers for inputs (dynamic count)
        std::array<DeviceBuffer, MAX_INPUTS> d_inputs_;
        DeviceBuffer d_clip_embedding_table_;
        DeviceBuffer d_clip_tag_ids_;
        DeviceBuffer d_clip_tag_offsets_;

        // pinned host staging buffers for FP16 inputs
        std::array<Pinned16, MAX_INPUTS> h_staging_fp16_;
        // pinned host staging buffers for FP32 conversion from FP16 sources
        std::array<Pinned32, MAX_INPUTS> h_staging_fp32_;
        std::array<std::vector<int32_t>, MAX_INPUTS> h_staging_int32_;
        // expected input element types queried from the model
        std::array<ONNXTensorElementDataType, MAX_INPUTS> input_elem_types_{};
        cudaEvent_t outputReadyEvent_{nullptr};
        bool metricsEnabled_{false};
        std::chrono::milliseconds metricsLogInterval_{2000};
        struct RuntimeMetrics
        {
            uint64_t batches{0};
            uint64_t h2dCopyNs{0};
            uint64_t sessionRunNs{0};
            uint64_t d2hCopyNs{0};
            uint64_t outputWaitNs{0};
            uint64_t sigmoidNs{0};
            uint64_t h2dBytes{0};
            uint64_t d2hBytes{0};
        } metrics_{};
        std::chrono::steady_clock::time_point lastMetricsLog_{};

        /**
         * @brief Initializes the ONNX session
         * @param modelPath Path to the ONNX model file
         */
        void initSession(const std::string &modelPath);

        // alloc / resize device buffers to hold batch elements
        void ensureDeviceCapacity(size_t batch, const PromptBatch &batchRef);

        // copy current batch from host vectors to device buffers
        void copyBatchToDevice(const PromptBatch &batch);
        void initClipEmbeddingTableIfNeeded(const PromptBatch &batch);
        bool tryRunCudaClipPooling(size_t clipInputIdx, const PromptBatch &batch);
        void logMetricsIfNeeded();

        // (re)bind inputs/outputs once when shapes change
        void rebuildBindingIfNeeded();
    };

} // namespace generator