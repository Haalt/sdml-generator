#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
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
     */
    class ScoringEngine
    {
    public:
        /**
         * @brief Constructs a new ScoringEngine instance
         * @param modelPath Path to the ONNX model file
         * @param threadId Thread ID for CUDA stream assignment
         */
        explicit ScoringEngine(const std::string &modelPath, int threadId = 0);

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

    private:
        std::unique_ptr<Ort::Session> session_;
        Ort::MemoryInfo memory_info_;
        Ort::MemoryInfo memory_info_cuda_;
        std::vector<const char *> input_names_;
        std::vector<const char *> output_names_;
        std::unique_ptr<Ort::IoBinding> io_binding_;
        bool binding_needs_rebuild_ = true;
        cudaStream_t cuda_stream_;
        int thread_id_;

        // device buffers for the 13 inputs (fixed ordering following input_names_)
        std::array<DeviceBuffer, 13> d_inputs_;
        // ORT allocates outputs on device based on model infered dtype/shape

        // pinned host staging buffers for FP16 inputs (reused every iteration)
        std::array<Pinned16, 13> h_staging_fp16_;
        // expected input element types queried from the model
        std::array<ONNXTensorElementDataType, 13> input_elem_types_{};

        /**
         * @brief Initializes the ONNX session
         * @param modelPath Path to the ONNX model file
         */
        void initSession(const std::string &modelPath);

        // alloc / resize device buffers to hold batch elements
        void ensureDeviceCapacity(size_t batch);

        // copy current batch from host vectors to device buffers
        void copyBatchToDevice(const PromptBatch &batch);

        // (re)bind inputs/outputs once when shapes change (shouldn't happen)
        void rebuildBindingIfNeeded();
    };

} // namespace generator