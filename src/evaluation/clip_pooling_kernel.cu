#include "evaluation/clip_pooling_kernel.hpp"

#include <cuda_fp16.h>
#include <string>

namespace generator
{
    namespace
    {
        __global__ void clipPoolingKernel(const uint32_t *tagIdsFlat,
                                          const uint32_t *tagOffsets,
                                          const __half *embeddingTable,
                                          size_t embeddingRows,
                                          size_t embedDim,
                                          __half *output,
                                          size_t totalElems)
        {
            const size_t globalIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (globalIdx >= totalElems)
                return;

            const size_t sampleIdx = globalIdx / embedDim;
            const size_t dimIdx = globalIdx % embedDim;
            const uint32_t begin = tagOffsets[sampleIdx];
            const uint32_t end = tagOffsets[sampleIdx + 1];

            float sum = 0.0f;
            uint32_t count = 0;
            for (uint32_t i = begin; i < end; ++i)
            {
                const uint32_t tagId = tagIdsFlat[i];
                if (static_cast<size_t>(tagId) >= embeddingRows)
                    continue;
                const size_t rowOffset = static_cast<size_t>(tagId) * embedDim + dimIdx;
                sum += __half2float(embeddingTable[rowOffset]);
                ++count;
            }

            output[globalIdx] = (count > 0) ? __float2half_rn(sum / static_cast<float>(count))
                                            : __float2half_rn(0.0f);
        }
    } // namespace

    bool launchClipPoolingKernel(const uint32_t *dTagIdsFlat,
                                 const uint32_t *dTagOffsets,
                                 size_t batchSize,
                                 const uint16_t *dEmbeddingTableFp16,
                                 size_t embeddingRows,
                                 size_t embedDim,
                                 uint16_t *dOutputPooledFp16,
                                 cudaStream_t stream,
                                 std::string *outError)
    {
        if (dTagIdsFlat == nullptr || dTagOffsets == nullptr || dEmbeddingTableFp16 == nullptr ||
            dOutputPooledFp16 == nullptr || batchSize == 0 || embedDim == 0)
        {
            if (outError)
                *outError = "Invalid arguments (null pointer or zero batch/dim)";
            return false;
        }

        const size_t totalElems = batchSize * embedDim;
        constexpr int blockSize = 256;
        const int gridSize = static_cast<int>((totalElems + blockSize - 1) / blockSize);

        clipPoolingKernel<<<gridSize, blockSize, 0, stream>>>(
            dTagIdsFlat,
            dTagOffsets,
            reinterpret_cast<const __half *>(dEmbeddingTableFp16),
            embeddingRows,
            embedDim,
            reinterpret_cast<__half *>(dOutputPooledFp16),
            totalElems);

        cudaError_t err = cudaGetLastError();
        if (outError && err != cudaSuccess)
            *outError = std::string(cudaGetErrorString(err));
        return err == cudaSuccess;
    }
} // namespace generator
