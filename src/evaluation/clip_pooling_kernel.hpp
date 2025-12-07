#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace generator
{
    bool launchClipPoolingKernel(const uint32_t *dTagIdsFlat,
                                 const uint32_t *dTagOffsets,
                                 size_t batchSize,
                                 const uint16_t *dEmbeddingTableFp16,
                                 size_t embeddingRows,
                                 size_t embedDim,
                                 uint16_t *dOutputPooledFp16,
                                 cudaStream_t stream,
                                 std::string *outError = nullptr);
} // namespace generator
