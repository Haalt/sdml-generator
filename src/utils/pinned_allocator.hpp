#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace generator
{

    // Simple allocator that uses cudaHostAlloc to obtain page-locked (pinned) host memory.
    // It fulfils the minimal requirements of an STL allocator, enough for std::vector.
    // Pinned memory lets cudaMemcpyAsync DMA directly without an internal bounce buffer.

    template <class T>
    class CudaPinnedAllocator
    {
    public:
        using value_type = T;

        CudaPinnedAllocator() noexcept = default;
        template <class U>
        constexpr CudaPinnedAllocator(const CudaPinnedAllocator<U> &) noexcept {}

        [[nodiscard]] T *allocate(std::size_t n)
        {
            if (n == 0)
                return nullptr;
            void *ptr = nullptr;
            cudaError_t err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocPortable);
            if (err != cudaSuccess)
            {
                throw std::bad_alloc();
            }
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, std::size_t) noexcept
        {
            if (p)
                cudaFreeHost(p);
        }
    };

    template <class T, class U>
    bool operator==(const CudaPinnedAllocator<T> &, const CudaPinnedAllocator<U> &) { return true; }

    template <class T, class U>
    bool operator!=(const CudaPinnedAllocator<T> &, const CudaPinnedAllocator<U> &) { return false; }

    // Convenience alias
    template <typename T>
    using PinnedVector = std::vector<T, CudaPinnedAllocator<T>>;

} // namespace generator