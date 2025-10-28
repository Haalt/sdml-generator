#pragma once

#include <condition_variable>
#include <cstddef>
#include <chrono>
#include <deque>
#include <mutex>
#include <utility>

namespace generator
{
    template <typename T>
    class BlockingBoundedQueue
    {
    public:
        explicit BlockingBoundedQueue(size_t capacity) : capacity_(capacity > 0 ? capacity : 1) {}

        BlockingBoundedQueue(const BlockingBoundedQueue &) = delete;
        BlockingBoundedQueue &operator=(const BlockingBoundedQueue &) = delete;

        bool enqueue(T value)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cvNotFull_.wait(lock, [this]()
                            { return closed_ || queue_.size() < capacity_; });
            if (closed_)
                return false;
            queue_.emplace_back(std::move(value));
            cvNotEmpty_.notify_one();
            return true;
        }

        bool tryEnqueue(T value)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || queue_.size() >= capacity_)
                return false;
            queue_.emplace_back(std::move(value));
            cvNotEmpty_.notify_one();
            return true;
        }

        bool waitDequeue(T &out)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cvNotEmpty_.wait(lock, [this]()
                             { return closed_ || !queue_.empty(); });
            if (queue_.empty())
                return false;
            out = std::move(queue_.front());
            queue_.pop_front();
            cvNotFull_.notify_one();
            return true;
        }

        template <class Rep, class Period>
        bool waitDequeueFor(T &out, const std::chrono::duration<Rep, Period> &timeout)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cvNotEmpty_.wait_for(lock, timeout, [this]()
                                 { return closed_ || !queue_.empty(); });
            if (queue_.empty())
                return false;
            out = std::move(queue_.front());
            queue_.pop_front();
            cvNotFull_.notify_one();
            return true;
        }

        bool tryDequeue(T &out)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty())
                return false;
            out = std::move(queue_.front());
            queue_.pop_front();
            cvNotFull_.notify_one();
            return true;
        }

        void close()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
            cvNotEmpty_.notify_all();
            cvNotFull_.notify_all();
        }

        size_t sizeApprox() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

    private:
        size_t capacity_;
        mutable std::mutex mutex_;
        std::condition_variable cvNotEmpty_;
        std::condition_variable cvNotFull_;
        std::deque<T> queue_;
        bool closed_{false};
    };
} // namespace generator
