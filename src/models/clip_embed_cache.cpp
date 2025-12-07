#include "models/clip_embed_cache.hpp"

#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace generator
{
    namespace
    {
        inline uint16_t float_to_half(float value)
        {
            __m128 f = _mm_set_ss(value);
            __m128i h = _mm_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
        }
        inline float half_to_float_local(uint16_t h)
        {
            const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
            uint32_t exp = static_cast<uint32_t>(h & 0x7C00u) >> 10;
            uint32_t mant = static_cast<uint32_t>(h & 0x03FFu);
            if (exp == 0)
            {
                if (mant == 0)
                {
                    uint32_t val = sign;
                    float out;
                    std::memcpy(&out, &val, sizeof(out));
                    return out;
                }
                while ((mant & 0x0400u) == 0)
                {
                    mant <<= 1;
                    exp -= 1;
                }
                exp += 1;
                mant &= ~0x0400u;
            }
            else if (exp == 31)
            {
                uint32_t val = sign | 0x7F800000u | (mant << 13);
                float out;
                std::memcpy(&out, &val, sizeof(out));
                return out;
            }
            exp = exp + (127 - 15);
            uint32_t val = sign | (exp << 23) | (mant << 13);
            float out;
            std::memcpy(&out, &val, sizeof(out));
            return out;
        }
    } // namespace

    ClipEmbedCache::ClipEmbedCache(const std::string &tagVocabPath, const std::string &clipCachePath)
    {
        if (!tagVocabPath.empty())
        {
            loadVocab(tagVocabPath);
        }
        loadCache(clipCachePath);
        std::cout << "ClipEmbedCache: loaded " << vocab_.size() << " embeddings (dim=" << embedDim_
                  << ") from " << clipCachePath << ", vocab size=" << vocab_.size() << std::endl;
    }

    std::string ClipEmbedCache::preprocessTag(const std::string &tag)
    {
        std::string result;
        result.reserve(tag.size() + 4);

        for (size_t i = 0; i < tag.size(); ++i)
        {
            char ch = tag[i];

            if (ch == '_')
            {
                result.push_back(' ');
            }
            else if (ch == '(')
            {
                // If already escaped like "\(", keep it as-is.
                if (i > 0 && tag[i - 1] == '\\')
                {
                    result.push_back('(');
                }
                else
                {
                    result.push_back('\\');
                    result.push_back('(');
                }
            }
            else if (ch == ')')
            {
                // If followed by '\' don't add another escape.
                if (tag[i - 1] == '\\')
                {
                    result.push_back(')');
                }
                else
                {
                    result.push_back('\\');
                    result.push_back(')');
                }
            }
            else
            {
                result.push_back(ch);
            }
        }

        return result;
    }

    void ClipEmbedCache::loadVocab(const std::string &tagVocabPath)
    {
        std::ifstream file(tagVocabPath);
        if (!file.is_open())
        {
            throw std::runtime_error("ClipEmbedCache: failed to open tag vocab file: " + tagVocabPath);
        }
        nlohmann::json json;
        try
        {
            file >> json;
        }
        catch (const nlohmann::json::parse_error &e)
        {
            throw std::runtime_error("ClipEmbedCache: failed to parse tag vocab JSON: " + std::string(e.what()));
        }
        if (!json.is_array())
        {
            throw std::runtime_error("ClipEmbedCache: tag vocab JSON must be an array");
        }
        vocab_.reserve(json.size());
        tagToId_.reserve(json.size());
        for (const auto &entry : json)
        {
            if (entry.contains("name") && entry["name"].is_string())
            {
                std::string raw = entry["name"].get<std::string>();
                std::string normalized = preprocessTag(raw);
                vocab_.push_back(normalized);
                const uint32_t id = static_cast<uint32_t>(vocab_.size() - 1);
                tagToId_.emplace(std::move(normalized), id);
            }
        }
        std::cout << "ClipEmbedCache: loaded " << vocab_.size() << " tags from vocab" << std::endl;
    }

    void ClipEmbedCache::loadCache(const std::string &clipCachePath)
    {
        std::ifstream file(clipCachePath, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("ClipEmbedCache: failed to open binary cache file: " + clipCachePath);
        }
        uint32_t numEntries = 0;
        uint32_t embedDim = 0;
        file.read(reinterpret_cast<char *>(&numEntries), sizeof(uint32_t));
        file.read(reinterpret_cast<char *>(&embedDim), sizeof(uint32_t));
        if (!file)
        {
            throw std::runtime_error("ClipEmbedCache: failed to read binary cache header");
        }
        embedDim_ = embedDim;
        embeddingsFp16_.assign(static_cast<size_t>(numEntries) * static_cast<size_t>(embedDim_), 0);
        if (vocab_.empty())
        {
            vocab_.reserve(numEntries);
            tagToId_.reserve(numEntries);
        }
        for (uint32_t i = 0; i < numEntries; ++i)
        {
            uint32_t nameLen = 0;
            file.read(reinterpret_cast<char *>(&nameLen), sizeof(uint32_t));
            if (!file)
            {
                throw std::runtime_error("ClipEmbedCache: truncated binary cache at entry " + std::to_string(i));
            }
            std::string tagName(nameLen, '\0');
            file.read(tagName.data(), nameLen);
            if (!file)
            {
                throw std::runtime_error("ClipEmbedCache: truncated tag name at entry " + std::to_string(i));
            }
            if (vocab_.empty() || i >= vocab_.size())
            {
                std::string normalizedTag = preprocessTag(tagName);
                const uint32_t id = static_cast<uint32_t>(vocab_.size());
                vocab_.push_back(normalizedTag);
                tagToId_.emplace(std::move(normalizedTag), id);
            }
            std::vector<float> embedding(embedDim);
            file.read(reinterpret_cast<char *>(embedding.data()), embedDim * sizeof(float));
            if (!file)
            {
                throw std::runtime_error("ClipEmbedCache: truncated embedding at entry " + std::to_string(i));
            }
            const size_t rowBase = static_cast<size_t>(i) * static_cast<size_t>(embedDim_);
            for (uint32_t d = 0; d < embedDim_; ++d)
            {
                embeddingsFp16_[rowBase + d] = float_to_half(embedding[d]);
            }
        }
    }

    std::vector<uint16_t> ClipEmbedCache::computePooledEmbedding(const std::vector<uint32_t> &tagIds) const
    {
        std::vector<uint16_t> pooledFp16(embedDim_, 0);
        if (tagIds.empty() || embedDim_ == 0)
        {
            return pooledFp16;
        }
        std::vector<float> pooled(embedDim_, 0.0f);
        const size_t embeddingRows = embedDim_ > 0 ? (embeddingsFp16_.size() / static_cast<size_t>(embedDim_)) : 0;
        int matchCount = 0;
        for (uint32_t tagId : tagIds)
        {
            const size_t tagIdx = static_cast<size_t>(tagId);
            if (tagIdx >= vocab_.size() || tagIdx >= embeddingRows)
            {
                continue;
            }
            const size_t rowBase = tagIdx * static_cast<size_t>(embedDim_);
            for (uint32_t d = 0; d < embedDim_; ++d)
            {
                pooled[d] += half_to_float_local(embeddingsFp16_[rowBase + d]);
            }
            ++matchCount;
        }
        if (matchCount > 0)
        {
            const float invCount = 1.0f / static_cast<float>(matchCount);
            for (uint32_t d = 0; d < embedDim_; ++d)
            {
                pooledFp16[d] = float_to_half(pooled[d] * invCount);
            }
        }
        return pooledFp16;
    }

} // namespace generator
