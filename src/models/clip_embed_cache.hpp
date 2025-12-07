#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace generator
{

    /**
     * @brief Cache for pre-computed CLIP text embeddings indexed by tag ID.
     *
     * Loads a preprocessed vocabulary and binary CLIP cache, storing embeddings in a
     * contiguous row-major FP16 table for cache-friendly pooling in the hot path.
     */
    class ClipEmbedCache
    {
    public:
        /**
         * @brief Constructs a ClipEmbedCache by loading vocab and embeddings.
         * @param tagVocabPath Optional path to tag vocab JSON (array of {"name": "tag_name"}).
         * @param clipCachePath Path to binary embedding cache file.
         */
        ClipEmbedCache(const std::string &tagVocabPath, const std::string &clipCachePath);

        /**
         * @brief Computes mean-pooled CLIP embedding for sampled tag IDs.
         * @param tagIds Vector of sampled tag IDs from the loaded vocabulary.
         * @return Mean-pooled embedding vector in FP16 of size embedDim.
         */
        std::vector<uint16_t> computePooledEmbedding(const std::vector<uint32_t> &tagIds) const;

        /**
         * @brief Returns number of vocab entries available for random sampling.
         */
        size_t vocabSize() const { return vocab_.size(); }

        /**
         * @brief Returns tag text by ID for output formatting.
         */
        const std::string &getTagById(uint32_t tagId) const { return vocab_[static_cast<size_t>(tagId)]; }

        /**
         * @brief Returns tag ID for an already-normalized vocab tag.
         */
        std::optional<uint32_t> lookupTagId(const std::string &tag) const
        {
            auto it = tagToId_.find(tag);
            if (it == tagToId_.end())
            {
                return std::nullopt;
            }
            return it->second;
        }

        /**
         * @brief Returns the embedding dimension (e.g., 768 for CLIP ViT-L/14).
         */
        uint32_t getEmbedDim() const { return embedDim_; }

        /**
         * @brief Returns row count of the contiguous embedding table.
         */
        size_t embeddingRowCount() const
        {
            return embedDim_ > 0 ? (embeddingsFp16_.size() / static_cast<size_t>(embedDim_)) : 0;
        }

        /**
         * @brief Returns read-only contiguous FP16 embedding table.
         */
        const std::vector<uint16_t> &getEmbeddingTableFp16() const { return embeddingsFp16_; }

        /**
         * @brief Returns the number of embeddings loaded.
         */
        size_t size() const { return vocab_.size(); }

    private:
        /**
         * @brief Preprocesses a raw tag: replace '_' with ' ', escape parentheses.
         * @param tag The raw tag string.
         * @return Preprocessed tag string.
         */
        static std::string preprocessTag(const std::string &tag);

        /**
         * @brief Loads the tag vocabulary from a JSON file.
         * @param tagVocabPath Path to the JSON file.
         */
        void loadVocab(const std::string &tagVocabPath);

        /**
         * @brief Loads embeddings from a binary cache file.
         * @param clipCachePath Path to the binary cache file.
         */
        void loadCache(const std::string &clipCachePath);

        std::vector<std::string> vocab_;        // Preprocessed tag names (ordered)
        std::unordered_map<std::string, uint32_t> tagToId_; // normalized tag -> vocab index
        std::vector<uint16_t> embeddingsFp16_;  // row-major: [tagId * embedDim_ + d]
        uint32_t embedDim_{0};                  // Embedding dimension
    };

} // namespace generator
