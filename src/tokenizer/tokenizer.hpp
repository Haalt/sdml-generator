#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/types.hpp"

namespace generator
{

    /**
     * @brief Main tokenizer class for processing prompts and managing tokens
     *
     * This class handles tokenization of prompts, including special handling for LoRA tokens
     * and weight parsing. It provides a unified interface for both the generator and
     * evaluation components
     */
    class Tokenizer
    {
    public:
        static std::vector<std::string> splitTagString(const std::string &tag, char delimiter = ',');

        /**
         * @brief Constructs a new Tokenizer instance
         * @param tokenizerPath Path to the tokenizer configuration file
         * @param delimiter Character used to split tokens in prompts
         * @param stripParentheses Whether to strip parentheses from tags before tokenization
         */
        explicit Tokenizer(const std::string &tokenizerPath, char delimiter = ',', bool stripParentheses = true);

        /**
         * @brief Processes a prompt string into tokens
         * @param prompt The input prompt string
         * @param cfgScale The CFG scale value
         * @return Prompt containing the tokenized result
         */
        Prompt processPrompt(const std::vector<std::string> &prompt, float cfgScale);

        /**
         * @brief Converts a token string to its numerical index
         * @param token The token string to convert
         * @return Optional containing the token ID if found
         */
        std::optional<int64_t> tokenToIndex(const std::string &token) const;

        /**
         * @brief Converts a token ID back to its string representation
         * @param tokenId The token ID to convert
         * @return Optional containing the token string if found
         */
        std::optional<std::string> tokenToText(int64_t tokenId) const;

        /**
         * @brief Checks if a token is a LoRA token
         * @param token The token to check
         * @return true if the token is a LoRA token
         */
        bool isLoraToken(const std::string &token) const;

        /**
         * @brief Checks if a token ID represents a LoRA token
         * @param tokenId The token ID to check
         * @return true if the token ID represents a LoRA token
         */
        bool isLoraToken(int64_t tokenId) const;

        std::optional<std::string> loraTokenToText(int64_t LoraId) const;

        /**
         * @brief Converts a sampler string to its token ID
         * @param sampler The sampler name to convert
         * @return Optional containing the sampler token ID if found
         */
        std::optional<int64_t> samplerToToken(const std::string &sampler) const;

        /**
         * @brief Converts a sampler token ID back to its string representation
         * @param tokenId The sampler token ID to convert
         * @return Optional containing the sampler string if found
         */
        std::optional<std::string> samplerTokenToText(int64_t tokenId) const;

        /**
         * @brief Converts an upscaler string to its token ID
         * @param upscaler The upscaler name to convert
         * @return Optional containing the upscaler token ID if found
         */
        std::optional<int64_t> upscalerToToken(const std::string &upscaler) const;

        /**
         * @brief Converts an upscaler token ID back to its string representation
         * @param tokenId The upscaler token ID to convert
         * @return Optional containing the upscaler string if found
         */
        std::optional<std::string> upscalerTokenToText(int64_t tokenId) const;

        /**
         * @brief Gets the size of the sampler vocabulary
         * @return Number of samplers in the vocabulary
         */
        size_t getSamplerVocabSize() const { return samplerIndex_.size(); }

        /**
         * @brief Gets the size of the upscaler vocabulary
         * @return Number of upscalers in the vocabulary
         */
        size_t getUpscalerVocabSize() const { return upscalerIndex_.size(); }

        /**
         * @brief Precomputes tokens for a single tag string
         * @param tag The tag string to tokenize
         * @return Vector of token IDs (most will be length 1, some may be split by delimiter)
         */
        std::vector<int64_t> precomputeTagTokens(const std::string &tag) const;

        /**
         * @brief Precomputes LoRA token and weight for a LoRA tag string
         * @param loraTag The LoRA tag string (e.g., "<lora:name:1.2>")
         * @return Pair of LoRA token ID and weight, or nullopt if invalid
         */
        std::optional<std::pair<int64_t, float>> precomputeLoraToken(const std::string &loraTag) const;

        /**
         * @brief Assembles a prompt from precomputed token vectors
         * @param tokenVectors Vector of token vectors (one per tag)
         * @param loraTokens Vector of LoRA token/weight pairs
         * @param cfgScale The CFG scale value
         * @return Assembled Prompt structure
         */
        Prompt assemblePrompt(const std::vector<std::vector<int64_t>> &tokenVectors,
                              const std::vector<std::pair<int64_t, float>> &loraTokens,
                              float cfgScale) const;
        Prompt assemblePrompt(const std::vector<const std::vector<int64_t> *> &tokenVectorRefs,
                              const std::vector<std::pair<int64_t, float>> &loraTokens,
                              float cfgScale) const;

    private:
        char delimiter_;
        bool stripParentheses_;
        std::unordered_map<std::string, int64_t> wordIndex_;
        std::unordered_map<std::string, int64_t> loraIndex_;
        std::unordered_map<std::string, int64_t> samplerIndex_;
        std::unordered_map<std::string, int64_t> upscalerIndex_;
        std::unordered_map<int64_t, std::string> invertedMap_;
        std::unordered_map<int64_t, std::string> loraMap_;
        std::unordered_map<int64_t, std::string> samplerMap_;
        std::unordered_map<int64_t, std::string> upscalerMap_;

        // cache mapping raw tag strings (containing delimiters) to their pre computed vector of split subtokens
        // This avoids repeated calls to splitPrompt for frequently reused prompt fragments
        std::unordered_map<std::string, std::vector<std::string>> splitCache_;

        /**
         * @brief Splits a prompt string into individual tokens
         * @param prompt The prompt string to split
         * @return Vector of token strings
         */
        std::vector<std::string> splitPrompt(const std::string &prompt) const;

        /**
         * @brief Parses a token string to extract the token and its weight
         * @param tag The token string to parse
         * @return Pair containing the token string and its weight
         */
        std::pair<std::string, float> parseLoraTag(const std::string &tag) const;

        std::int64_t tokenToLoraToken(const std::int64_t token) const;
    };

} // namespace generator