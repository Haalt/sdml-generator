#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "tokenizer/tokenizer.hpp"

namespace generator
{

    Tokenizer::Tokenizer(const std::string &tokenizerPath, char delimiter)
        : delimiter_(delimiter)
    {
        std::ifstream file(tokenizerPath);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open tokenizer file: " + tokenizerPath);
        }

        nlohmann::json json;
        file >> json;

        // load word index
        auto wordIndex = json["word_index"];
        for (auto it = wordIndex.begin(); it != wordIndex.end(); ++it)
        {
            int64_t id = it.value().get<int64_t>();
            wordIndex_[it.key()] = id;
            invertedMap_[id] = it.key();
        }

        // load LoRA tokens
        auto loraIndex = json["lora_index"];
        for (auto it = loraIndex.begin(); it != loraIndex.end(); ++it)
        {
            int64_t id = it.value().get<int64_t>();
            loraIndex_[it.key()] = id;
            loraMap_[id] = it.key();
        }

        // load sampler tokens if they exist - legacy check
        if (json.contains("sampler_index"))
        {
            auto samplerIndex = json["sampler_index"];
            for (auto it = samplerIndex.begin(); it != samplerIndex.end(); ++it)
            {
                int64_t id = it.value().get<int64_t>();
                samplerIndex_[it.key()] = id;
                samplerMap_[id] = it.key();
            }
        }

        // Load upscaler tokens if they exist - legacy check
        if (json.contains("upscaler_index"))
        {
            auto upscalerIndex = json["upscaler_index"];
            for (auto it = upscalerIndex.begin(); it != upscalerIndex.end(); ++it)
            {
                int64_t id = it.value().get<int64_t>();
                upscalerIndex_[it.key()] = id;
                upscalerMap_[id] = it.key();
            }
        }
    }

    Prompt Tokenizer::processPrompt(const std::vector<std::string> &prompt, float cfgScale)
    {
        Prompt result;
        // store the raw CFG scale: normalization deferred
        result.cfgScale = std::clamp(cfgScale, 1.0f, MAX_CFG_SCALE);

        result.tokens.reserve(MAX_SEQUENCE_LENGTH);
        result.loras.reserve(MAX_LORAS);
        result.weights.reserve(MAX_LORAS);

        result.numLoras = 0;

        // TODO: refactor
        for (const auto &tag : prompt)
        {
            // fast path: most tag strings have no delimiter, process directly
            if (tag.find(delimiter_) == std::string::npos)
            {
                const std::string &tokenStr = tag;

                if (!tokenStr.empty() && tokenStr[0] == '<')
                {
                    auto [token, weight] = parseLoraTag(tokenStr);
                    auto wordIt = wordIndex_.find(token);
                    if (wordIt != wordIndex_.end() && result.numLoras < MAX_LORAS)
                    {
                        result.weights.push_back(weight);
                        result.loras.push_back(tokenToLoraToken(wordIt->second));
                        ++result.numLoras;
                    }
                }
                else
                {
                    auto it = wordIndex_.find(tokenStr);
                    if (it != wordIndex_.end() && result.tokens.size() < MAX_SEQUENCE_LENGTH)
                    {
                        result.tokens.push_back(it->second);
                    }
                }
                continue; // done with this tag
            }

            // slow path: delimiters present, split into subtokens
            // attempt to retrieve cached split subtokens for this tag
            const std::vector<std::string> *tokensVecPtr = nullptr;
            auto cacheIt = splitCache_.find(tag);

            if (cacheIt != splitCache_.end())
            {
                // Cache hit: reuse the previously split tokens
                tokensVecPtr = &cacheIt->second;
            }
            else
            {
                // cache miss: perform splitting and cache the result
                auto tokensVec = splitPrompt(tag);
                cacheIt = splitCache_.emplace(tag, std::move(tokensVec)).first;
                tokensVecPtr = &cacheIt->second;
            }

            // process each subtoken obtained
            for (const auto &tokenStr : *tokensVecPtr)
            {
                if (!tokenStr.empty() && tokenStr[0] == '<')
                {
                    auto [token, weight] = parseLoraTag(tokenStr);
                    auto wordIt = wordIndex_.find(token);
                    if (wordIt != wordIndex_.end() && result.numLoras < MAX_LORAS)
                    {
                        result.weights.push_back(weight);
                        result.loras.push_back(tokenToLoraToken(wordIt->second));
                        ++result.numLoras;
                    }
                }
                else
                {
                    auto it = wordIndex_.find(tokenStr);
                    if (it != wordIndex_.end() && result.tokens.size() < MAX_SEQUENCE_LENGTH)
                    {
                        result.tokens.push_back(it->second);
                    }
                }
            }
        }

        result.sequenceLength = static_cast<int64_t>(result.tokens.size());

        // pad arrays to fixed size (zero left fill) for fast copy to tensors
        // TODO: measure impact
        result.tokens.resize(MAX_SEQUENCE_LENGTH, 0);
        result.loras.resize(MAX_LORAS, 0);
        result.weights.resize(MAX_LORAS, 0.0f);

        return result;
    }

    std::optional<int64_t> Tokenizer::tokenToIndex(const std::string &token) const
    {
        if (auto it = wordIndex_.find(token); it != wordIndex_.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    std::optional<std::string> Tokenizer::loraTokenToText(int64_t LoraId) const
    {
        if (auto it = loraMap_.find(LoraId); it != loraMap_.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    std::optional<std::string> Tokenizer::tokenToText(int64_t tokenId) const
    {
        if (auto it = invertedMap_.find(tokenId); it != invertedMap_.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

    bool Tokenizer::isLoraToken(const std::string &token) const
    {
        std::string prefix = "lora ";
        return token.size() >= prefix.size() && token.compare(0, prefix.size(), prefix) == 0;
    }

    bool Tokenizer::isLoraToken(int64_t tokenId) const
    {
        return isLoraToken(invertedMap_.at(tokenId));
    }

    std::int64_t Tokenizer::tokenToLoraToken(const std::int64_t token) const
    {
        return loraIndex_.at(invertedMap_.at(token));
    }

    std::vector<std::string> Tokenizer::splitPrompt(const std::string &prompt) const
    {
        std::vector<std::string> tokens;
        std::stringstream ss(prompt);
        std::string token;

        while (std::getline(ss, token, delimiter_))
        {
            // trim whitespaces
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);

            if (!token.empty())
            {
                tokens.push_back(token);
            }
        }

        return tokens;
    }

    std::pair<std::string, float> Tokenizer::parseLoraTag(const std::string &tag) const
    {
        float weight = 1.0f;
        std::string token = tag;

        // check for weight in format "token:weight"
        size_t colonPos = tag.find(':', 6);
        if (colonPos != std::string::npos)
        {
            token = tag.substr(6, colonPos - 6);
            try
            {
                weight = std::stof(tag.substr(colonPos + 1));
            }
            catch (const std::exception &)
            {
                // if parsing fails, keep default weight
            }
        }

        return {"lora " + token, weight};
    }

    std::optional<int64_t> Tokenizer::samplerToToken(const std::string &sampler) const
    {
        if (auto it = samplerIndex_.find(sampler); it != samplerIndex_.end())
            return it->second;
        return std::nullopt;
    }

    std::optional<std::string> Tokenizer::samplerTokenToText(int64_t tokenId) const
    {
        if (auto it = samplerMap_.find(tokenId); it != samplerMap_.end())
            return it->second;
        return std::nullopt;
    }

    std::optional<int64_t> Tokenizer::upscalerToToken(const std::string &upscaler) const
    {
        if (auto it = upscalerIndex_.find(upscaler); it != upscalerIndex_.end())
            return it->second;
        return std::nullopt;
    }

    std::optional<std::string> Tokenizer::upscalerTokenToText(int64_t tokenId) const
    {
        if (auto it = upscalerMap_.find(tokenId); it != upscalerMap_.end())
            return it->second;
        return std::nullopt;
    }

    std::vector<int64_t> Tokenizer::precomputeTagTokens(const std::string &tag) const
    {
        std::vector<int64_t> result;

        // fast path: most tags have no delimiter: process directly
        if (tag.find(delimiter_) == std::string::npos)
        {
            auto it = wordIndex_.find(tag);
            if (it != wordIndex_.end())
            {
                result.push_back(it->second);
            }
            return result;
        }

        // slow path: delimiters present, split into subtokens
        auto tokens = splitPrompt(tag);
        result.reserve(tokens.size());

        for (const auto &tokenStr : tokens)
        {
            auto it = wordIndex_.find(tokenStr);
            if (it != wordIndex_.end())
            {
                result.push_back(it->second);
            }
        }

        return result;
    }

    std::optional<std::pair<int64_t, float>> Tokenizer::precomputeLoraToken(const std::string &loraTag) const
    {
        if (loraTag.empty() || loraTag[0] != '<')
            return std::nullopt;

        auto [token, weight] = parseLoraTag(loraTag);
        auto wordIt = wordIndex_.find(token);
        if (wordIt != wordIndex_.end())
        {
            int64_t loraTokenId = tokenToLoraToken(wordIt->second);
            return std::make_pair(loraTokenId, weight);
        }

        return std::nullopt;
    }

    Prompt Tokenizer::assemblePrompt(const std::vector<std::vector<int64_t>> &tokenVectors,
                                     const std::vector<std::pair<int64_t, float>> &loraTokens,
                                     float cfgScale) const
    {
        Prompt result;
        result.cfgScale = std::clamp(cfgScale, 1.0f, MAX_CFG_SCALE);

        result.tokens.reserve(MAX_SEQUENCE_LENGTH);
        result.loras.reserve(MAX_LORAS);
        result.weights.reserve(MAX_LORAS);

        // Assemble regular tokens
        for (const auto &tokenVec : tokenVectors)
        {
            if (result.tokens.size() >= MAX_SEQUENCE_LENGTH)
                break;

            size_t remaining = MAX_SEQUENCE_LENGTH - result.tokens.size();
            if (tokenVec.size() <= remaining)
            {
                result.tokens.insert(result.tokens.end(), tokenVec.begin(), tokenVec.end());
            }
            else
            {
                result.tokens.insert(result.tokens.end(), tokenVec.begin(), tokenVec.begin() + remaining);
                break;
            }
        }

        // assemble LoRA tokens
        result.numLoras = 0;
        for (const auto &[loraToken, weight] : loraTokens)
        {
            if (result.numLoras < MAX_LORAS)
            {
                result.loras.push_back(loraToken);
                result.weights.push_back(weight);
                ++result.numLoras;
            }
        }

        result.sequenceLength = static_cast<int64_t>(result.tokens.size());

        // pad arrays to fixed size (zero left fill) for fast copy to tensors
        result.tokens.resize(MAX_SEQUENCE_LENGTH, 0);
        result.loras.resize(MAX_LORAS, 0);
        result.weights.resize(MAX_LORAS, 0.0f);

        return result;
    }

} // namespace generator