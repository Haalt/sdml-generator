#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "models/tag_dict.h"
#include "utils/fast_rand.hpp"

// FD
namespace generator
{
    class Tokenizer;
}

#include "tokenizer/tokenizer.hpp"

#define VARIATION_THRESHOLD 50

class Lora
{
public:
    enum LORA_TYPE
    {
        CONDITIONAL,
        MAIN,
        SECONDARY,
        TAG
    };

protected:
    using func_ptr = std::function<std::vector<std::string>()>;

    const std::vector<std::string> const_tags_;
    const std::vector<std::string> const_variations_;
    const std::vector<std::string> const_banned_key_tags_;
    std::vector<std::string> tags_;
    std::vector<std::string> variations_;
    std::vector<std::string> banned_key_tags_;
    std::string prefix_tag_;
    std::string lora_tag_;
    float min_weight_;
    float max_weight_;
    LORA_TYPE type_;

    // precomputed token data
    std::vector<std::vector<int64_t>> precomputed_tags_;
    std::vector<std::vector<int64_t>> precomputed_variations_;
    std::optional<std::pair<int64_t, float>> precomputed_lora_token_;
    std::unordered_set<std::string> cached_banned_union_; // precomputed union of banned tags for fast filtering

    func_ptr reroll_tags = nullptr;
    func_ptr reroll_variations = nullptr;

    Lora(std::vector<std::string> tags, std::vector<std::string> variations,
         std::string lora_tag, std::vector<std::string> banned_key_tags,
         float min_weight = 1.f, float max_weight = 1.f,
         std::string prefix_tag = "lora")
        : const_tags_(tags), const_variations_(variations),
          const_banned_key_tags_(banned_key_tags), tags_(tags),
          variations_(variations), banned_key_tags_(banned_key_tags),
          prefix_tag_(prefix_tag), lora_tag_(lora_tag),
          min_weight_(min_weight), max_weight_(max_weight), type_() {}

public:
    float get_weight(void) const
    {
        float min = .2f, max = 1.2f;

        // TODO: re enable weight bounds
        // return std::lerp(min, max, rng::urand());
        float r = rng::urand();
        return min * (1.0 - r) + (max * r);
    }

    virtual ~Lora(void) {}

    // initialize precomputed tokens (should be called after construction)
    void precompute_tokens(
        const generator::Tokenizer *tokenizer,
        std::unordered_map<std::string, std::vector<int64_t>> *tagTokenCache = nullptr,
        std::unordered_map<std::string, std::vector<int64_t>> *variationTokenCache = nullptr)
    {
        if (!tokenizer)
            return;

        // precompute regular tags
        precomputed_tags_.clear();
        precomputed_tags_.reserve(tags_.size());
        for (const auto &tag : tags_)
        {
            if (tagTokenCache != nullptr)
            {
                auto cached = tagTokenCache->find(tag);
                if (cached != tagTokenCache->end())
                {
                    precomputed_tags_.push_back(cached->second);
                }
                else
                {
                    std::vector<int64_t> computed = tokenizer->precomputeTagTokens(tag);
                    tagTokenCache->emplace(tag, computed);
                    precomputed_tags_.push_back(std::move(computed));
                }
            }
            else
            {
                precomputed_tags_.push_back(tokenizer->precomputeTagTokens(tag));
            }
        }

        // precompute variations
        precomputed_variations_.clear();
        precomputed_variations_.reserve(variations_.size());
        for (const auto &variation : variations_)
        {
            if (variationTokenCache != nullptr)
            {
                auto cached = variationTokenCache->find(variation);
                if (cached != variationTokenCache->end())
                {
                    precomputed_variations_.push_back(cached->second);
                }
                else
                {
                    std::vector<int64_t> computed = tokenizer->precomputeTagTokens(variation);
                    variationTokenCache->emplace(variation, computed);
                    precomputed_variations_.push_back(std::move(computed));
                }
            }
            else
            {
                precomputed_variations_.push_back(tokenizer->precomputeTagTokens(variation));
            }
        }

        std::string lora_tag_str = "<" + prefix_tag_ + ":" + lora_tag_ + ":" + std::to_string(get_weight()) + ">";
        precomputed_lora_token_ = tokenizer->precomputeLoraToken(lora_tag_str);
    }

    void set_reroll_tags(func_ptr func) { reroll_tags = func; }
    void set_reroll_variations(func_ptr func) { reroll_variations = func; }

    void reroll(void)
    {
        banned_key_tags_ = const_banned_key_tags_; // may only be usefull on ConditionalVariantLora

        if (reroll_tags != nullptr)
        {
            tags_ = const_tags_;
            std::vector<std::string> extra = reroll_tags();
            tags_.insert(tags_.end(), extra.begin(), extra.end());
        }
        if (reroll_variations != nullptr)
        {
            variations_ = const_variations_;
            std::vector<std::string> extra = reroll_variations();
            variations_.insert(variations_.end(), extra.begin(), extra.end());
        }

        // rebuild cached banned union once after reroll
        cached_banned_union_.clear();
        for (const auto &key : banned_key_tags_)
        {
            const std::shared_ptr<const Tag_dict> tg = Tag_dict::getInstance();
            const std::vector<std::string> vec = tg->getListNode(key);
            cached_banned_union_.insert(vec.begin(), vec.end());
        }
    }

    std::string get_lora_tag(void) const { return lora_tag_; }
    Lora::LORA_TYPE get_type(void) const { return type_; }
    std::string get_prefix_tag(void) const { return prefix_tag_; }
    std::optional<std::pair<int64_t, float>> get_precomputed_lora_token(void) const { return precomputed_lora_token_; }

    virtual std::vector<std::string>
    remove_banned_tags(std::vector<std::string> tags)
    {
        if (cached_banned_union_.empty() && !banned_key_tags_.empty())
        {
            for (const auto &key : banned_key_tags_)
            {
                const std::shared_ptr<const Tag_dict> tg = Tag_dict::getInstance();
                const std::vector<std::string> vec = tg->getListNode(key);
                cached_banned_union_.insert(vec.begin(), vec.end());
            }
        }

        tags.erase(std::remove_if(tags.begin(), tags.end(),
                                  [&](const std::string &t)
                                  {
                                      return cached_banned_union_.find(t) != cached_banned_union_.end();
                                  }),
                   tags.end());

        return tags;
    }

    virtual std::vector<std::string> get_tags(void)
    {
        std::vector<std::string> tags;

        tags.push_back("<" + prefix_tag_ + ":" + lora_tag_ + ":" +
                       std::to_string(get_weight()) + ">");
        tags.insert(tags.end(), tags_.begin(), tags_.end());

        for (const auto &t : variations_)
        {
            if (rng::chance(VARIATION_THRESHOLD))
                tags.push_back(t);
        }

        return tags;
    }

    // get precomputed tokens and LoRA info
    virtual void append_precomputed_tokens(std::vector<const std::vector<int64_t> *> &token_vector_refs,
                                           std::vector<std::pair<int64_t, float>> &lora_tokens) const
    {
        if (precomputed_lora_token_)
        {
            lora_tokens.push_back(*precomputed_lora_token_);
        }

        for (const auto &tokenVec : precomputed_tags_)
        {
            token_vector_refs.push_back(&tokenVec);
        }

        for (size_t i = 0; i < precomputed_variations_.size(); ++i)
        {
            if (rng::chance(VARIATION_THRESHOLD))
            {
                token_vector_refs.push_back(&precomputed_variations_[i]);
            }
        }
    }

    virtual std::pair<std::vector<std::vector<int64_t>>, std::vector<std::pair<int64_t, float>>> get_precomputed_tokens(void) const
    {
        std::vector<std::vector<int64_t>> token_vectors;
        std::vector<std::pair<int64_t, float>> lora_tokens;
        std::vector<const std::vector<int64_t> *> token_refs;
        token_refs.reserve(precomputed_tags_.size() + precomputed_variations_.size());
        append_precomputed_tokens(token_refs, lora_tokens);
        token_vectors.reserve(token_refs.size());
        for (const auto *tokenVecPtr : token_refs)
        {
            if (tokenVecPtr != nullptr)
            {
                token_vectors.push_back(*tokenVecPtr);
            }
        }

        return {token_vectors, lora_tokens};
    }
};

class ConditionalVariantLora : public Lora
{
    bool force_ban_ = false;
    std::vector<std::string> extra_ban_categories_;

public:
    ConditionalVariantLora(std::vector<std::string> tags, std::vector<std::string> variations,
                           std::string lora_tag, std::vector<std::string> banned_key_tags,
                           float min_weight = 1.f, float max_weight = 1.f,
                           std::string prefix_tag = "lora")
        : Lora(tags, variations, lora_tag, banned_key_tags, min_weight,
               max_weight, prefix_tag)
    {
        force_ban_ = variations.size() > 0 && rng::chance(VARIATION_THRESHOLD);
        type_ = Lora::CONDITIONAL;
    }

    virtual std::vector<std::string> get_tags(void) override
    {
        std::vector<std::string> tags;

        force_ban_ = rng::chance(VARIATION_THRESHOLD) && variations_.size() > 0;

        tags.push_back("<" + prefix_tag_ + ":" + lora_tag_ + ":" +
                       std::to_string(get_weight()) + ">");
        tags.insert(tags.end(), tags_.begin(), tags_.end());

        if (!force_ban_)
            return tags;

        tags.push_back(variations_[rng::irange(0, variations_.size() - 1)]);
        return tags;
    }

    virtual void append_precomputed_tokens(std::vector<const std::vector<int64_t> *> &token_vector_refs,
                                           std::vector<std::pair<int64_t, float>> &lora_tokens) const override
    {
        bool force_ban = rng::chance(VARIATION_THRESHOLD) && precomputed_variations_.size() > 0;

        if (precomputed_lora_token_)
        {
            lora_tokens.push_back(*precomputed_lora_token_);
        }

        for (const auto &tokenVec : precomputed_tags_)
        {
            token_vector_refs.push_back(&tokenVec);
        }

        if (!force_ban)
            return;

        size_t variation_idx = rng::irange(0, precomputed_variations_.size() - 1);
        token_vector_refs.push_back(&precomputed_variations_[variation_idx]);
    }

    void register_extra_banned(std::vector<std::string> categories)
    {
        extra_ban_categories_.insert(extra_ban_categories_.end(), categories.begin(), categories.end());
    }

    virtual std::vector<std::string> remove_banned_tags(std::vector<std::string> tags) override
    {
        if (force_ban_)
        {
            banned_key_tags_.insert(banned_key_tags_.end(), extra_ban_categories_.begin(), extra_ban_categories_.end());
        }
        return Lora::remove_banned_tags(tags);
    }
};

class MainLora : public Lora
{
public:
    MainLora(std::vector<std::string> tags, std::vector<std::string> variations,
             std::string lora_tag, std::vector<std::string> banned_key_tags,
             float min_weight = 1.f, float max_weight = 1.f,
             std::string prefix_tag = "lora")
        : Lora(tags, variations, lora_tag, banned_key_tags, min_weight,
               max_weight, prefix_tag)
    {
        type_ = Lora::MAIN;
    }
};

class SecondaryLora : public Lora
{
public:
    SecondaryLora(std::vector<std::string> tags, std::vector<std::string> variations,
                  std::string lora_tag, std::vector<std::string> banned_key_tags,
                  float min_weight = 1.f, float max_weight = 1.f,
                  std::string prefix_tag = "lora")
        : Lora(tags, variations, lora_tag, banned_key_tags, min_weight,
               max_weight, prefix_tag)
    {
        type_ = Lora::SECONDARY;
    }
};

class TagLora : public Lora
{
public:
    TagLora(std::vector<std::string> tags, std::vector<std::string> variations,
            std::string lora_tag, std::vector<std::string> banned_key_tags,
            float min_weight = 1.f, float max_weight = 1.f,
            std::string prefix_tag = "lora")
        : Lora(tags, variations, lora_tag, banned_key_tags, min_weight,
               max_weight, prefix_tag)
    {
        type_ = Lora::TAG;
    }
};
