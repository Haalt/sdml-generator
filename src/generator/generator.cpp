#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "generator/generator.h"
#include "models/Lora.h"
#include "models/Tag_list.h"
#include "random/pcg_random.hpp"
#include "tokenizer/tokenizer.hpp"

namespace generator
{
    // TODO: move somewhere else
    constexpr int LOWER_MAX_LORA = 2;
    constexpr int UPPER_MAX_LORA = 7; // exclusive

    std::unique_ptr<Prompt> Generator::generatePrompt() const
    {
        // Initialize high-performance PCG random generator
        static thread_local pcg32 rng(std::random_device{}());
        static thread_local uint32_t counter = 0;

        auto prompt = std::make_unique<Prompt>();

        static const bool useCudaClipPooling = []()
        {
            const char *env = std::getenv("SDML_CLIP_POOLING_CUDA");
            return env != nullptr && std::string(env) == "1";
        }();
        static const bool hostFallback = []()
        {
            const char *env = std::getenv("SDML_CLIP_POOLING_HOST_FALLBACK");
            return env != nullptr && std::string(env) == "1";
        }();
        const bool clipScoringMode = (modelSettings_.modelType == "clip_embed" && clipEmbedCache_ != nullptr);
        const bool clipRandomVocabMode = (modelSettings_.generationMode == "clip_random_vocab");

        // ======= clip_embed + clip_random_vocab path =======
        if (clipScoringMode && clipRandomVocabMode)
        {
            const size_t vocabSize = clipEmbedCache_->vocabSize();
            if (vocabSize > 0)
            {
                const int numTags = static_cast<int>(rng(51)) + 5; // 5-55 tags
                std::vector<uint32_t> selectedTagIds;
                selectedTagIds.reserve(static_cast<size_t>(numTags));
                for (int i = 0; i < numTags; ++i)
                {
                    selectedTagIds.push_back(static_cast<uint32_t>(rng(vocabSize)));
                }
                prompt->clipTagIds = std::move(selectedTagIds);
                prompt->clipDim = clipEmbedCache_->getEmbedDim();
                if (!useCudaClipPooling || hostFallback)
                {
                    prompt->clipEmb = clipEmbedCache_->computePooledEmbedding(prompt->clipTagIds);
                }
            }
            // Set empty tokens/loras for clip_embed (model doesn't use them)
            prompt->tokens.clear();
            prompt->loras.assign(MAX_LORAS, 0);
            prompt->weights.assign(MAX_LORAS, 0.0f);
            prompt->cfgScale = cfg_scales_[rng(cfg_scales_.size())];
            prompt->sequenceLength = 0;
            prompt->numLoras = 0;
            prompt->modelId = modelSettings_.modelId;
            prompt->profileId = modelSettings_.profileId;
        }
        // ======= expert_dict path: existing tag/LoRA generation =======
        else
        {
            Lora *selected_main = nullptr;
            Lora *selected_conditional = nullptr;
            std::unordered_set<Lora *> selected_loras;
            // Only process LoRAs if the model uses them
            if (modelSettings_.useLoras)
            {
                if (counter % 50000 == 0)
                {
                    lora_controller_.reroll_loras();
                    lora_controller_.precompute_all_tokens(&tokenizer_);
                }
                selected_loras.reserve(50);
                int probability = rng(71) + 30;                                        // 30-100%
                int max_loras = rng(UPPER_MAX_LORA - LOWER_MAX_LORA) + LOWER_MAX_LORA; // 2-6
                for (int i = 0; i < max_loras; ++i)
                {
                    if (static_cast<int>(rng(100)) < probability)
                    {
                        break;
                    }
                    std::vector<Lora *> loras = lora_controller_.get_loras(selected_main, selected_loras, selected_conditional);
                    if (loras.empty())
                    {
                        break;
                    }
                    Lora *selected = loras[rng(loras.size())];
                    if (selected == nullptr || selected_loras.count(selected))
                    {
                        continue;
                    }
                    if (selected->get_type() == Lora::MAIN)
                        selected_main = selected;
                    if (selected->get_type() == Lora::CONDITIONAL)
                        selected_conditional = selected;
                    else
                        selected_loras.insert(selected);
                }
            }
            tag_list_.reset();
            tag_list_.generate_tags(rng(50) + 5); // 10-30 tags
            // Apply banned tag filtering from selected LoRAs (only if using LoRAs)
            if (modelSettings_.useLoras)
            {
                for (Lora *l : selected_loras)
                {
                    tag_list_.set_tags(l->remove_banned_tags(tag_list_.get_tags()));
                }
                if (selected_conditional != nullptr)
                {
                    tag_list_.set_tags(selected_conditional->remove_banned_tags(tag_list_.get_tags()));
                }
            }
            // collect all token vectors and LoRA tokens
            std::vector<const std::vector<int64_t> *> all_token_vectors;
            std::vector<std::pair<int64_t, float>> all_lora_tokens;
            std::vector<std::vector<int64_t>> dynamic_token_storage;
            // add base tag tokens
            const auto &base_tokens = tag_list_.get_precomputed_tokens();
            all_token_vectors.reserve(base_tokens.size() + 32);
            for (const auto &tokenVec : base_tokens)
            {
                all_token_vectors.push_back(&tokenVec);
            }
            // add LoRA tokens from selected LoRAs (only if using LoRAs)
            if (modelSettings_.useLoras)
            {
                for (const Lora *lora : selected_loras)
                {
                    lora->append_precomputed_tokens(all_token_vectors, all_lora_tokens);
                }
                // add conditional LoRA tokens
                if (selected_conditional != nullptr)
                {
                    selected_conditional->append_precomputed_tokens(all_token_vectors, all_lora_tokens);
                }
            }
            // check for tag LoRAs and add their tokens (only if using LoRAs)
            std::vector<std::string> tags = tag_list_.get_tags();
            bool is_vanilla = !modelSettings_.useLoras || (selected_loras.empty() && selected_conditional == nullptr);
            if (modelSettings_.useLoras)
            {
                const std::unordered_map<std::string, TagLora *> &tag_loramap = lora_controller_.get_tag_lora_map();
                for (const std::string &tag : tags)
                {
                    auto res = tag_loramap.find(tag);
                    if (res != tag_loramap.end())
                    {
                        is_vanilla = false;
                        const Lora *lora = static_cast<const Lora *>(res->second);
                        lora->append_precomputed_tokens(all_token_vectors, all_lora_tokens);
                    }
                }
            }
            // add configured extra tag for vanilla prompts (if any)
            const std::vector<uint32_t> *selectedExtraClipIds = nullptr;
            if (is_vanilla && rng(2) && !vanilla_extra_tags_.empty())
            {
                const size_t extraTagIndex = rng(vanilla_extra_tags_.size());
                const std::string &extraTag = vanilla_extra_tags_[extraTagIndex];
                auto extra_tokens = tokenizer_.precomputeTagTokens(extraTag);
                if (!extra_tokens.empty())
                {
                    dynamic_token_storage.push_back(std::move(extra_tokens));
                    all_token_vectors.push_back(&dynamic_token_storage.back());
                }
                if (extraTagIndex < vanilla_extra_clip_tag_ids_.size())
                {
                    selectedExtraClipIds = &vanilla_extra_clip_tag_ids_[extraTagIndex];
                }
            }
            *prompt = tokenizer_.assemblePrompt(all_token_vectors, all_lora_tokens, cfg_scales_[rng(cfg_scales_.size())]);
            if (clipScoringMode)
            {
                const auto &precomputedClipTagIds = tag_list_.get_precomputed_clip_tag_ids();
                size_t totalClipTags = 0;
                for (const auto &tagClipIds : precomputedClipTagIds)
                {
                    totalClipTags += tagClipIds.size();
                }
                if (selectedExtraClipIds != nullptr)
                {
                    totalClipTags += selectedExtraClipIds->size();
                }
                prompt->clipTagIds.clear();
                prompt->clipTagIds.reserve(totalClipTags);
                for (const auto &tagClipIds : precomputedClipTagIds)
                {
                    prompt->clipTagIds.insert(prompt->clipTagIds.end(), tagClipIds.begin(), tagClipIds.end());
                }
                if (selectedExtraClipIds != nullptr)
                {
                    prompt->clipTagIds.insert(
                        prompt->clipTagIds.end(),
                        selectedExtraClipIds->begin(),
                        selectedExtraClipIds->end());
                }
                prompt->clipDim = clipEmbedCache_->getEmbedDim();
                if (!useCudaClipPooling || hostFallback)
                {
                    prompt->clipEmb = clipEmbedCache_->computePooledEmbedding(prompt->clipTagIds);
                }
            }
            // Set model ID and profile ID from settings
            prompt->modelId = modelSettings_.modelId;
            prompt->profileId = modelSettings_.profileId;
        }

        // ======= Common: sampler, steps, upscaler, denoise =======

        // Sampler selection
        if (!samplers_.empty())
        {
            const std::string &selectedSampler = samplers_[rng(samplers_.size())];
            auto samplerToken = tokenizer_.samplerToToken(selectedSampler);
            prompt->sampler = samplerToken.value_or(0);
        }
        else
        {
            prompt->sampler = 0;
        }

        // Steps: use model-specific range from settings
        int stepRange = modelSettings_.maxSteps - modelSettings_.minSteps;
        if (stepRange <= 0)
            stepRange = 1;
        prompt->steps = rng(stepRange + 1) + modelSettings_.minSteps;
        prompt->stepsLog =
            std::log1p(static_cast<float>(prompt->steps)) / std::log1p(60.0f);
        prompt->stepsBucket = prompt->steps / 5;

        // upscaler selection and settings
        if (!upscalers_.empty())
        {
            const std::string &selectedUpscaler = upscalers_[rng(upscalers_.size())];
            auto upscalerToken = tokenizer_.upscalerToToken(selectedUpscaler);
            prompt->upscaler = upscalerToken.value_or(0);
            prompt->upscalerEnabled = true;
            prompt->upscalerSteps = rng(16) + 10;
        }
        else
        {
            prompt->upscaler = 0;
            prompt->upscalerEnabled = false;
            prompt->upscalerSteps = 0;
        }

        // Denoising strength (0.0-1.0 range)
        prompt->denoisingStrength = static_cast<float>(rng(1001)) / 1000.0f;

        counter++;

        return prompt;
    }

    std::string Generator::formatPrompt(const generator::ScoredPrompt &processedPrompt) const
    {
        std::stringstream ss;
        bool hasWrittenAnyTag = false;
        const bool hasClipPrompt = (modelSettings_.modelType == "clip_embed" &&
                                    clipEmbedCache_ != nullptr &&
                                    processedPrompt.prompt->clipDim > 0 &&
                                    !processedPrompt.prompt->clipTagIds.empty());
        if (hasClipPrompt)
        {
            for (size_t i = 0; i < processedPrompt.prompt->clipTagIds.size(); ++i)
            {
                uint32_t tagId = processedPrompt.prompt->clipTagIds[i];
                if (static_cast<size_t>(tagId) < clipEmbedCache_->vocabSize())
                {
                    if (hasWrittenAnyTag)
                    {
                        ss << ", ";
                    }
                    ss << clipEmbedCache_->getTagById(tagId);
                    hasWrittenAnyTag = true;
                }
            }
        }
        else
        {
            // one-hot: reconstruct from token IDs
            for (int64_t i = 0; i < processedPrompt.prompt->sequenceLength; ++i)
            {
                int64_t tokenId = processedPrompt.prompt->tokens[i];
                if (tokenId == 0)
                    break;
                auto textOpt = tokenizer_.tokenToText(tokenId);
                if (textOpt)
                {
                    std::string text = *textOpt;
                    if (hasWrittenAnyTag)
                    {
                        ss << ", ";
                    }
                    ss << text;
                    hasWrittenAnyTag = true;
                }
            }
        }
        // append LoRAs if any
        for (size_t i = 0; i < static_cast<size_t>(processedPrompt.prompt->numLoras); ++i)
        {
            int64_t loraId = processedPrompt.prompt->loras[i];
            if (loraId == 0)
                break;
            auto textOpt = tokenizer_.loraTokenToText(loraId); // lora xxx
            if (textOpt)
            {
                std::string text = *textOpt;
                text = text.substr(5);
                const Lora *l = lora_controller_.get_lora_by_name(text);
                if (l == nullptr)
                {
                    std::cerr << "Unknown lora: " << text << std::endl;
                    continue;
                }
                if (hasWrittenAnyTag)
                {
                    ss << ", ";
                }
                // <lora: or <lyco:
                std::string type = "<" + l->get_prefix_tag() + ":";
                ss << type << text << ":" << processedPrompt.prompt->weights[i] << ">";
                hasWrittenAnyTag = true;
            }
        }

        std::string tags = ss.str();
        Prompt prompt = *processedPrompt.prompt;

        // shoudn't fail
        std::string samplerName = *tokenizer_.samplerTokenToText(prompt.sampler);

        // Get upscaler name from tokenizer
        std::string upscalerName = "None"; // fallback
        if (auto upscalerOpt = tokenizer_.upscalerTokenToText(prompt.upscaler))
        {
            upscalerName = *upscalerOpt;
        }

        std::string promptFormat =
            "--prompt \"" + positive_prompt_ + ", {tags}\" --negative_prompt \"" + negative_prompt_ + "\" --sampler_name \"" + samplerName + "\" --steps " + std::to_string(prompt.steps) + " --seed -1 --width " + std::to_string(width_) + " --height " + std::to_string(height_) + " --cfg_scale {cfg_scale} --batch_size " + std::to_string(batch_size_) + " --n_iter " + std::to_string(n_iter_);

        // add upscaler settings if enabled
        if (prompt.upscalerEnabled)
        {
            std::ostringstream hrScaleStream;
            hrScaleStream << hr_scale_;
            promptFormat += " --enable_hr true --hr_scale " + hrScaleStream.str() + " --hr_second_pass_steps " + std::to_string(prompt.upscalerSteps) + " --denoising_strength " + std::to_string(prompt.denoisingStrength) + " --hr_upscaler \"" + upscalerName + "\"";
        }
        else
        {
            promptFormat += " --enable_fr false";
        }

        if (!prompt_extra_settings_.empty())
            promptFormat += " " + prompt_extra_settings_;

        // replace tags and cfg_scale in the format
        size_t tagsPos = promptFormat.find("{tags}");
        if (tagsPos != std::string::npos)
        {
            promptFormat.replace(tagsPos, 6, tags);
        }

        size_t cfgPos = promptFormat.find("{cfg_scale}");
        if (cfgPos != std::string::npos)
        {
            // Output raw (unnormalized) cfg_scale, rounded to nearest 0.5
            float rawCfg = std::round(prompt.cfgScale * 2.0f) / 2.0f;
            promptFormat.replace(cfgPos, 11, std::to_string(rawCfg));
        }

        // Prepend model identifier if model name is set
        if (!modelSettings_.modelName.empty())
        {
            promptFormat = "--sd_model " + modelSettings_.modelName + " " + promptFormat;
        }

        return promptFormat;
    }
} // namespace generator