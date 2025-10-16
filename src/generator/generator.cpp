#include <cmath>
#include <memory>
#include <random>
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

    std::unique_ptr<Prompt> Generator::generatePrompt(void) const
    {
        // Initialize high-performance PCG random generator
        static thread_local pcg32 rng(std::random_device{}());
        static thread_local uint32_t counter = 0;

        if (counter % 5000 == 0)
            lora_controller_.reroll_loras();

        tag_list_.reset();
        tag_list_.generate_tags(rng(21) + 10); // 10-30 tags

        Lora *selected_main = nullptr;
        Lora *selected_conditional = nullptr;
        std::unordered_set<Lora *> selected_loras;

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

        // first, apply banned tag filtering from selected LoRAs
        for (Lora *l : selected_loras)
        {
            tag_list_.set_tags(l->remove_banned_tags(tag_list_.get_tags()));
        }

        if (selected_conditional != nullptr)
        {
            tag_list_.set_tags(selected_conditional->remove_banned_tags(tag_list_.get_tags()));
        }

        // collect all token vectors and LoRA tokens
        std::vector<std::vector<int64_t>> all_token_vectors;
        std::vector<std::pair<int64_t, float>> all_lora_tokens;

        // add base tag tokens
        const auto &base_tokens = tag_list_.get_precomputed_tokens();
        all_token_vectors.insert(all_token_vectors.end(), base_tokens.begin(), base_tokens.end());

        // add LoRA tokens from selected LoRAs
        for (const Lora *lora : selected_loras)
        {
            auto [token_vectors, lora_tokens] = lora->get_precomputed_tokens();
            all_token_vectors.insert(all_token_vectors.end(), token_vectors.begin(), token_vectors.end());
            all_lora_tokens.insert(all_lora_tokens.end(), lora_tokens.begin(), lora_tokens.end());
        }

        // add conditional LoRA tokens
        if (selected_conditional != nullptr)
        {
            auto [token_vectors, lora_tokens] = selected_conditional->get_precomputed_tokens();
            all_token_vectors.insert(all_token_vectors.end(), token_vectors.begin(), token_vectors.end());
            all_lora_tokens.insert(all_lora_tokens.end(), lora_tokens.begin(), lora_tokens.end());
        }

        // check for tag LoRAs and add their tokens
        std::vector<std::string> tags = tag_list_.get_tags();
        bool is_vanilla = selected_loras.empty() && selected_conditional == nullptr;

        const std::unordered_map<std::string, TagLora *> &tag_loramap = lora_controller_.get_tag_lora_map();

        for (const std::string &tag : tags)
        {
            auto res = tag_loramap.find(tag);
            if (res != tag_loramap.end())
            {
                is_vanilla = false;
                const Lora *lora = static_cast<const Lora *>(res->second);
                auto [token_vectors, lora_tokens] = lora->get_precomputed_tokens();
                all_token_vectors.insert(all_token_vectors.end(), token_vectors.begin(), token_vectors.end());
                all_lora_tokens.insert(all_lora_tokens.end(), lora_tokens.begin(), lora_tokens.end());
            }
        }

        // add configured extra tag for vanilla prompts (if any)
        if (is_vanilla && rng(2) && !vanilla_extra_tags_.empty())
        {
            const std::string &extraTag = vanilla_extra_tags_[rng(vanilla_extra_tags_.size())];
            auto extra_tokens = tokenizer_.precomputeTagTokens(extraTag);
            if (!extra_tokens.empty())
            {
                all_token_vectors.push_back(extra_tokens);
            }
        }

        auto prompt = std::make_unique<Prompt>();
        *prompt = tokenizer_.assemblePrompt(all_token_vectors, all_lora_tokens, cfg_scales_[rng(cfg_scales_.size())]);

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

        // steps (30-60 range)
        prompt->steps = rng(31) + 30;
        prompt->stepsLog =
            std::log1p(static_cast<float>(prompt->steps)) / std::log1p(60.0f);
        prompt->stepsBucket = prompt->steps / 60; // TODO: FIX ONCE MODELS SPEC IS CORRECT

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

        for (int64_t i = 0; i < processedPrompt.prompt->sequenceLength; ++i)
        {
            int64_t tokenId = processedPrompt.prompt->tokens[i];
            if (tokenId == 0)
                break;

            auto textOpt = tokenizer_.tokenToText(tokenId);
            if (textOpt)
            {
                std::string text = *textOpt;
                ss << text;

                if (i < processedPrompt.prompt->sequenceLength - 1 || processedPrompt.prompt->numLoras != 0)
                    ss << ", ";
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

                // <lora: or <lyco:
                std::string type = "<" + l->get_prefix_tag() + ":";

                ss << type << text << ":" << processedPrompt.prompt->weights[i] << ">";

                if (i + 1 < static_cast<size_t>(processedPrompt.prompt->numLoras))
                    ss << ", ";
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
            promptFormat += " --enable_hr true --hr_scale " + std::to_string(hr_scale_) + " --hr_second_pass_steps " + std::to_string(prompt.upscalerSteps) + " --denoising_strength " + std::to_string(prompt.denoisingStrength) + " --hr_upscaler \"" + upscalerName + "\"";
        }
        else
        {
            promptFormat += " --enable_fr false";
        }

        // replace tags and cfg_scale in the format
        size_t tagsPos = promptFormat.find("{tags}");
        if (tagsPos != std::string::npos)
        {
            promptFormat.replace(tagsPos, 6, tags);
        }

        size_t cfgPos = promptFormat.find("{cfg_scale}");
        if (cfgPos != std::string::npos)
        {
            float normalizedCfg = prompt.cfgScale;
            // round to nearest 0.5
            normalizedCfg = std::round(normalizedCfg * 2.0f) / 2.0f;

            promptFormat.replace(cfgPos, 11, std::to_string(normalizedCfg));
        }

        return promptFormat;
    }
} // namespace generator