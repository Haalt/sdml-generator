#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include "models/Lora_controller.h"
#include "models/Tag_list.h"
#include "tokenizer/tokenizer.hpp"
#include "utils/types.hpp"

namespace generator
{
    class Generator
    {
        Lora_controller &lora_controller_;
        Tag_list &tag_list_;
        Tokenizer &tokenizer_;

        std::vector<std::string> samplers_;
        std::vector<std::string> upscalers_;
        std::vector<std::string> vanilla_extra_tags_;

        std::vector<float> cfg_scales_;

        size_t width_, height_, n_iter_, batch_size_, hr_scale_;
        std::string positive_prompt_, negative_prompt_;

        void load_json_config(const std::string &config_path)
        {
            // defaults
            samplers_.clear();
            upscalers_.clear();

            width_ = 512;
            height_ = 512;
            n_iter_ = 8;
            batch_size_ = 1;
            hr_scale_ = 2;
            positive_prompt_.clear();
            negative_prompt_.clear();
            vanilla_extra_tags_.clear();

            std::ifstream in(config_path);
            if (!in)
            {
                std::cerr << "Failed to open generator config: " << config_path << std::endl;
                throw std::runtime_error("generator config open failed");
            }

            nlohmann::json j;
            in >> j;

            if (j.contains("samplers") && j["samplers"].is_array())
            {
                samplers_.clear();
                for (const auto &v : j["samplers"])
                    if (v.is_string())
                        samplers_.push_back(v.get<std::string>());
            }
            if (samplers_.empty())
            {
                std::cerr << "No samplers provided in generator config (samplers array missing or empty)." << std::endl;
                throw std::runtime_error("no samplers provided in config");
            }

            if (j.contains("upscalers") && j["upscalers"].is_array())
            {
                upscalers_.clear();
                for (const auto &v : j["upscalers"])
                    if (v.is_string())
                        upscalers_.push_back(v.get<std::string>());
            }
            if (upscalers_.empty())
            {
                std::cerr << "No upscalers provided in generator config; defaulting to {\"None\"}." << std::endl;
                upscalers_ = {"None"};
            }

            if (j.contains("width") && j["width"].is_number_unsigned())
                width_ = j["width"].get<size_t>();
            if (j.contains("height") && j["height"].is_number_unsigned())
                height_ = j["height"].get<size_t>();
            if (j.contains("n_iter") && j["n_iter"].is_number_unsigned())
                n_iter_ = j["n_iter"].get<size_t>();
            if (j.contains("batch_size") && j["batch_size"].is_number_unsigned())
                batch_size_ = j["batch_size"].get<size_t>();
            if (j.contains("hr_scale") && j["hr_scale"].is_number_unsigned())
                hr_scale_ = j["hr_scale"].get<size_t>();

            if (j.contains("positive_prompt") && j["positive_prompt"].is_string())
                positive_prompt_ = j["positive_prompt"].get<std::string>();
            if (j.contains("negative_prompt") && j["negative_prompt"].is_string())
                negative_prompt_ = j["negative_prompt"].get<std::string>();

            if (j.contains("vanilla_extra_tags") && j["vanilla_extra_tags"].is_array())
            {
                vanilla_extra_tags_.clear();
                for (const auto &v : j["vanilla_extra_tags"])
                    if (v.is_string())
                        vanilla_extra_tags_.push_back(v.get<std::string>());
            }
        }

    public:
        Generator(Lora_controller &lora_controller, Tag_list &taglist, Tokenizer &tokenizer, const std::string &config_path) : lora_controller_(lora_controller), tag_list_(taglist), tokenizer_(tokenizer)
        {
            for (double val = 3.0; val <= 11.0; val += 0.5)
                cfg_scales_.push_back(val);

            load_json_config(config_path);
        }

        std::unique_ptr<Prompt> generatePrompt(void) const;

        std::string formatPrompt(const ScoredPrompt &prompt) const;
    };

} // namespace generator