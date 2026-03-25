#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace generator
{
    /**
     * @brief Configuration for a single SD model (e.g., SD1.5, SDXL)
     *
     * Each model profile encapsulates all model-specific settings including
     * dictionary, LoRA configuration, generation parameters, tokenizer behavior,
     * and scoring model configuration.
     */
    struct ModelProfile
    {
        static constexpr const char *MODEL_TYPE_ONE_HOT = "one-hot";
        static constexpr const char *MODEL_TYPE_CLIP_EMBED = "clip_embed";
        static constexpr const char *GENERATION_MODE_EXPERT_DICT = "expert_dict";
        static constexpr const char *GENERATION_MODE_CLIP_RANDOM_VOCAB = "clip_random_vocab";

        int32_t profileId{0};             // Unique profile index (auto-assigned, 0, 1, 2, ...)
        int32_t modelId{0};               // SD model identifier (can be shared across profiles)
        std::string name;                 // Human-readable name ("sd15", "sdxl")
        std::string modelType{MODEL_TYPE_ONE_HOT}; // Scoring model type: "one-hot" or "clip_embed"
        std::string generationMode;       // Prompt generation policy (empty = derive from modelType)
        std::string dictPath;             // Path to tag dictionary JSON
        std::string lorasPath;            // Path to LoRAs JSON (empty = no LoRAs)
        std::string generatorConfigPath;  // Path to generator config JSON
        bool stripParentheses{true};      // Tokenizer: strip parentheses from tags
        int minSteps{20};                 // Minimum generation steps
        int maxSteps{30};                 // Maximum generation steps
        bool useLoras{true};              // Whether this model uses LoRAs
        bool useTagsAsLoras{false};       // Filtering: use tags as LoRA buckets
        // clip_embed specific paths
        std::string tagVocabPath;  // Path to tag vocab JSON (clip_embed only)
        std::string clipCachePath; // Path to binary CLIP embedding cache (clip_embed only)
        // Per-model scoring configuration
        std::string onnxModelPath;  // Path to ONNX scoring model for this SD model
        float scoreThreshold{0.8f}; // Minimum score to accept prompts for this model
        int batchSize{0};           // Scoring batch size (0 = auto/default)
        int maxOutputPrompts{0};    // Maximum output prompts for this model (0 = unlimited)

        bool usesClipScoring() const
        {
            return modelType == MODEL_TYPE_CLIP_EMBED;
        }

        std::string effectiveGenerationMode() const
        {
            if (!generationMode.empty())
            {
                return generationMode;
            }
            return usesClipScoring() ? GENERATION_MODE_CLIP_RANDOM_VOCAB : GENERATION_MODE_EXPERT_DICT;
        }

        /**
         * @brief Validates the profile configuration
         * @throws std::runtime_error if validation fails
         */
        void validate() const
        {
            if (name.empty())
            {
                throw std::runtime_error("ModelProfile: name cannot be empty");
            }
            if (modelType != MODEL_TYPE_ONE_HOT && modelType != MODEL_TYPE_CLIP_EMBED)
            {
                throw std::runtime_error("ModelProfile '" + name + "': modelType must be 'one-hot' or 'clip_embed'");
            }
            const std::string resolvedGenerationMode = effectiveGenerationMode();
            if (resolvedGenerationMode != GENERATION_MODE_EXPERT_DICT &&
                resolvedGenerationMode != GENERATION_MODE_CLIP_RANDOM_VOCAB)
            {
                throw std::runtime_error(
                    "ModelProfile '" + name + "': generationMode must be 'expert_dict' or 'clip_random_vocab'");
            }
            if (resolvedGenerationMode == GENERATION_MODE_EXPERT_DICT && dictPath.empty())
            {
                throw std::runtime_error(
                    "ModelProfile '" + name + "': dictPath cannot be empty for expert_dict generation");
            }
            if (generatorConfigPath.empty())
            {
                throw std::runtime_error("ModelProfile '" + name + "': generatorConfigPath cannot be empty");
            }
            if (onnxModelPath.empty())
            {
                throw std::runtime_error("ModelProfile '" + name + "': onnxModelPath cannot be empty");
            }
            if (minSteps <= 0 || maxSteps <= 0)
            {
                throw std::runtime_error("ModelProfile '" + name + "': steps must be positive");
            }
            if (minSteps > maxSteps)
            {
                throw std::runtime_error("ModelProfile '" + name + "': minSteps cannot exceed maxSteps");
            }
            if (useLoras && lorasPath.empty())
            {
                throw std::runtime_error("ModelProfile '" + name + "': useLoras is true but lorasPath is empty");
            }
            if (scoreThreshold < 0.0f || scoreThreshold > 1.0f)
            {
                throw std::runtime_error("ModelProfile '" + name + "': scoreThreshold must be between 0 and 1");
            }
            if (batchSize < 0)
            {
                throw std::runtime_error("ModelProfile '" + name + "': batchSize cannot be negative");
            }
            if (maxOutputPrompts < 0)
            {
                throw std::runtime_error("ModelProfile '" + name + "': maxOutputPrompts cannot be negative");
            }
            if (usesClipScoring())
            {
                if (clipCachePath.empty())
                {
                    throw std::runtime_error("ModelProfile '" + name + "': clip_embed requires clipCachePath");
                }
            }
            if (resolvedGenerationMode == GENERATION_MODE_CLIP_RANDOM_VOCAB)
            {
                if (!usesClipScoring())
                {
                    throw std::runtime_error(
                        "ModelProfile '" + name + "': clip_random_vocab generation requires clip_embed scoring");
                }
                if (clipCachePath.empty())
                {
                    throw std::runtime_error(
                        "ModelProfile '" + name + "': clip_random_vocab generation requires clipCachePath");
                }
            }
        }
    };

    /**
     * @brief Master configuration containing shared resources and all model profiles
     */
    struct ModelsConfig
    {
        std::string tokenizerPath;          // Shared tokenizer JSON
        std::vector<ModelProfile> profiles; // List of model profiles

        /**
         * @brief Loads configuration from a JSON file
         * @param configPath Path to the models_config.json file
         * @return Parsed ModelsConfig
         * @throws std::runtime_error if loading or parsing fails
         */
        static ModelsConfig loadFromJson(const std::string &configPath)
        {
            std::ifstream file(configPath);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open models config file: " + configPath);
            }
            nlohmann::json json;
            try
            {
                file >> json;
            }
            catch (const nlohmann::json::parse_error &e)
            {
                throw std::runtime_error("Failed to parse models config JSON: " + std::string(e.what()));
            }
            ModelsConfig config;
            // Parse shared settings
            if (!json.contains("tokenizer") || !json["tokenizer"].is_string())
            {
                throw std::runtime_error("ModelsConfig: 'tokenizer' is required");
            }
            config.tokenizerPath = json["tokenizer"].get<std::string>();
            // Optional top-level fallback values for backward compatibility
            std::string fallbackOnnxModel;
            float fallbackScoreThreshold = 0.8f;
            if (json.contains("onnx_model") && json["onnx_model"].is_string())
            {
                fallbackOnnxModel = json["onnx_model"].get<std::string>();
            }
            if (json.contains("score_threshold") && json["score_threshold"].is_number())
            {
                fallbackScoreThreshold = json["score_threshold"].get<float>();
            }
            // Parse profiles
            if (!json.contains("profiles") || !json["profiles"].is_array())
            {
                throw std::runtime_error("ModelsConfig: 'profiles' array is required");
            }
            int32_t nextProfileId = 0;
            for (const auto &profileJson : json["profiles"])
            {
                ModelProfile profile;
                profile.profileId = nextProfileId++;
                if (profileJson.contains("id") && profileJson["id"].is_number_integer())
                {
                    profile.modelId = profileJson["id"].get<int32_t>();
                }
                if (profileJson.contains("name") && profileJson["name"].is_string())
                {
                    profile.name = profileJson["name"].get<std::string>();
                }
                if (profileJson.contains("dict") && profileJson["dict"].is_string())
                {
                    profile.dictPath = profileJson["dict"].get<std::string>();
                }
                if (profileJson.contains("loras") && profileJson["loras"].is_string())
                {
                    profile.lorasPath = profileJson["loras"].get<std::string>();
                }
                if (profileJson.contains("generator_config") && profileJson["generator_config"].is_string())
                {
                    profile.generatorConfigPath = profileJson["generator_config"].get<std::string>();
                }
                if (profileJson.contains("strip_parentheses") && profileJson["strip_parentheses"].is_boolean())
                {
                    profile.stripParentheses = profileJson["strip_parentheses"].get<bool>();
                }
                if (profileJson.contains("min_steps") && profileJson["min_steps"].is_number_integer())
                {
                    profile.minSteps = profileJson["min_steps"].get<int>();
                }
                if (profileJson.contains("max_steps") && profileJson["max_steps"].is_number_integer())
                {
                    profile.maxSteps = profileJson["max_steps"].get<int>();
                }
                if (profileJson.contains("use_loras") && profileJson["use_loras"].is_boolean())
                {
                    profile.useLoras = profileJson["use_loras"].get<bool>();
                }
                if (profileJson.contains("use_tags_as_loras") && profileJson["use_tags_as_loras"].is_boolean())
                {
                    profile.useTagsAsLoras = profileJson["use_tags_as_loras"].get<bool>();
                }
                if (profileJson.contains("model_type") && profileJson["model_type"].is_string())
                {
                    profile.modelType = profileJson["model_type"].get<std::string>();
                }
                if (profileJson.contains("generation_mode") && profileJson["generation_mode"].is_string())
                {
                    profile.generationMode = profileJson["generation_mode"].get<std::string>();
                }
                if (profileJson.contains("tag_vocab") && profileJson["tag_vocab"].is_string())
                {
                    profile.tagVocabPath = profileJson["tag_vocab"].get<std::string>();
                }
                if (profileJson.contains("clip_cache") && profileJson["clip_cache"].is_string())
                {
                    profile.clipCachePath = profileJson["clip_cache"].get<std::string>();
                }
                if (profileJson.contains("max_output_prompts") && profileJson["max_output_prompts"].is_number_integer())
                {
                    profile.maxOutputPrompts = profileJson["max_output_prompts"].get<int>();
                }
                // Per-model scoring configuration (with fallback to top-level)
                if (profileJson.contains("onnx_model") && profileJson["onnx_model"].is_string())
                {
                    profile.onnxModelPath = profileJson["onnx_model"].get<std::string>();
                }
                else
                {
                    profile.onnxModelPath = fallbackOnnxModel;
                }
                if (profileJson.contains("score_threshold") && profileJson["score_threshold"].is_number())
                {
                    profile.scoreThreshold = profileJson["score_threshold"].get<float>();
                }
                else
                {
                    profile.scoreThreshold = fallbackScoreThreshold;
                }
                if (profileJson.contains("batch_size") && profileJson["batch_size"].is_number_integer())
                {
                    profile.batchSize = profileJson["batch_size"].get<int>();
                }
                profile.validate();
                config.profiles.push_back(std::move(profile));
            }
            if (config.profiles.empty())
            {
                throw std::runtime_error("ModelsConfig: at least one profile is required");
            }
            return config;
        }

        /**
         * @brief Gets a profile by its unique profile index
         * @param profileId The auto-assigned profile index
         * @return Pointer to the profile, or nullptr if out of range
         */
        const ModelProfile *getProfileByProfileId(int32_t profileId) const
        {
            if (profileId >= 0 && static_cast<size_t>(profileId) < profiles.size() &&
                profiles[static_cast<size_t>(profileId)].profileId == profileId)
            {
                return &profiles[static_cast<size_t>(profileId)];
            }
            // Fallback linear search (shouldn't be needed)
            for (const auto &p : profiles)
            {
                if (p.profileId == profileId)
                    return &p;
            }
            return nullptr;
        }

        /**
         * @brief Gets the first profile with a given SD model ID
         * @param modelId The SD model ID to look up (may match multiple profiles)
         * @return Pointer to the first matching profile, or nullptr if not found
         */
        const ModelProfile *getProfileById(int32_t modelId) const
        {
            for (const auto &p : profiles)
            {
                if (p.modelId == modelId)
                    return &p;
            }
            return nullptr;
        }

        /**
         * @brief Gets a profile by name
         * @param name The model name to look up
         * @return Pointer to the profile, or nullptr if not found
         */
        const ModelProfile *getProfileByName(const std::string &name) const
        {
            for (const auto &p : profiles)
            {
                if (p.name == name)
                    return &p;
            }
            return nullptr;
        }

        /**
         * @brief Prints configuration summary to stdout
         */
        void printSummary() const
        {
            std::cout << "===== Models Configuration =====" << std::endl;
            std::cout << "Tokenizer: " << tokenizerPath << std::endl;
            std::cout << "Number of Profiles: " << profiles.size() << std::endl;
            for (const auto &p : profiles)
            {
                std::cout << "\n  Profile " << p.profileId << " [SD Model " << p.modelId << "] " << p.name << std::endl;
                std::cout << "    Model Type: " << p.modelType << std::endl;
                std::cout << "    Generation Mode: " << p.effectiveGenerationMode() << std::endl;
                std::cout << "    ONNX Model: " << p.onnxModelPath << std::endl;
                std::cout << "    Score Threshold: " << p.scoreThreshold << std::endl;
                std::cout << "    Batch Size: " << (p.batchSize > 0 ? std::to_string(p.batchSize) : "auto") << std::endl;
                std::cout << "    Dict: " << p.dictPath << std::endl;
                std::cout << "    LoRAs: " << (p.lorasPath.empty() ? "(none)" : p.lorasPath) << std::endl;
                std::cout << "    Generator Config: " << p.generatorConfigPath << std::endl;
                std::cout << "    Strip Parentheses: " << (p.stripParentheses ? "yes" : "no") << std::endl;
                std::cout << "    Steps: " << p.minSteps << "-" << p.maxSteps << std::endl;
                std::cout << "    Use LoRAs: " << (p.useLoras ? "yes" : "no") << std::endl;
                std::cout << "    Use Tags as LoRAs: " << (p.useTagsAsLoras ? "yes" : "no") << std::endl;
                std::cout << "    Max Output Prompts: " << (p.maxOutputPrompts > 0 ? std::to_string(p.maxOutputPrompts) : "unlimited") << std::endl;
                if (p.usesClipScoring())
                {
                    std::cout << "    Tag Vocab: " << (p.tagVocabPath.empty() ? "(from clip cache)" : p.tagVocabPath) << std::endl;
                    std::cout << "    CLIP Cache: " << p.clipCachePath << std::endl;
                }
            }
            std::cout << "================================" << std::endl;
        }
    };

} // namespace generator
