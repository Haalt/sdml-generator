#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace generator
{

    constexpr size_t MAX_SEQUENCE_LENGTH = 81; // Maximum token sequence length
    constexpr int64_t MAX_LORAS = 7;           // Maximum number of LoRA tokens
    constexpr float MAX_CFG_SCALE = 11.0f;     // Maximum CFG scale for normalization

    struct Prompt
    {
        std::vector<int64_t> tokens;
        std::vector<int64_t> loras;
        std::vector<float> weights;
        float cfgScale;
        int64_t sequenceLength;
        int64_t numLoras;

        int64_t sampler;         // Sampler token ID
        int64_t steps;           // Raw steps value (30-60)
        float stepsLog;          // Log-scaled steps [0-1]
        int64_t stepsBucket;     // Bucketed steps (0-9)
        int64_t upscaler;        // Upscaler token ID
        bool upscalerEnabled;    // Whether upscaler is used
        int64_t upscalerSteps;   // Upscaler steps (10-25)
        float denoisingStrength; // Denoising strength [0-1]
    };

    // Batch of prompts for inference
    struct PromptBatch
    {
        std::vector<int64_t> token_indices; // [batch_size * MAX_SEQUENCE_LENGTH]
        std::vector<float> token_mask;      // [batch_size * MAX_SEQUENCE_LENGTH]
        std::vector<int64_t> lora_ids;      // [batch_size * MAX_LORAS]
        std::vector<float> lora_weights;    // [batch_size * MAX_LORAS]
        std::vector<float> cfg_scales;      // [batch_size]
        std::vector<float> num_loras;       // [batch_size]
        std::vector<int64_t> sampler_ids;   // [batch_size]
        std::vector<float> steps_log;       // [batch_size]
        std::vector<int64_t> steps_bucket;  // [batch_size]
        std::vector<int64_t> upscaler_ids;  // [batch_size]
        std::vector<float> up_has;          // [batch_size]
        std::vector<float> up_steps;        // [batch_size]
        std::vector<float> denoise;         // [batch_size]
        size_t batch_size;
        size_t sequence_length;
    };

    // prompt paired with its model score
    struct ScoredPrompt
    {
        float score;
        std::shared_ptr<Prompt> prompt;
    };

} // namespace generator