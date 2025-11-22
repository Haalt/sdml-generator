#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace generator
{

    constexpr size_t MAX_SEQUENCE_LENGTH = 81;            // Maximum token sequence length
    constexpr int64_t MAX_LORAS = 7;                      // Maximum number of LoRA tokens
    constexpr float MAX_CFG_SCALE = 11.0f;                // Maximum CFG scale for normalization (matches Python training)
    constexpr unsigned DEFAULT_SCORING_BATCH_SIZE = 4096; // Default scoring batch size
    constexpr unsigned MIN_SCORING_BATCH_SIZE = 64;       // Lower bound for scoring batch size

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
        int32_t modelId{0};      // SD model ID (for ONNX model input)
        int32_t profileId{0};    // Unique profile index (for internal routing)

        // clip_embed model: mean-pooled CLIP embedding in FP16 (d_clip uint16 values, e.g. 768)
        std::vector<uint16_t> clipEmb;
        // clip_embed model: sampled vocab tag IDs (for deferred formatting)
        std::vector<uint32_t> clipTagIds;
        // clip_embed model: embedding dimension used by scorer input
        uint32_t clipDim{0};
    };

    // Batch of prompts for inference
    struct PromptBatch
    {
        // one-hot model inputs
        std::vector<int64_t> token_indices; // [batch_size * MAX_SEQUENCE_LENGTH]
        std::vector<float> token_mask;      // [batch_size * MAX_SEQUENCE_LENGTH]
        // clip_embed model input (FP16 host buffer)
        std::vector<uint16_t> clip_emb; // [batch_size * clip_dim]
        // clip_embed tag metadata for CUDA pooling
        std::vector<uint32_t> clip_tag_ids_flat; // concatenated tag IDs for all samples
        std::vector<uint32_t> clip_tag_offsets;  // size batch_size+1, prefix offsets into clip_tag_ids_flat
        size_t clip_dim{0};                      // 0 = one-hot, >0 = clip_embed (e.g. 768)
        // shared inputs
        std::vector<int64_t> lora_ids;     // [batch_size * MAX_LORAS]
        std::vector<float> lora_weights;   // [batch_size * MAX_LORAS]
        std::vector<float> cfg_scales;     // [batch_size]
        std::vector<float> num_loras;      // [batch_size]
        std::vector<int64_t> sampler_ids;  // [batch_size]
        std::vector<float> steps_log;      // [batch_size]
        std::vector<int64_t> steps_bucket; // [batch_size]
        std::vector<int64_t> upscaler_ids; // [batch_size]
        std::vector<float> up_has;         // [batch_size]
        std::vector<float> up_steps;       // [batch_size]
        std::vector<float> denoise;        // [batch_size]
        std::vector<int64_t> model_ids;    // [batch_size]
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