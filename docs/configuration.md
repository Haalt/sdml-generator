# Configuration Files

The SDML Generator uses three main JSON configuration files to define how prompts are generated, scored, and formatted. Each file serves a specific purpose in the generation pipeline.

## Overview

- **`dict.json`**: Tag dictionary defining categories, hierarchies, and selection rules
- **`loras.json`**: LoRA/LyCORIS model definitions, weights, and compatibility matrix
- **`generator_config.json`**: Technical generation settings (samplers, dimensions, etc.)

## Models Configuration (`models_config.json`)

The runtime profile config now separates generation behavior from scoring representation on a per-profile basis.

```json
{
  "tokenizer": "tokenizer.json",
  "profiles": [
    {
      "id": 0,
      "name": "sd15",
      "model_type": "one-hot",
      "generation_mode": "expert_dict",
      "dict": "dicts/sd15.json",
      "loras": "loras/sd15.json",
      "generator_config": "configs/sd15.json",
      "use_loras": true
    },
    {
      "id": 1,
      "name": "sdxl_clip_expert",
      "model_type": "clip_embed",
      "generation_mode": "expert_dict",
      "dict": "dicts/sdxl.json",
      "clip_cache": "clip/cache.bin",
      "generator_config": "configs/sdxl.json",
      "use_loras": false,
      "use_tags_as_loras": true
    }
  ]
}
```

### Profile Fields

- **`model_type`**: Chooses the scorer input representation. Supported values are `one-hot` and `clip_embed`.
- **`generation_mode`**: Chooses how prompts are generated. Supported values are `expert_dict` and `clip_random_vocab`.
- **`dict`**: Required when `generation_mode` is `expert_dict`.
- **`clip_cache`**: Required when `model_type` is `clip_embed`.
- **`tag_vocab`**: Optional when `model_type` is `clip_embed`. If omitted, the cache row names are used as the CLIP vocabulary.

### Supported Combinations

- **`one-hot` + `expert_dict`**: Existing full expert-system generation with token-based scoring.
- **`clip_embed` + `clip_random_vocab`**: Existing flat-vocab CLIP sampling path.
- **`clip_embed` + `expert_dict`**: Expert-system generation with CLIP scoring. Dict tags are validated against the CLIP vocab at startup, and CLIP tag IDs are precomputed before the hot path starts.

If `generation_mode` is omitted, the default remains backward compatible:

- **`one-hot`** defaults to **`expert_dict`**
- **`clip_embed`** defaults to **`clip_random_vocab`**

## Tag Dictionary (`dict.json`)

**Purpose**: Defines the vocabulary and rules for tag-based prompt generation.

### Structure

```json
{
  "tags": {
    "category_name": ["tag1", "tag2", "tag3"],
    "nested_category": {
      "subcategory": ["subtag1", "subtag2"]
    }
  },
  "rules": {
    "unique_categories": ["category_name"],
    "filter_categories": ["category_name"],
    "exclusive_groups": {
      "group_name": ["category1", "category2"]
    },
    "incompatible_categories": {
      "category1": ["category2", "category3"]
    }
  }
}
```

### Tags Section

The `tags` section defines a hierarchical structure of categories and their associated tags:

- **Categories**: Top-level keys representing tag groupings (e.g., "characters", "styles", "lighting")
- **Tag Arrays**: Simple arrays of string tags for a category
- **Nested Categories**: Objects containing subcategories for more complex hierarchies

**Examples**:

```json
{
  "characters": ["1girl", "1boy", "cat ears", "fox tail"],
  "styles": {
    "anime": ["anime", "manga style", "cel shading"],
    "realistic": ["photorealistic", "hyperrealistic", "detailed"]
  },
  "lighting": [
    "soft lighting",
    "dramatic shadows",
    "golden hour",
    "rim lighting"
  ]
}
```

### Rules Section

The `rules` section defines constraints and relationships between categories:

#### Unique Categories

Categories where only one tag can be selected per generation:

```json
"unique_categories": ["characters", "background"]
```

#### Filter Categories

Optional categories used for tag-as-LoRA diversity filtering when `use_tags_as_loras` is enabled:

```json
"filter_categories": ["characters", "clothes.clothing"]
```

If omitted, filtering falls back to `unique_categories` for backward compatibility. If present as an empty list, no dictionary categories participate in tag-as-LoRA filtering.

#### Exclusive Groups

Groups of categories that cannot be used together:

```json
"exclusive_groups": {
  "style_conflict": ["styles.anime", "styles.realistic"],
  "lighting_conflict": ["lighting.day", "lighting.night"]
}
```

#### Incompatible Categories

Per-category incompatibility mapping:

```json
"incompatible_categories": {
  "styles.anime": ["styles.realistic", "styles.photography"],
  "characters.robot": ["characters.organic"]
}
```

### Category Path Notation

Use dot notation for nested categories:

- "`styles`" - refers to the entire styles category
- "`styles.anime`" - refers to the anime subcategory within styles
- "`lighting.outdoor.day`" - refers to nested subcategories

### Validation

The generator validates:

- Category names must be valid identifiers (`[a-zA-Z_][a-zA-Z0-9_]*`)
- Referenced categories in rules must exist in the tags section
- No circular dependencies in incompatibility rules
- Maximum 64 categories (due to bitmask optimization)

## LoRA/LyCORIS Configuration (`loras.json`)

**Purpose**: Defines LoRA/LyCORIS models, their properties, and compatibility relationships.

### Structure

```json
{
  "instances": [
    {
      "name": "unique_lora_name",
      "class": "MainLora|SecondaryLora|TagLora|ConditionalVariantLora",
      "disabled": false,
      "fields": {
        "lora_tag": "actual_lora_filename",
        "min_weight": 0.5,
        "max_weight": 1.0,
        "const_tags": { "type": "list", "items": ["tag1", "tag2"] },
        "const_variations": {
          "type": "list",
          "items": ["variation1", "variation2"]
        },
        "const_banned_key_tags": { "type": "list", "items": ["banned_tag"] }
      }
    }
  ],
  "tag_lora_map": {
    "trigger_word": "lora_instance_name"
  },
  "compatibility_map": {
    "lora1": ["compatible_lora2", "compatible_lora3"]
  }
}
```

### LoRA/LyCORIS Classes

#### MainLora

Primary LoRA/LyCORIS models that define the main style or theme:

- Higher selection probability
- Can be used as base for other LoRA/LyCORIS models
- Typically style-defining (anime, realistic, etc.)

#### SecondaryLora

Supporting LoRA/LyCORIS models that complement main models:

- Lower selection probability
- Usually character-specific or detail-enhancing
- Compatible with multiple main LoRA/LyCORIS models

#### TagLora

LoRA/LyCORIS models triggered by specific tags in the prompt:

- Activated when certain tags are present
- Mapped via `tag_lora_map`
- Conditional activation based on tag dictionary selections

#### ConditionalVariantLora

Advanced LoRA/LyCORIS models with complex activation conditions:

- Can have additional banned categories
- Support for reroll configurations
- Most flexible but complex setup

### Fields Configuration

#### Basic Fields

- **`lora_tag`**: The actual LoRA/LyCORIS filename/identifier (required)
- **`prefix_tag`**: Prefix in prompt format (default: "lora")
- **`min_weight`/`max_weight`**: Weight range for random selection (0.0-2.0)

#### Tag Collections

Tag collections use List Node format:

```json
"const_tags": {
  "type": "list",
  "items": ["tag1", "tag2", "tag3"]
}
```

- **`const_tags`**: Tags always added when this model is used
- **`const_variations`**: Variations always added with this model
- **`const_banned_key_tags`**: Tags categories incompatible when this model is being selected

### Reroll Configurations

Advanced LoRA/LyCORIS models can define reroll behaviors for dynamic tag/variation selection. Use an array of nodes (or wrap the array in a list node):

```json
"reroll_tags": [
  { "type": "choice",  "items": ["option1", "option2", "option3"] },
  { "type": "choices", "items": ["multi1", "multi2", "multi3", "multi4"] },
  { "type": "list",    "items": ["always1", "always2"] }
]
```

- **`choice`**: Selects exactly one item from the group
- **`choices`**: Selects a random non-empty subset from the group
- **`list`**: Always includes all items from the group

### Tag LoRA/LyCORIS Mapping

Maps trigger words to TagLora instances:

```json
"tag_lora_map": {
  "cat ears": "cat_character_lora",
  "mecha": "mecha_style_lora",
  "panoramic sunset": "panoramic_sunset_lora"
}
```

When the generator selects "cat ears" from the tag dictionary, it automatically includes the "cat_character_lora" LoRA/LyCORIS model.

### LoRA/LyCORIS Compatibility

Defines which MainLora and SecondaryLoras can be used together:

```json
"compatibility_map": {
  "anime_style_v1": ["character_lora_v2", "outfit_lora_v1"],
  "realistic_style": ["lighting_lora", "detail_enhancer"]
}
```

The generator uses this to:

- Enforce compatible LoRA/LyCORIS combinations
- Avoid conflicting LoRA/LyCORIS models
- Build coherent style combinations

### Example Configuration

```json
{
  "instances": [
    {
      "name": "anime_base",
      "class": "MainLora",
      "fields": {
        "lora_tag": "anime_style_v3",
        "min_weight": 0.7,
        "max_weight": 1.0,
        "const_tags": { "type": "list", "items": ["anime", "cel shading"] },
        "const_variations": {
          "type": "list",
          "items": ["vibrant colors", "clean lines"]
        }
      }
    },
    {
      "name": "cat_character",
      "class": "TagLora",
      "fields": {
        "lora_tag": "cat_ears_v2",
        "min_weight": 0.5,
        "max_weight": 0.8,
        "const_tags": { "type": "list", "items": ["cat ears", "tail"] },
        "const_banned_key_tags": {
          "type": "list",
          "items": ["robot", "mechanical"]
        }
      }
    }
  ],
  "tag_lora_map": {
    "cat ears": "cat_character"
  },
  "compatibility_map": {
    "anime_base": ["cat_character"]
  }
}
```

## Generator Configuration (`generator_config.json`)

**Purpose**: Technical settings for prompt generation and Stable Diffusion parameters.

### Structure

```json
{
  "samplers": ["Euler a", "DPM++ 2M Karras"],
  "upscalers": ["R-ESRGAN 4x+", "None"],
  "width": 512,
  "height": 512,
  "n_iter": 8,
  "batch_size": 1,
  "hr_scale": 2,
  "prompt_extra_settings": "--hr_resize_x 1024 --hr_resize_y 1024",
  "positive_prompt": "masterpiece, best quality",
  "negative_prompt": "lowres, bad anatomy",
  "vanilla_extra_tags": ["focus"]
}
```

### Required Fields

#### Samplers

List of sampling methods available for selection:

```json
"samplers": [
  "Euler a",
  "DPM++ 2M Karras",
  "DPM++ SDE Karras",
  "DDIM",
  "LMS"
]
```

The generator randomly selects from this list for each prompt. Must contain at least one sampler and match the tokenizer's samplers.

### Optional Fields

#### Upscalers

Available upscaling methods:

```json
"upscalers": [
  "R-ESRGAN 4x+",
  "R-ESRGAN 4x+ Anime6B",
  "SwinIR_4x",
  "None"
]
```

Defaults to `["None"]` if not specified.

#### Image Dimensions

- **`width`**: Image width in pixels (default: 512, must be multiple of 8)
- **`height`**: Image height in pixels (default: 512, must be multiple of 8)

#### Generation Parameters

- **`n_iter`**: Number of iterations/images (default: 8)
- **`batch_size`**: Batch size for generation (default: 1)
- **`hr_scale`**: High-resolution scale factor (default: 2). Accepts positive integers or floating point values greater than 0.

#### Prompt Templates

- **`positive_prompt`**: Base positive prompt prepended to all generated prompts
- **`negative_prompt`**: Base negative prompt used for all generations

#### Prompt Extra Settings

Optional string appended verbatim to every generated CLI prompt. Useful for Stable Diffusion WebUI flags such as high-resolution resize dimensions.

```json
"prompt_extra_settings": "--hr_resize_x 1024 --hr_resize_y 1024"
```

#### Vanilla Extra Tags

Optional list of tags to occasionally append to "vanilla" prompts (prompts without any selected LoRA/LyCORIS models or tag-triggered models).

- If configured, the generator will, with a 50% chance, select one tag at random from this list and append it to vanilla prompts.
- If omitted or empty, nothing extra is added.

Example:

```json
"vanilla_extra_tags": ["focus", "portrait", "wide shot"]
```

### CFG Scale Generation

The generator automatically creates CFG scale values ranging from 3.0 to 11.0 in 0.5 increments. This cannot be configured via JSON.

### Output Format Integration

The generator formats all prompts for direct use with Stable Diffusion WebUI's "prompts from file or textbox" script, including:

- Proper LoRA/LyCORIS syntax: `<lora:filename:weight>`
- Sampler and step specifications
- Dimension and scaling parameters
- Positive and negative prompt separation

### Example Configuration

```json
{
  "samplers": ["Euler a", "DPM++ 2M Karras", "DPM++ SDE Karras", "DDIM"],
  "upscalers": ["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B", "None"],
  "width": 768,
  "height": 768,
  "n_iter": 12,
  "batch_size": 2,
  "hr_scale": 2,
  "positive_prompt": "masterpiece, best quality, high resolution, detailed",
  "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
}
```

## Configuration Validation

### Schema Validation

All configuration files are validated against JSON schemas:

- [`dict.schema.json`](schemas/dict.schema.json) - Tag dictionary validation
- [`loras.schema.json`](schemas/loras.schema.json) - LoRA/LyCORIS configuration validation
- [`generator_config.schema.json`](schemas/generator_config.schema.json) - Generator settings validation

### Runtime Validation

The generator performs additional validation:

- **Cross-references**: Ensures LoRA/LyCORIS compatibility references exist
- **Tag mappings**: Validates tag_lora_map references point to valid TagLora instances
- **Category limits**: Enforces 64-category limit for bitmask optimization
- **Weight ranges**: Ensures LoRA/LyCORIS weights are within valid bounds (0.0-2.0)

### Error Handling

Configuration errors result in:

- Clear error messages indicating the problem location
- Graceful fallbacks where possible (e.g., default upscalers)
- Application termination for critical configuration issues

## Best Practices

### Tag Dictionary Design

- **Logical grouping**: Group related tags into meaningful categories
- **Balanced categories**: Avoid categories with too many or too few tags
- **Clear naming**: Use descriptive category names that reflect their purpose
- **Rule efficiency**: Use exclusive groups for mutual exclusions rather than individual incompatible_categories

### LoRA/LyCORIS Organization

- **Consistent naming**: Use clear, descriptive names for LoRA/LyCORIS instances
- **Weight tuning**: Test weight ranges to find optimal values for each model
- **Compatibility mapping**: Define compatibility relationships to improve generation quality
- **Tag coordination**: Ensure LoRA/LyCORIS tags align with tag dictionary categories

### Validation Tools

Use JSON schema validators to check configuration files before running:

```bash
# Example using ajv-cli
ajv validate -s docs/schemas/dict.schema.json -d your_dict.json
ajv validate -s docs/schemas/loras.schema.json -d your_loras.json
ajv validate -s docs/schemas/generator_config.schema.json -d your_config.json
```
