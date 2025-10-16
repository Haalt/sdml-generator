# SDML Generator

**High-throughput Stable Diffusion prompt generator with GPU-accelerated quality scoring**

Generates large batches of structured prompts, scores each one with a machine learning model, and outputs high-quality results ready for immediate use.

## 🎯 What This Does

A C++ application that generates **450,000 Stable Diffusion prompts per second** while scoring each one for predicted image quality using a trained neural network model.

### For Artists & Creators

- **Structured generation**: Combines tags from configurable categories (styles, subjects, lighting, etc.)
- **Quality scoring**: Each prompt gets a 0-1 score predicting image quality
- **Diversity control**: Prevents overuse of any single style or LoRA/LyCORIS model
- **WebUI integration**: Output format compatible with Stable Diffusion WebUI "prompts from file or textbox" native script

### For Technical Users

- **High throughput**: ~450k prompts/s on RTX 4070
- **GPU acceleration**: ONNX runtime with FP16 precision
- **Multi-threaded pipeline**: Lock-free queues and deterministic work assignment
- **Configurable filtering**: Adjustable quality thresholds and diversity constraints

### Key Features

- **JSON configuration**: Define your own tag dictionaries, LoRA/LyCORIS sets, and generation rules
- **ONNX model scoring**: Uses trained models from the sdml-prompt-quality project
- **Memory efficient**: Handles large prompt sets without excessive RAM usage
- **Cross-platform**: Linux primary, Windows experimental

## 🔬 How It Works

Three-stage pipeline processing:

```
Generate → Score → Filter → Output
```

### 1. **Generation**

- Samples tags from JSON-defined categories (characters, styles, lighting, etc.)
- Selects compatible LoRA/LyCORIS models with weight ranges
- Applies generation rules and constraints from configuration

### 2. **Quality Scoring**

- Tokenizes each prompt using the provided tokenizer
- Runs ONNX model inference on GPU to get quality scores (0-1 range)
- Batches prompts for efficient GPU utilization

### 3. **Filtering & Output**

- Applies configurable quality threshold
- Enforces diversity constraints (per-model usage limits)
- Outputs prompts in format compatible with Stable Diffusion WebUI "prompts from file or textbox" script

## 🚀 Performance

**Benchmark Results** (RTX 4070 + i7-9700K):

- **450,000 prompts/second** with quality scoring
- **125 LoRA/LyCORIS models** simultaneously managed
- **2,000+ vocabulary terms** in tag dictionary
- **9 generator threads, 2 scoring sessions**

The high throughput comes from GPU batch processing, lock-free queues, and deterministic work assignment.

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **CUDA Toolkit**: Version 11.x or 12.x
- **ONNX Runtime**: GPU-enabled version
- **Models**: Download from the [sdml-prompt-quality](https://github.com/Haalt/sdml-prompt-quality) project

### Installation & Setup

**Linux** (Primary platform):

```bash
# 1. Verify ONNX Runtime installation
export ONNXRUNTIME_ROOT=/opt/onnxruntime-gpu  # or your path
make check-onnx

# 2. Build the generator
make

# 3. Test the installation
./generator --help
```

**Windows** (Experimental):

- MSVC compilation supported
- TensorRT integration not yet verified on Windows

### Running Your First Generation

Use the included demo files to get started immediately:

```bash
./generator \
  --dict demo-assets/dict.demo.json \
  --loras demo-assets/loras.demo.json \
  --generator_config demo-assets/generator_config.demo.json \
  --model /path/to/your/model.onnx \
  --tokenizer /path/to/tokenizer.json \
  --output my_prompts.txt \
  --threshold 0.80 \
  --threads 4
```

**Operation:**

1. Generator creates and scores prompts continuously
2. Terminal shows throughput statistics
3. Stop with `Ctrl+C` when desired
4. Accepted prompts written to output file on shutdown

### Getting the AI Models

The quality scoring requires two files from [sdml-prompt-quality](https://github.com/Haalt/sdml-prompt-quality):

1. **ONNX Model**: Export your trained model

   ```bash
   sdpq export-onnx --checkpoint best_model.pt --out model.onnx --fp16
   ```

2. **Tokenizer**: Use the matching tokenizer JSON from your training

## ⚙️ Configuration

### Command Line Options

**Required Files:**

- `--dict <path>`: Tag dictionary with art categories and rules
- `--loras <path>`: LoRA/LyCORIS model definitions and compatibility
- `--generator_config <path>`: Samplers, upscalers, and formatting options
- `--model <path>`: ONNX quality scoring model
- `--tokenizer <path>`: Matching tokenizer for the scoring model

**Tuning Parameters:**

- `--output <path>`: Where to save generated prompts (default: `prompts.txt`)
- `--threshold <float>`: Minimum quality score to accept (default: `0.8`)
- `--threads <int>`: Number of generator threads (default: `4`)

### Configuration Files Explained

Each JSON configuration file controls a different aspect of generation:

**`dict.json`** - Your Creative Vocabulary:

```json
{
  "tags": {
    "characters": ["1girl", "1boy", "cat ears"],
    "styles": {
      "anime": ["anime", "manga style", "cel shading"],
      "realistic": ["photorealistic", "hyperrealistic", "detailed"]
    },
    "lighting": ["soft lighting", "dramatic shadows"]
  },
  "rules": {
    "unique_categories": ["characters"],
    "exclusive_groups": {
      "style_group": [
        ["styles.anime", "styles.manga"],
        ["styles.realistic", "styles.photorealistic"]
      ]
    }
  }
}
```

**Exclusive Groups Explained:**

- Each named group (e.g., `"style_group"`) contains multiple category arrays
- Only tags from **one array** within each group can be present in a single prompt
- If tags from `["styles.anime", "styles.manga"]` are selected, no tags from `["styles.realistic", "styles.photorealistic"]` can be added
- This ensures style consistency while allowing flexibility within each style category

**`loras.json`** - LoRA/LyCORIS Model Management:

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
        },
        "const_banned_key_tags": { "type": "list", "items": [] }
      }
    },
    {
      "name": "character_lora_v2",
      "class": "SecondaryLora",
      "fields": {
        "lora_tag": "character_lora_v2",
        "min_weight": 0.4,
        "max_weight": 0.8,
        "const_tags": { "type": "list", "items": ["detailed character"] },
        "const_variations": { "type": "list", "items": [] },
        "const_banned_key_tags": { "type": "list", "items": [] }
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
        "const_variations": { "type": "list", "items": [] },
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
    "anime_base": ["character_lora_v2"]
  }
}
```

**`generator_config.json`** - Technical Settings:

```json
{
  "samplers": ["Euler a", "DPM++ 2M Karras"],
  "upscalers": ["R-ESRGAN 4x+", "None"],
  "width": 512,
  "height": 512,
  "positive_prompt": "masterpiece, best quality",
  "negative_prompt": "lowres, bad anatomy",
  "vanilla_extra_tags": ["focus"]
}
```

For complete configuration documentation and schemas, see [`docs/configuration.md`](docs/configuration.md).

## 🎨 Use Cases

### Digital Artists

- Generate prompt variations for artwork series
- Explore different artistic style combinations systematically
- Test LoRA/LyCORIS model combinations at scale
- Create prompts for batch image generation

### AI Researchers

- Generate large prompt datasets for model training
- Analyze relationships between prompt structure and image quality
- Benchmark inference systems with realistic prompt distributions
- Create training data for prompt optimization research

### Content Creators

- Generate prompts matching specific content themes
- Create quality-controlled prompt sets for projects
- Compare different prompting strategies
- Integrate with existing creative workflows

## 🔧 Technical Architecture

**High-Level Pipeline:**

```
Generator Threads     →     Scoring Engine   →   Quality Filter   →   Output
      ↓                           ↓                     ↓
Tag Sampling                AI Evaluation        Diversity Logic
LoRA/LyCORIS Selection      GPU Inference        Memory Management
Rule Application            Batch Processing     Quality Control
```

**Technical Features:**

- **Lock-free Queues**: SPSC queues for zero-contention communication
- **Deterministic Scheduling**: Modulo-based work assignment
- **CUDA Stream Management**: Per-thread streams for parallel GPU execution
- **Multi-objective Filtering**: Quality thresholds with diversity constraints

For complete technical details, see [`docs/architecture.md`](docs/architecture.md).

## 🔗 Related Projects

This is part of a complete machine learning pipeline for Stable Diffusion optimization:

- **[SDML Specs](https://github.com/Haalt/sdml-specs)**: Schemas and specifications for the entire ecosystem
- **[SDML Image Classifier](https://github.com/Haalt/sdml-image-classifier)**: PyTorch-based image quality assessment
- **[SDML Prompt Quality](https://github.com/Haalt/sdml-prompt-quality)**: Neural network training for prompt scoring
- **[SDML Rec](https://github.com/Haalt/sdml-rec)**: Dataset building and management tools
