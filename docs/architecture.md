# Architecture Documentation

## Overview

The `sdml-generator` is a high-throughput, GPU-accelerated Stable Diffusion prompt generator designed for quality-at-scale. It employs a multi-threaded pipeline architecture with lockstep scheduling, SPSC (Single Producer Single Consumer) queues, and deterministic task assignment to achieve ~450k prompts/s throughput on RTX 4070.

## System Architecture

The system follows a producer-consumer pipeline with three main stages:

```
[Generator Threads]     →     [Scoring Thread Pool] → [Filtering Thread] → [Output]
       ↓                              ↓                      ↓
Tag Dictionary                   ONNX Model            Quality Filter
LoRA/LyCORIS Controller          GPU Scoring           Diversity Logic
Tokenizer                        ORT Engine            Memory Management
```

## Core Components

### 1. Main Application (`main.cpp`)

**Purpose**: Entry point, configuration parsing, and orchestration of the entire pipeline.

**Key Responsibilities**:

- CLI argument parsing and validation
- Signal handling for graceful shutdown
- Component initialization and lifecycle management
- Final output writing with shuffling

**Architecture Pattern**:

- Uses RAII for resource management
- Signal-driven shutdown with atomic flags
- Single-shot output writing on termination

### 2. Generator Thread Pool (`generator/`)

**Purpose**: Parallel generation of candidate prompts from structured dictionaries.

#### Components:

**`GeneratorThread`** (`generator_thread.hpp/cpp`):

- Manages multiple generator worker threads
- Coordinates with scoring and filtering pipelines
- Handles thread-local component initialization

**`Generator`** (`generator.h/cpp`):

- Core prompt generation logic
- Composes prompts from tag dictionaries and LoRA/LyCORIS configurations
- Applies generation rules and constraints

#### Architecture Details:

- **Thread-local Components**: Each generator thread maintains its own instances of:
  - `Lora_controller`: LoRA/LyCORIS selection and compatibility logic
  - `Tag_list`: Tag sampling and category management
  - `Tokenizer`: Prompt tokenization and preprocessing
- **Lock-free Queues**: Uses `moodycamel::ReaderWriterQueue` for zero-contention communication
- **Deterministic Assignment**: Generator threads are assigned to scoring threads via modulo operation

### 3. Evaluation System (`evaluation/`)

**Purpose**: GPU-accelerated scoring and intelligent filtering of generated prompts.

#### Components:

**`ScoringEngine`** (`scoring_engine.hpp/cpp`):

- ONNX Runtime integration with GPU acceleration
- TensorRT optimization support
- Batch processing for throughput optimization

**Key Features**:

- **CUDA Stream Management**: Per-thread CUDA streams for parallel GPU execution
- **Device Buffer Management**: Pre-allocated GPU memory with dynamic resizing
- **FP16 Support**: Half-precision inference for memory efficiency
- **Input Tensor Management**: Handles 13 input tensors with fixed ordering

**`ScoringThreadPool`** (`scoring_thread.hpp/cpp`):

- Pool of scoring threads for parallel model inference
- Deterministic work assignment from generator threads
- Batch accumulation and processing

**`FilteringThread`** (`filtering_thread.hpp/cpp`):

- Advanced quality and diversity filtering
- LoRA/LyCORIS usage tracking and penalty system
- Memory-efficient prompt storage with stable IDs

#### Filtering Algorithm:

- **Score Thresholding**: Configurable minimum quality score
- **LoRA/LyCORIS Diversity**: Per-model caps with usage penalties
- **Replacement Logic**: Smart replacement of weaker prompts
- **Memory Management**: FIFO eviction with configurable limits

### 4. Data Models (`models/`)

**Purpose**: Structured data management for tags, LoRA/LyCORIS models, and generation rules.

#### Components:

**`Tag_dict`** (`tag_dict.h`):

- Hierarchical tag organization with path-based access
- Category-based conflict resolution (unique, exclusive groups)
- 64-bit mask system for fast conflict detection
- JSON-based configuration loading

**`Lora_controller`** (`Lora_controller.h`):

- LoRA/LyCORIS compatibility management
- Multiple LoRA/LyCORIS types: Main, Secondary, Conditional, Tag-based
- Compatibility matrix computation
- Dynamic LoRA/LyCORIS selection with constraints

**Architecture Patterns**:

- **Singleton Pattern**: Global tag dictionary instance
- **Factory Pattern**: JSON-based object construction
- **Compatibility Matrix**: Pre-computed compatibility relationships
- **Bit Masking**: Fast category conflict detection

### 5. Tokenization System (`tokenizer/`)

**Purpose**: Conversion of text prompts to numerical representations for model inference.

#### Components:

**`Tokenizer`** (`tokenizer.hpp/cpp`):

- Multi-vocabulary tokenization (words, LoRA/LyCORIS models, samplers, upscalers)
- Weight parsing for LoRA/LyCORIS tokens
- Caching for performance optimization

**Key Features**:

- **Multiple Vocabularies**: Separate token spaces for different prompt components
- **Weight Parsing**: Extraction of LoRA/LyCORIS weights from token strings
- **Split Caching**: Cached tokenization results for frequently used fragments
- **Bidirectional Mapping**: Token-to-ID and ID-to-token conversion

### 6. Utilities (`utils/`)

**Purpose**: High-performance utility components for the entire system.

#### Components:

**`types.hpp`**: Core data structures

- `Prompt`: Complete prompt representation with all metadata
- `PromptBatch`: Batch structure for GPU inference
- `ScoredPrompt`: Prompt with associated quality score

**`readerwriterqueue.h`**: Lock-free SPSC queue implementation

- Wait-free in common path
- Memory-efficient allocation strategy
- Cache-line alignment for performance

**`fast_rand.hpp`**: Thread-local random number generation

- PCG (Permuted Congruential Generator) for speed
- Thread-local instances for zero contention
- Specialized distributions for common use cases

**`atomicops.h`**: Low-level atomic operations

- Platform-specific optimizations
- Memory ordering guarantees

## Data Flow

### 1. Prompt Generation Flow

```
1. Generator Thread:
   - Sample from Tag_dict categories
   - Apply LoRA/LyCORIS selection rules
   - Generate CFG scale, sampler, upscaler settings
   - Create Prompt structure

2. Tokenization:
   - Convert text tokens to numerical IDs
   - Parse LoRA/LyCORIS weights
   - Create PromptBatch for GPU inference

3. Queue Assignment:
   - Deterministic assignment: generator_id % num_scoring_threads
   - Push to assigned SPSC queue
```

### 2. Scoring Flow

```
1. Scoring Thread:
   - Accumulate prompts into batches
   - Transfer batch to GPU memory
   - Execute ONNX model inference
   - Extract quality scores

2. Device Memory Management:
   - Pre-allocated GPU buffers
   - Dynamic resizing based on batch size
   - CUDA stream synchronization

3. Result Processing:
   - Create ScoredPrompt structures
   - Push to filtering thread queue
```

### 3. Filtering Flow

```
1. Quality Filtering:
   - Apply score threshold
   - Check LoRA/LyCORIS usage constraints
   - Calculate effective scores with penalties

2. Diversity Management:
   - Track LoRA/LyCORIS usage counts
   - Apply per-model caps
   - Select replacement candidates

3. Memory Management:
   - Stable ID allocation
   - FIFO eviction for memory limits
   - Lazy deletion in min-heaps
```

## Performance Optimizations

### 1. Concurrency Design

**Lock-free Communication**:

- SPSC queues between pipeline stages
- Thread-local random number generators
- Atomic flags for coordination

**Deterministic Assignment**:

- Modulo-based work distribution
- Eliminates load balancing overhead
- Predictable performance characteristics

### 2. Memory Management

**GPU Memory**:

- Pre-allocated device buffers
- Batch-oriented memory transfers
- CUDA stream parallelism

**CPU Memory**:

- Object pooling for prompt structures
- Stable ID system for efficient indexing
- Cache-friendly data structures

### 3. Algorithmic Optimizations

**Bit Masking**:

- 64-bit masks for category conflicts
- O(1) compatibility checking
- Pre-computed conflict matrices

**Batch Processing**:

- GPU inference batching
- Amortized model loading costs
- Vectorized operations where possible

## Configuration System

### 1. Tag Dictionary (`dict.json`)

```json
{
  "tags": {
    "category1": ["tag1", "tag2"],
    "category2": {
      "subcategory": ["tag3", "tag4"]
    }
  },
  "rules": {
    "unique_categories": ["category1"],
    "exclusive_groups": {
      "group1": ["category2", "category3"]
    }
  }
}
```

### 2. LoRA/LyCORIS Configuration (`loras.json`)

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

### 3. Generator Configuration (`generator_config.json`)

```json
{
  "samplers": ["Euler a", "DPM++ 2M Karras"],
  "upscalers": ["R-ESRGAN 4x+", "None"],
  "width": 512,
  "height": 512,
  "positive_prompt": "masterpiece, best quality",
  "negative_prompt": "lowres, bad anatomy"
}
```

## Error Handling and Reliability

### 1. Graceful Shutdown

- Signal handling (SIGINT, SIGTERM)
- Pipeline drainage before termination
- Final output writing on shutdown

### 2. Resource Management

- RAII for GPU resources
- Exception safety in critical paths
- Memory leak prevention

### 3. Error Recovery

- GPU memory allocation failures
- Model loading errors
- Configuration validation

## Scalability Considerations

### 1. Horizontal Scaling

- Configurable thread counts
- Independent scoring sessions
- Memory usage scaling

### 2. GPU Utilization

- Multiple CUDA streams
- Batch size optimization
- TensorRT engine caching

### 3. Memory Efficiency

- Configurable prompt limits
- Lazy deletion strategies
- Buffer reuse patterns

## Future Extensibility

### 1. Pluggable Components

- Scorer backend abstraction
- Generation strategy interfaces
- Output format modules

### 2. Model Support

- Multi-model scoring
- Dynamic model loading
