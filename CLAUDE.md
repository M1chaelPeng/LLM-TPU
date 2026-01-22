# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-TPU is a deployment framework for running Large Language Models (LLMs) and Vision-Language Models (VLMs) on Sophgo's BM1684X/BM1688 TPU chips. The project provides pre-compiled models and demo implementations for both Python and C++ environments.

**Tech Stack**: Python (primary), C++, TPU-MLIR compiler, PyTorch, Transformers

**Target Hardware**: BM1684X/BM1688 (CV186X) TPUs in SoC (Airbox) or PCIE environments

## Quick Start

```bash
# Run a model demo
./run.sh --model qwen3
./run.sh --model qwen2.5vl
./run.sh --model internvl3

# For PCIE environments, use the provided docker
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

## Model Compilation

Models are compiled using `llm_convert.py` (TPU-MLIR tool) from HuggingFace weights to bmodel format:

```bash
# Basic compilation
llm_convert.py -m /path/to/model -s 2048 -q w4f16 -c bm1684x -o output_dir

# Key parameters
# -m: model path (use AWQ/GPTQ quantized models when available)
# -s: sequence length
# -q: quantization (w4bf16, w4f16, bf16, f16)
# -c: chip target (bm1684x, bm1688, cv186x)
# -o: output directory

# Dynamic compilation (for variable input lengths)
llm_convert.py -m /path/to/model -s 8192 -q w4bf16 -c bm1684x --dynamic -o output_dir

# Prefill with KV cache (reduced latency)
llm_convert.py -m /path/to/model -s 8192 -q w4bf16 -c bm1684x --use_block_with_kv --max_input_length 512 -o output_dir

# Multi-device support
llm_convert.py -m /path/to/model -s 2048 -q w4bf16 -c bm1684x --num_device 2 -o output_dir
```

**Note**: Choose quantization type based on model's `config.json` `torch_dtype`:
- `bfloat16` → use `w4bf16`
- `float16` → use `w4f16`

## Codebase Architecture

### Directory Structure

```
models/[ModelName]/
├── config/                 # Model configuration and tokenizer files
├── python_demo/           # Python inference implementation
├── cpp_demo/              # C++ inference implementation
├── python_demo_parallel/  # Multi-device support
├── compile/               # Traditional compilation scripts (export_onnx.py + compile.sh)
├── eval_demo/             # Model evaluation implementations
└── run_demo.sh            # Model-specific demo runner

support/
├── include/               # TPU runtime API headers (bmruntime_interface.h)
└── lib/                   # Runtime libraries

harness/                   # Evaluation benchmarks (C-Eval, MMLU)
eval/                      # VLM accuracy testing tools
docs/                      # Documentation (Quick Start, MLIR Install, FAQ)
```

### Model-Specific Structure

Each model follows a consistent pattern:
- `python_demo/pipeline.py`: Main inference class using tokenizer + chat module
- `python_demo/CMakeLists.txt`: Build config for chat.cpython*.so
- `python_demo/chat.pyx`: Cython bindings to TPU runtime (compiled to .so)
- `config/`: Tokenizer and generation config from HuggingFace model

### Runtime Architecture

**Two-stage inference**:
1. `forward_first(tokens)`: Process input tokens and generate first output token
2. `forward_next()`: Autoregressively generate subsequent tokens

**Key interfaces** (from `support/include/bmruntime_interface.h`):
- `bmrt_create()`: Initialize runtime with device handle
- `bmrt_load_bmodel()`: Load compiled bmodel
- `bmrt_inference()`: Execute inference
- `bmrt_tensor_*()`: Tensor management utilities

### Advanced Features

**Dynamic Compilation**: Models adapt to actual input length, reducing latency for variable-length inputs. Supported by: Qwen3, Qwen2.5VL, MiniCPM4, InternVL3, Qwen3VL

**Prefill with KV Cache**: Historical context stored as KV cache instead of tokens, significantly reducing latency. Supported by: Qwen3VL, Qwen2.5VL, Qwen3, InternVL3

**Multi-core Inference**: Model parallelism across 2/4/6/8 devices for larger models or acceleration. See `Qwen2_5/python_demo_parallel/`

**Prompt Sharing**: Shared prefill cache for repeated prompts (system prompts). Divides model into prompt/prefill/decode stages.

**Random Sampling**: Support for temperature, top_p, top_k sampling via `--do_sample` flag

### Inference Pipeline

Typical flow in `pipeline.py`:
1. Initialize model with device IDs and bmodel path
2. Load tokenizer from config directory
3. Format input using `tokenizer.apply_chat_template()`
4. Call `model.forward_first(tokens)` for first token
5. Loop with `model.forward_next()` until EOS or max length
6. Decode tokens incrementally, handling multi-token characters (check for "�" in decoded output)

## Development Guidelines

### Adding New Models

Each model needs:
1. Compilation configuration (or use `llm_convert.py` for one-click compilation)
2. Python demo with `pipeline.py` inheriting model-specific class
3. C++ demo (optional but recommended)
4. Config directory with tokenizer files from HuggingFace

### Code Style

**C++**: Follow `.clang-tidy` configuration at project root
- camelBack naming for members, parameters, variables
- Modern C++ practices (make_unique, nullptr, override, etc.)
- Performance-oriented checks (inefficient copies, unnecessary value params)

### Common Issues

See `docs/FAQ.md` for 20+ common issues:
- **"!!!" output**: Precision/quantization issue, try switching W4BF16↔W4F16
- **Memory errors**: Check with `bm-smi`, use `memory_edit.sh` to adjust partitions
- **Driver version**: Must be ≥0.5.1 for PCIE
- **Emoji/encoding issues**: Handle multi-token characters in decode loop
- **Library mismatches**: Ensure torch==2.0.1+cpu, transformers version per model requirements

### Evaluation

```bash
# VLM accuracy testing
python eval/eval_qwen3vl.py --model_path bmodel_path --datasets A-OKVQA

# LLM benchmarking
python harness/C-Eval/evaluate_qwen2.py
```

## Key Commands Reference

| Task | Command |
|------|---------|
| Run model demo | `./run.sh --model <model_name>` |
| Compile model | `llm_convert.py -m <path> -s <len> -q <quant> -c <chip> -o <dir>` |
| Monitor TPU memory | `bm-smi` |
| Check bmodel info | `model_tool --info xxx.bmodel` |
| Adjust memory partitions | `./memory_edit.sh -c -npu 7168 -vpu 3072 -vpp 4096` |
| Build Python demo | `cd python_demo/build && cmake .. && make` |
| Build C++ demo | `mkdir build && cd build && cmake .. && make` |

## Supported Models

**40+ LLMs**: Qwen1.5/2/2.5/3, QwQ-32B, DeepSeek-R1-Distill-Qwen, Llama2/3, ChatGLM3/4, MiniCPM3/4, Phi-3/4, Gemma, Baichuan2, Mistral, Yi, WizardCoder, RWKV6/7

**VLMs**: Qwen3-VL, Qwen2.5-VL, Qwen2-VL, InternVL3, MiniCPM-V-2_6, Llama3.2-Vision, Molmo, Gemma3, NVILA, DeepSeek-Janus-Pro

See `models/` directory for complete list and model-specific documentation.

## Important Notes

- Always use `trust_remote_code=True` when loading tokenizers from custom models
- Quantization type (W4BF16 vs W4F16) must match model's `torch_dtype` in config.json
- For production deployment, prefer AWQ/GPTQ quantized models over floating-point
- Dynamic compilation recommended for VLMs with variable image sizes
- Prefill KV cache recommended for low-latency multi-turn conversations
- SoC environments (Airbox) can run pre-compiled models without TPU-MLIR installation
