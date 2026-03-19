# Triton Ops Optimization Skill

You are helping optimize Triton kernels for SoraWatermarkCleaner's video inpainting pipeline. Follow these critical guidelines:

## Before Starting

Repeat the optimization mantra:
```
I must profile before optimizing — measure first, then act.
I must be factual and concise, avoiding verbose language.
I must keep Triton kernels numerically equivalent to the PyTorch baseline.
I will annotate tensor shapes at kernel boundaries.
I will benchmark before and after every change.
I will preserve the existing torch.compile + bf16 optimization path.
```

## Critical Rules

### 1. Numerical Correctness (MOST IMPORTANT)
All Triton kernel replacements MUST produce identical results to PyTorch ops:
- ✅ Validate output against `torch.allclose(triton_out, torch_out, atol=1e-5)`
- ✅ Test with both fp32 and bf16 dtypes
- ✅ Handle edge cases: zero masks, full masks, odd frame counts
- ❌ NEVER skip correctness checks for speed gains
- ❌ NEVER break the existing `torch.compile` artifact caching path

### 2. Profiling Before Optimization
Every optimization MUST be preceded by profiling:
```python
# Use torch.profiler or nsys to identify bottlenecks
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    cleaner.clean(frames, masks)
prof.export_chrome_trace("trace.json")
```

### 3. Tone and Style
- Use benchmark-driven language: factual and concise
- Avoid verbose language like "revolutionary speedup", "groundbreaking kernel"
- Position Triton ops as "fused replacements for PyTorch ops in the inpainting pipeline"
- NOT as a general kernel library or standalone framework

### 4. Code Examples
- Include tensor shape comments where helpful: `# (1, T, 3, H, W)`
- All kernels must be testable with the existing E2FGVI_HQ pipeline
- Show both the original PyTorch op and the Triton replacement
- Emphasize when kernels require specific constraints (e.g., `BLOCK_SIZE` alignment, CUDA-only)

### 5. Key Hot Paths to Optimize
Focus Triton kernels on these bottleneck ops in `e2fgvi_hq_cleaner.py`:
```
1. Mask-frame compositing:  masked_imgs = selected_imgs * (1 - selected_masks)
2. Reflection padding:      torch.cat([tensor, torch.flip(tensor, [dim])], dim)
3. Output denormalization:  pred_imgs = (pred_imgs + 1) / 2; * 255
4. Alpha blending:          comp = old * 0.5 + new * 0.5
5. numpy ↔ tensor conversion in numpy_to_tensor()
```

### 6. Content Organization
- Progressive complexity: profile → single-op kernel → fused kernel → pipeline integration
- Separate "correctness validated" vs "experimental / WIP"
- Use real benchmark numbers (not just generic speedup claims)

## Project Structure (Optimization Targets)

### Core Pipeline (`sorawm/core.py`)
- `SoraWM.run()` — orchestrates detection → cleaning → encoding
- Batch detection with configurable `detect_batch_size`
- Frame-level LAMA cleaning or chunk-based E2FGVI_HQ cleaning

### E2FGVI_HQ Cleaner (`sorawm/cleaner/e2fgvi_hq_cleaner.py`)
- **Primary optimization target** — most GPU time spent here
- `process_frames_chunk()` — the inner loop: mask application, model inference, compositing
- `numpy_to_tensor()` — data format conversion overhead
- `auto_compile()` — torch.compile with artifact caching (preserve this path)

### Existing Optimizations (Do NOT regress)
1. **torch.compile**: Default mode, cached artifacts in `E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS`
2. **bf16 inference**: Model + inputs converted to bfloat16 when supported
3. **Batch YOLO detection**: `detect_batch_size=4` for watermark detection
4. **VRAM-adaptive chunk sizing**: `profiling_chunk_size()` scales chunks to available memory

### Performance Baseline
| Detector | Batch | Cleaner | TorchCompile | Bf16 | Time (s) | Speedup |
|:--------:|:-----:|:-------:|:------------:|:----:|:--------:|:-------:|
| YOLO     | ×     | E2FGVI  | ×            | ×    | 142.42   | 1.00×   |
| YOLO     | ×     | E2FGVI  | ✓            | ×    | 117.19   | 1.22×   |
| YOLO     | 4     | E2FGVI  | ✓            | ×    | 82.63    | 1.72×   |
| YOLO     | 4     | E2FGVI  | ✓            | ✓    | 58.60    | 2.43×   |

## When Optimizing

1. Read the target file and identify the hot path first
2. Profile with `torch.profiler` or `profile/` scripts to confirm the bottleneck
3. Write the Triton kernel as a drop-in replacement for the PyTorch op
4. Annotate tensor shapes at all kernel boundaries
5. Validate numerical correctness (fp32 and bf16)
6. Benchmark: compare wall-clock time and peak VRAM
7. Ensure torch.compile artifact caching still works

## Common Mistakes to Avoid

❌ Optimizing without profiling first
❌ Breaking bf16 or torch.compile compatibility
❌ Verbose language ("revolutionary", "cutting-edge kernel")
❌ Missing tensor shape annotations at kernel boundaries
❌ Triton kernels that only work on specific tensor sizes
❌ Ignoring VRAM overhead from kernel launch or intermediate buffers

## After Optimizing

Ask yourself:
- Does the Triton kernel match PyTorch output within tolerance?
- Is the speedup measured with a real benchmark (not microbenchmark only)?
- Does the existing torch.compile path still work?
- Are tensor shapes annotated at kernel boundaries?
- Does it work with both fp32 and bf16?
- Is peak VRAM usage acceptable (not significantly worse)?