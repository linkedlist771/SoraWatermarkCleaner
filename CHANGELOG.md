# Changelog

## [Unreleased] — auto_optimize/mar19 (2026-03-19)

### Performance — Inference Pipeline Optimization

End-to-end processing time for `resources/dog_vs_sam.mp4` (280 frames, 704×1280):

| Path | Before | After | Improvement |
|------|--------|-------|-------------|
| LAMA | 37.95s | 23.41s | **−38%** |
| E2FGVI-HQ (warm) | 58.33s | 39.51s | **−32%** |

#### LAMA Path

- **Batched LAMA inference** (`LamaCleaner.clean_batch`): process N frames through the LAMA JIT model simultaneously instead of one at a time. Batch=4 reduces per-frame time from 99 ms to 54 ms (1.8× speedup). Added efficient batch numpy ops — stack + transpose the full batch at once rather than calling `norm_img()` per frame.
- **Optimal YOLO detection batch size**: increased `DEFAULT_DETECT_BATCH_SIZE` from 4 → 16. Detection for 280 frames drops from 0.64s to 0.40s (throughput plateaus at batch=16 for 704×1280 inputs on RTX 4090).
- **Detection frame reuse**: frames accumulated during the YOLO detection pass are reused directly for cleaning, eliminating a second `VideoLoader` startup and full video re-read (~1.7s saved).

#### E2FGVI-HQ Path

- **MAX_ADAPTED_CHUNK_SIZE=50 cap** (`sorawm/constants.py`): when the GPU has large free VRAM (e.g. 22 GB → `adapted_chunk_size = 5 × 22 = 110`), the inner chunk size becomes 22 frames, causing `range(0, 22, 5)` to yield 5 model-forward calls per inner chunk instead of the optimal 2. Capping at 50 keeps `inner_chunk_size = 0.2 × 50 = 10` → `range(0, 10, 5) = [0, 5]` = exactly 2 calls.
- **Removed `torch.cuda.empty_cache()`** from the outer chunk loop: each call was adding ~0.7s of synchronization overhead with no memory benefit for the compiled model.
- **`torch.inference_mode()`** instead of `torch.no_grad()` in the model-forward inner loop.
- **Direct GPU tensor conversion**: frames and masks are moved to GPU in the target dtype (bfloat16) in a single `.to(device=…, dtype=…)` call, skipping the float32 intermediate on CPU.
- **Pre-computed padding**: `h_pad` / `w_pad` computed once per video instead of per inner chunk.
- **Set-based `get_ref_index`**: neighbor exclusion uses `set(neighbor_ids)` for O(1) lookup instead of O(n) list search.
- **FFmpeg preset**: `slow` → `medium` (encoding ~0.5s faster with negligible quality difference).

#### Experiments Tried and Discarded

| Experiment | Result | Reason discarded |
|---|---|---|
| `neighbor_stride=10` | 44s warm (worse) | Larger attention windows (up to 21 frames) increase quadratic cost more than halving call count helps |
| GPU compositing (accumulate on GPU, single CPU transfer per chunk) | 41s warm (worse) | The direct-GPU-tensor optimization already eliminates the CPU bottleneck it was targeting; extra GPU allocations add pressure |
| `torch.compile(mode='reduce-overhead')` | cold=287s, warm=39s | Cold compilation penalty (5 min) outweighs 2s warm gain |
| `torch.compile(mode='max-autotune')` | failed | Graph breaks in InpaintGenerator prevent autotuning |
| `torch.compile(dynamic=True)` | failed | Graph breaks |
| `torch.compile(fullgraph=True)` | failed | Graph breaks |
| fp32 inference | warm=65s | Much slower than bfloat16 (41s); bf16 is clearly optimal |
| Single outer chunk per segment | OOM | 7 GB needed but only 4.39 GB free |

#### Environment Note

The prior session (PyTorch ~2.5, Jan 2026) achieved E2FGVI warm=29.13s with the same algorithmic optimizations. With PyTorch 2.9.1 the floor is ~39.5s warm due to changes in torch.compile's kernel selection. `torch.compile` still provides a 48% speedup over eager (78s → 41s).

---

## Prior Optimizations (from earlier sessions)

- **torch.compile + bfloat16**: E2FGVI model compiled with `mode='default'` and converted to bf16. Saves ~17s vs eager fp32 execution.
- **Preload all frames once**: all video frames are loaded into a numpy array before segment processing, replacing per-segment `VideoLoader` calls.
- **F.pad 6-element tuple for 5D tensors**: correct padding for `(1, T, 3, H, W)` tensors using `(0, w_pad, 0, h_pad, 0, 0)` instead of 4-element form.
- **Chunk-level bfloat16 conversion**: dtype cast happens at the inner chunk level rather than per model-forward call.
