# auto_optimize

This is an experiment to have the LLM autonomously optimize the inference pipeline.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `auto_optimize/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b auto_optimize/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context on the inference pipeline:
   - `README.md` — repository context.
   - `sorawm/core.py` — the main pipeline orchestrator: video loading, detection, cleaning dispatch, FFmpeg encoding. **Modifiable.**
   - `sorawm/cleaner/lama_cleaner.py` — LAMA cleaner wrapper: per-frame mask generation + IOPaint model call. **Modifiable.**
   - `sorawm/cleaner/e2fgvi_hq_cleaner.py` — E2FGVI-HQ cleaner: chunked video inpainting with temporal consistency. **Modifiable.**
   - `sorawm/models/model/e2fgvi_hq.py` — E2FGVI-HQ model architecture (InpaintGenerator). **Modifiable.**
   - `sorawm/watermark_detector.py` — YOLO watermark detection. **Modifiable.**
   - `sorawm/utils/video_utils.py` — VideoLoader, frame merging utilities. **Modifiable.**
   - `sorawm/utils/mem_utils.py` — VRAM profiling. **Modifiable.**
   - `sorawm/constants.py` — constants like `CHUNK_SIZE_PER_GB_VRAM`. **Modifiable.**
   - `sorawm/configs.py` — paths, default settings. **Modifiable.**
   - `benchmark/benchmark_batch.py` — benchmark script for measuring end-to-end time. **Modifiable.**
4. **What you CANNOT modify**:
   - `sorawm/iopaint/` — the IOPaint submodule (LAMA model internals). It is third-party code.
   - Model weight files (`.pt`, `.pth`). Do not retrain or fine-tune.
   - `pyproject.toml` — do not add or remove dependencies. Use only what's already available.
5. **Prepare the benchmark video**: Check that `resources/dog_vs_sam.mp4` exists (this is the standard benchmark video). If not, tell the human to provide one.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The Pipeline

The full inference pipeline has these stages, in order:

```
Input Video
    ↓
1. VideoLoader (FFmpeg pipe → BGR frames)
    ↓
2. YOLO Batch Detection (detect_batch_size=4)
    ↓
3. Missed-frame Imputation (ruptures + interval averaging)
    ↓
4. Watermark Cleaning:
   ├─ LAMA: per-frame mask → LaMa(image, mask) → output
   └─ E2FGVI_HQ: segment chunking → InpaintGenerator → overlap blending
    ↓
5. FFmpeg Encoding (H.264) + Audio Merge
    ↓
Output Video
```

**The bottleneck is stage 4** — LAMA and E2FGVI-HQ model inference dominate total processing time. But all stages are fair game for optimization.

## Experimentation

Each experiment processes the benchmark video end-to-end. You launch it as:

```bash
uv run python benchmark/benchmark_batch.py > run.log 2>&1
```

The benchmark script uses `perf_counter` and prints timing like:

```
[E2FGVI_HQ + torch.compile + batch + bf16] Time elapsed: 58.60s
```

You can extract timing from the log:

```bash
grep "Time elapsed" run.log
```

**What you CAN do:**
- Modify any file listed as **Modifiable** above. Everything is fair game: tensor operations, data layout, batch strategies, memory management, padding logic, reference frame selection, chunk sizing, FFmpeg pipe configuration, numpy↔torch conversions, async I/O, etc.
- Modify or extend `benchmark/benchmark_batch.py` to add more granular timing, but keep it measuring the same end-to-end `SoraWM.run()` call.

**What you CANNOT do:**
- Modify `sorawm/iopaint/` (third-party LAMA internals). Wrap, batch, or call it differently — but don't edit the source.
- Modify or replace model weight files. The architecture changes must remain compatible with existing checkpoints.
- Install new packages or add dependencies. Only use what's in `pyproject.toml`.
- Sacrifice output quality for speed in ways that produce visible artifacts. Minor, imperceptible quality changes are acceptable for meaningful speed gains.

**The goal is simple: minimize end-to-end processing time for the benchmark video.** Both LAMA and E2FGVI-HQ paths should be optimized. VRAM usage is a soft constraint — some increase is acceptable for meaningful speed gains, but don't cause OOM on typical GPUs (24GB).

**Simplicity criterion**: All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Removing unnecessary work and getting equal or better speed is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the speedup magnitude. A 0.5s improvement that adds 50 lines of hacky code? Probably not worth it. A 0.5s improvement from deleting code? Definitely keep. Same speed but cleaner code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the benchmark as-is.

## Optimization Ideas (non-exhaustive)

Here are some promising directions. You are not limited to these — use your judgment:

- **Tensor operations**: Avoid redundant numpy↔torch conversions, keep data on GPU longer, use in-place ops.
- **Padding**: The E2FGVI model pads to `(mod 60, mod 108)` via flip+cat — can this be done more efficiently (e.g. `F.pad`)?
- **Reference frame selection**: `get_ref_index` uses a naive loop with `not in` checks on a list. Could use sets or vectorized selection.
- **Batch LAMA inference**: Currently LAMA processes one frame at a time. Can multiple frames be batched?
- **Chunk size tuning**: `CHUNK_SIZE_PER_GB_VRAM=5` may not be optimal. Profile different values.
- **FFmpeg pipe**: The encoding preset is `slow` — `medium` or `fast` trades quality for encode speed. `bgr24` pipe format may not be optimal.
- **Memory management**: `torch.cuda.empty_cache()` between chunks may not be needed, or timing could be improved.
- **Data loading**: `VideoLoader.get_slice()` re-reads from the start each time. Can we cache or seek more efficiently?
- **Async I/O**: Overlap FFmpeg encoding with model inference (pipe writes while GPU computes next chunk).
- **torch.compile**: Experiment with different compile modes (`reduce-overhead`, `max-autotune`) and backends.
- **Mixed precision**: Extend bf16 to more operations, or try fp16 where safe.
- **Neighbor stride / ref_length**: Larger `neighbor_stride` means fewer model calls per chunk. Trade-off with quality.

## Output format

The benchmark script prints timing for each configuration. Example:

```
[LAMA] Time elapsed: 44.33s
[E2FGVI_HQ + torch.compile + batch + bf16] Time elapsed: 58.60s
```

You can also add more granular profiling (detection time, cleaning time, encoding time) if it helps diagnose bottlenecks.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	lama_time_s	e2fgvi_time_s	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. lama_time_s: LAMA path total time in seconds (e.g. 44.33) — use 0.00 for crashes or if not tested
3. e2fgvi_time_s: E2FGVI path total time in seconds (e.g. 58.60) — use 0.00 for crashes or if not tested
4. peak_vram_gb: peak VRAM in GB, round to .1f (e.g. 12.3) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	lama_time_s	e2fgvi_time_s	peak_vram_gb	status	description
a1b2c3d	44.33	58.60	12.3	keep	baseline
b2c3d4e	38.10	58.60	12.5	keep	batch LAMA inference (batch=4)
c3d4e5f	44.33	52.10	13.1	keep	replace flip+cat padding with F.pad
d4e5f6g	0.00	0.00	0.0	crash	async FFmpeg pipe (race condition)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `auto_optimize/mar19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Identify the next optimization idea. Prioritize changes that target the biggest bottleneck.
3. Implement the optimization by editing the relevant source files.
4. git commit with a descriptive message.
5. Run the benchmark: `uv run python benchmark/benchmark_batch.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
6. Read the results: `grep "Time elapsed" run.log`
7. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't fix it after a few attempts, give up on this idea.
8. Record results in the tsv (NOTE: do not commit results.tsv, leave it untracked by git).
9. If processing time improved (lower), you "advance" the branch, keeping the git commit.
10. If processing time is equal or worse, you `git reset --hard` back to where you started.

The idea is that you are a completely autonomous performance engineer trying things out. If they work, keep. If they don't, discard. You're advancing the branch so you can iterate. If you feel stuck, try a different stage of the pipeline, combine near-misses, or try more radical approaches.

**Timeout**: Each benchmark run should complete within ~5 minutes for LAMA and ~10 minutes for E2FGVI. If a run exceeds 15 minutes total, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: If it's a typo or easy fix, fix and re-run. If the optimization approach itself is fundamentally broken, log "crash", revert, and move on.

**Quality check**: After a speed improvement, quickly sanity-check the output video exists and has reasonable file size. If the output is visibly broken (0-byte file, solid color frames), treat it as a crash.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the source files for inefficiencies, profile individual functions, try combining previous improvements, try more radical restructuring. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running overnight. If each experiment takes ~5 minutes then you can run ~12/hour, for a total of about 100 over an 8-hour sleep. The user wakes up to a history of optimization experiments with measured speedups, all completed by you while they slept!
