# Auto Triton Optimize

This is a task to have the agent find the bottleneck during the `E2FGVIHDCleaner`'s inference, then write Triton ops to reduce latency.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b <tag>` from current master.
3. **Read the in-scope files**: The repo is relatively large, but you only need to focus on the `profile` dir. Below is a brief description of every file in that directory:

   | File | Type | Description |
   |------|------|-------------|
   | `profile_clean.sh` | Shell | Nsys profiling launcher for `run_clean.py`. Traces CUDA, cuBLAS, NVTX, OS runtime, and cuDNN events. |
   | `profile_whole_infer.sh` | Shell | Nsys profiling launcher for `run_whole.py`. Same trace flags as above. |
   | `profile_process_chunk.sh` | Shell | Nsys profiling launcher for `run_process_chunk.py`. Same trace flags as above. |
   | `run_clean.py` | Python | **End-to-end pipeline profiler.** Subclasses `SoraWM` as `ProfileSoraWM`, wrapping every major stage (video loading, watermark detection, breakpoint segmentation, per-segment cleaning, ffmpeg encoding, audio merge) with NVTX ranges. Runs on a real video file (`resources/dog_vs_sam.mp4`). |
   | `run_process_chunk.py` | Python | **Chunk-level cleaner profiler.** Subclasses `E2FGVIHDCleaner` as `ProfileE2FGVIHDCleaner`, adding NVTX ranges around chunk setup, tensor conversion, per-chunk `process_frames_chunk` calls, frame merging, and GPU cleanup. Loads pre-saved numpy arrays (`frames.npy`, `masks.npy`) so it skips detection and isolates the cleaning stage. |
   | `run_whole.py` | Python | **Model-level profiler.** Subclasses both `InpaintGenerator` (as `ProfileInpaintGenerator`) and `E2FGVIHDCleaner` (as `ProfileE2FGVIHDCleaner`). Adds fine-grained NVTX ranges inside the model's `forward()` and `forward_bidirect_flow()` (encoder, SpyNet, transformer blocks, decoder, etc.) as well as around the cleaner's sliding-window `process_frames_chunk` loop (neighbor/ref selection, masking, padding, model inference, D2H transfer, compositing). Also loads from numpy arrays. Contains a `raise RuntimeError("Stop here")` breakpoint in `process_frames_chunk` to cap profiling length. |

4. **Update shell scripts to export CSV stats**: The `.nsys-rep` files nsys generates are binary — the LLM cannot read them. Every profiling shell script must append `nsys stats` commands to export human-readable CSV summaries. Add the following block to the end of each `.sh` script:

   ```bash
   # Export LLM-readable CSV summaries
   nsys stats --report nvtx_sum      ${REPORT}.nsys-rep --format csv -o ${REPORT}_nvtx_sum
   nsys stats --report cuda_kern_sum ${REPORT}.nsys-rep --format csv -o ${REPORT}_cuda_kern_sum
   nsys stats --report cuda_api_sum  ${REPORT}.nsys-rep --format csv -o ${REPORT}_cuda_api_sum
   ```

   This produces three CSV files per run:

   | CSV file | What it tells you |
   |----------|-------------------|
   | `*_nvtx_sum.csv` | Per-stage wall time from NVTX annotations — maps directly to code sections |
   | `*_cuda_kern_sum.csv` | GPU kernel ranking by total time — shows which operator is the bottleneck |
   | `*_cuda_api_sum.csv` | CUDA API call breakdown (cudaMemcpy, cudaLaunchKernel, etc.) — reveals host-side overhead |

5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. You can modify every script under the `profile` dir to adjust the profiling process and the records for nsys.

What you need to do:

1. Use nsys to profile and identify the latency bottleneck.
2. Read the exported CSV files (`cat profiling/*_nvtx_sum.csv` and `cat profiling/*_cuda_kern_sum.csv`) to locate the slowest stages and kernels.
3. Write Triton ops or apply other optimization strategies to address the bottleneck.
4. **Validate correctness**: after each optimization, compute the error diff between the optimized output and the baseline output (e.g. mean absolute error or max pixel diff on a representative frame). The diff must remain reasonable — if it blows up, the optimization is invalid.
5. Re-profile to measure the latency improvement.
6. Record the results.
7. Each experiment should not exceed 2 minutes.

**What you CAN do:**

- Modify any file under the `profile` directory — profiling scripts, runner scripts, and any Triton kernel code you add.
- Add new Triton kernels or optimization wrappers targeting identified bottlenecks.
- Adjust NVTX annotations, profiling granularity, and nsys trace flags.
- Focus heavily on writing Triton ops — fused kernels, custom attention, fused normalization, fused activation, etc.

**What you CANNOT do:**

- Modify the core model architecture outside of the `profile` directory (the upstream `E2FGVIHDCleaner` and `InpaintGenerator` source files are read-only for correctness).
- Install new packages or add dependencies beyond what's already available.
- Change the evaluation or correctness criteria — the cleaned output must remain visually equivalent (error diff must stay reasonable).

**The goal is simple: identify the inference latency bottleneck and reduce it via Triton ops.** Use nsys profiling to pinpoint the slowest stages, then write Triton kernels or apply other GPU optimization strategies (operator fusion, memory layout changes, kernel tuning, etc.) to speed them up. Triton ops are the primary tool — prefer writing custom Triton kernels over other approaches.

**The first run**: Your very first run should always be to establish the baseline profile, so you will run the profiling script as-is.

### Reading profiling results

The `.nsys-rep` binary is not readable by the LLM. Instead, read the exported CSV files:

```bash
# Stage-level breakdown (from your NVTX annotations)
cat profiling/profile_process_chunk_nvtx_sum.csv

# GPU kernel hotspot ranking
cat profiling/profile_process_chunk_cuda_kern_sum.csv

# Host-side CUDA API overhead
cat profiling/profile_process_chunk_cuda_api_sum.csv
```

The typical workflow is: start with `nvtx_sum` to find the slowest stage, then drill into `cuda_kern_sum` to see which specific kernel dominates that stage.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	inference_ms	status	description
```

1. git commit hash (short, 7 chars)
2. total inference time in ms (e.g. 4523.1) — use 0.0 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	inference_ms	status	description
a1b2c3d	4523.1	keep	baseline profiling
b2c3d4e	3891.2	keep	fused softmax+dropout in transformer via Triton
c3d4e5f	4600.0	discard	custom Triton conv kernel (slower than cuDNN)
d4e5f6g	0.0	crash	broken Triton grid config
e5f6g7h	3750.0	keep	fused gelu+bias via Triton (error diff < 1e-5)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Identify the current bottleneck by reading the CSV stats from the latest profile:
   - `cat profiling/*_nvtx_sum.csv` — which stage is slowest?
   - `cat profiling/*_cuda_kern_sum.csv` — which GPU kernel dominates?
   - `cat profiling/*_cuda_api_sum.csv` — any host-side overhead (e.g. excessive cudaMemcpy)?
3. Write a Triton kernel or apply an optimization targeting that bottleneck. **Prefer Triton ops** — the primary goal is to replace slow PyTorch/cuDNN ops with hand-written Triton kernels.
4. **Validate error diff**: run a quick check to ensure the optimized output is numerically close to baseline (e.g. mean absolute error on output frames). If the diff is unreasonable, fix or discard.
5. git commit.
6. Run the experiment: `bash profile/profile_whole_infer.sh > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
7. Read out the results: `cat profiling/*_nvtx_sum.csv` to check if the targeted stage improved. If the CSV files are missing or empty, the run crashed — run `tail -n 50 run.log` to read the stack trace and attempt a fix.
8. Record the results in the TSV (NOTE: do not commit the results.tsv file, leave it untracked by git).
9. If inference time improved (lower) and error diff is reasonable, you "advance" the branch, keeping the git commit.
10. If inference time is equal or worse, or error diff is too large, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very sparingly (if ever).

**Timeout**: Each experiment should take ~2 minutes total (+ a few seconds for startup overhead). If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**Error diff**: After each optimization, verify the output hasn't diverged. A small numerical diff (e.g. max abs error < 1e-3 for float32 outputs) is expected and acceptable. If the error is large, the optimization is invalid — discard it.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the nsys traces for new angles, try combining previous near-misses, try more aggressive fusion strategies, explore memory layout optimizations, write more Triton kernels. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~2 minutes then you can run approx 30/hour, for a total of about 240 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!