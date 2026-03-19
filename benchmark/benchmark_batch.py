from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from sorawm.core import SoraWM
from sorawm.schemas import CleanerType


@contextmanager
def timer(name: str):
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    print(f"[{name}] Time elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    input_video_path = Path("resources/dog_vs_sam.mp4")
    output_video_path = Path("outputs/sora_watermark_removed")

    # 1. LAMA baseline
    sora_wm = SoraWM(cleaner_type=CleanerType.LAMA, detect_batch_size=8)
    with timer("LAMA"):
        sora_wm.run(input_video_path, Path(f"{output_video_path}_lama.mp4"))

    # 2. E2FGVI_HQ best config: torch.compile + batch + bf16
    sora_wm = SoraWM(
        cleaner_type=CleanerType.E2FGVI_HQ,
        enable_torch_compile=True,
        detect_batch_size=8,
        use_bf16=True,
    )
    with timer("E2FGVI_HQ + torch.compile + batch + bf16"):
        sora_wm.run(
            input_video_path,
            Path(f"{output_video_path}_e2fgvi_hq_torch_compile_batch_bf16.mp4"),
        )
    # Run again to use cached compile artifacts
    with timer("E2FGVI_HQ + torch.compile + batch + bf16"):
        sora_wm.run(
            input_video_path,
            Path(f"{output_video_path}_e2fgvi_hq_torch_compile_batch_bf16.mp4"),
        )
