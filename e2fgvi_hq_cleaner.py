from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from sorawm.configs import (
    E2FGVI_HQ_CHECKPOINT_PATH,
    E2FGVI_HQ_CHECKPOINT_REMOTE_URL,
)
from sorawm.constants import CHUNK_SIZE_PER_GB_VRAM
from sorawm.models.model.e2fgvi_hq import InpaintGenerator
from sorawm.utils.devices_utils import get_device
from sorawm.utils.download_utils import ensure_model_downloaded
from sorawm.utils.mem_utils import clear_gpu_memory, memory_profiling
from sorawm.utils.video_utils import merge_frames_with_overlap


def get_ref_index(
    frame_idx: int, neighbor_ids: List[int], length: int, ref_length: int, num_ref: int
) -> List[int]:
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
        end_idx = min(length - 1, frame_idx + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) >= num_ref:
                    break
                ref_index.append(i)
    return ref_index


def numpy_to_tensor(frames_np, masks_np):
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0).float()
    frames_tensor = frames_tensor / 255.0 * 2 - 1

    masks_tensor = torch.from_numpy(masks_np).unsqueeze(1).unsqueeze(0).float()
    masks_tensor = masks_tensor / 255.0

    return frames_tensor, masks_tensor


device = get_device()
if device.type == "mps":
    logger.warning(
        "E2FGVI_HQ Cleaner does not support MPS for inference in this project. Falling back to CPU."
    )
    device = torch.device("cpu")


class E2FGVIHDConfig(BaseModel):
    ref_length: int = 10
    num_ref: int = -1
    neighbor_stride: int = 5
    chunk_size_ratio: float = 0.2
    overlap_ratio: float = 0.05
    enable_torch_compile: bool = False
    use_bf16: bool = False


class E2FGVIHDCleaner:
    def __init__(
        self,
        ckpt_path: Path = E2FGVI_HQ_CHECKPOINT_PATH,
        config: E2FGVIHDConfig = E2FGVIHDConfig(),
    ):
        ensure_model_downloaded(ckpt_path, E2FGVI_HQ_CHECKPOINT_REMOTE_URL)

        self.config = config
        self.device = device

        self.model = InpaintGenerator().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.use_bf16 = config.use_bf16 and self.device.type == "cuda"
        if self.use_bf16:
            logger.info("Enabling bf16 inference for E2FGVI_HQ cleaner")
            self.model = self.model.to(dtype=torch.bfloat16)

        self._artifacts_saved = False
        self.artifacts_path = None

        self.profiling_chunk_size()
        self.auto_compile()

    def auto_compile(self):
        self.artifacts_path = None
        logger.info("torch.compile disabled")
        return

    def profiling_chunk_size(self):
        if self.device.type != "cuda":
            self.adapted_chunk_size = 1
            logger.debug(
                f"Chunk size is set to {self.adapted_chunk_size} for device type '{self.device.type}'"
            )
            return

        try:
            memory_profiling_results = memory_profiling()
            adapted_chunk_size = int(
                memory_profiling_results.free_memory * CHUNK_SIZE_PER_GB_VRAM
            )
            self.adapted_chunk_size = max(1, adapted_chunk_size)
            logger.debug(
                f"Chunk size is set to {self.adapted_chunk_size} based on the free VRAM {round(memory_profiling_results.free_memory, 2)}GB"
            )
        except Exception as e:
            self.adapted_chunk_size = 1
            logger.warning(
                f"Memory profiling failed: {e}. Falling back to chunk size {self.adapted_chunk_size}"
            )

    @property
    def chunk_size(self):
        return self.adapted_chunk_size

    def process_frames_chunk(
        self,
        chunk_length: int,
        neighbor_stride: int,
        imgs_chunk: torch.Tensor,
        masks_chunk: torch.Tensor,
        binary_masks_chunk: np.ndarray,
        frames_np_chunk: np.ndarray,
        h: int,
        w: int,
    ) -> List[np.ndarray]:
        comp_frames_chunk = [None] * chunk_length

        for f in tqdm(
            range(0, chunk_length, neighbor_stride),
            desc="    Frame progress",
            position=2,
            leave=False,
        ):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(chunk_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f,
                neighbor_ids,
                chunk_length,
                self.config.ref_length,
                self.config.num_ref,
            )

            selected_imgs = imgs_chunk[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_chunk[:1, neighbor_ids + ref_ids, :, :, :]

            if self.use_bf16:
                selected_imgs = selected_imgs.to(dtype=torch.bfloat16)
                selected_masks = selected_masks.to(dtype=torch.bfloat16)

            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w

                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                    :, :, :, : h + h_pad, :
                ]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                    :, :, :, :, : w + w_pad
                ]

                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2

                if pred_imgs.dtype == torch.bfloat16:
                    pred_imgs = pred_imgs.float()

                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(np.uint8) * binary_masks_chunk[
                        idx
                    ] + frames_np_chunk[idx] * (1 - binary_masks_chunk[idx])

                    if comp_frames_chunk[idx] is None:
                        comp_frames_chunk[idx] = img
                    else:
                        comp_frames_chunk[idx] = (
                            comp_frames_chunk[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        ).astype(np.uint8)

        return comp_frames_chunk

    def clean(self, frames: np.ndarray, masks: np.ndarray) -> List[np.ndarray]:
        video_length = len(frames)

        chunk_size = int(self.config.chunk_size_ratio * video_length)
        overlap_size = int(self.config.overlap_ratio * video_length)

        chunk_size = max(1, min(chunk_size, video_length))
        overlap_size = max(0, min(overlap_size, chunk_size - 1))

        step = max(1, chunk_size - overlap_size)
        num_chunks = int(np.ceil(video_length / step))

        h, w = frames[0].shape[:2]
        imgs_all, masks_all = numpy_to_tensor(frames, masks)

        binary_masks = np.expand_dims(masks > 0, axis=-1).astype(np.uint8)
        comp_frames = [None] * video_length

        logger.debug(
            f"Processing {video_length} frames in {num_chunks} chunks (chunk_size={chunk_size}, overlap={overlap_size})"
        )

        for chunk_idx in tqdm(
            range(num_chunks), desc="  Chunk", position=1, leave=False
        ):
            start_idx = chunk_idx * step
            end_idx = min(start_idx + chunk_size, video_length)
            actual_chunk_size = end_idx - start_idx

            imgs_chunk = imgs_all[:, start_idx:end_idx, :, :, :].to(self.device)
            masks_chunk = masks_all[:, start_idx:end_idx, :, :, :].to(self.device)
            frames_np_chunk = frames[start_idx:end_idx]
            binary_masks_chunk = binary_masks[start_idx:end_idx]

            comp_frames_chunk = self.process_frames_chunk(
                actual_chunk_size,
                self.config.neighbor_stride,
                imgs_chunk,
                masks_chunk,
                binary_masks_chunk,
                frames_np_chunk,
                h,
                w,
            )

            comp_frames = merge_frames_with_overlap(
                result_frames=comp_frames,
                chunk_frames=comp_frames_chunk,
                start_idx=start_idx,
                overlap_size=overlap_size,
                is_first_chunk=(chunk_idx == 0),
            )

            del imgs_chunk, masks_chunk, comp_frames_chunk
            clear_gpu_memory()

        return comp_frames


if __name__ == "__main__":
    import os

    import cv2

    frames_path = Path("examples/extract_frame_and_mask_frames.npy")
    masks_path = Path("examples/extract_frame_and_mask_masks.npy")
    frames_np = np.load(frames_path)
    masks_np = np.load(masks_path)

    frames_np = frames_np[:, :, :, ::-1].copy()

    cleaner = E2FGVIHDCleaner()
    comp_frames = cleaner.clean(frames_np, masks_np)

    fps = 30
    output_video_path = "results/output.mp4"
    h, w = frames_np[0].shape[:2]

    os.makedirs("results", exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in comp_frames:
        writer.write(frame.astype(np.uint8)[:, :, ::-1])
    writer.release()
    logger.info(f"Video saved to: {output_video_path}")