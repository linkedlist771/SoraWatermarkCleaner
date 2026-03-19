from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from sorawm.configs import (
    E2FGVI_HQ_CHECKPOINT_PATH,
    E2FGVI_HQ_CHECKPOINT_REMOTE_URL,
    E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS,
    E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS_BF16,
    ENABLE_E2FGVI_HQ_TORCH_COMPILE,
)
from sorawm.constants import CHUNK_SIZE_PER_GB_VRAM, MAX_ADAPTED_CHUNK_SIZE
from sorawm.models.model.e2fgvi_hq import InpaintGenerator
from sorawm.utils.devices_utils import get_device, is_bf16_supported
from sorawm.utils.download_utils import ensure_model_downloaded
from sorawm.utils.mem_utils import memory_profiling
from sorawm.utils.video_utils import merge_frames_with_overlap


def get_ref_index(
    frame_idx: int, neighbor_ids: List[int], length: int, ref_length: int, num_ref: int
) -> List[int]:
    """
    Compute reference frame indices to use for a given frame.

    Parameters:
        frame_idx (int): Index of the target frame around which references are selected.
        neighbor_ids (List[int]): Indices of neighbor frames that must be excluded from reference selection.
        length (int): Total number of frames in the sequence.
        ref_length (int): Step size (interval) between candidate reference frames.
        num_ref (int): Number of reference frames to select; if -1, select candidates across the entire sequence at intervals of `ref_length`.

    Returns:
        ref_index (List[int]): List of selected reference frame indices, spaced by `ref_length`, excluding any indices in `neighbor_ids`. When `num_ref` != -1, at most `num_ref` indices are returned and they are chosen from a window centered on `frame_idx`.
    """
    neighbor_set = set(neighbor_ids)
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_set:
                ref_index.append(i)
    else:
        start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
        end_idx = min(length, frame_idx + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_set:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


def numpy_to_tensor(frames_np, masks_np, target_device=None, target_dtype=torch.float32):
    """
    Convert numpy arrays to tensors, optionally directly to GPU with target dtype.
    Skips float32 intermediate when target_dtype is bfloat16.
    frames_np: (T, H, W, 3) uint8 [0, 255]
    masks_np: (T, H, W) uint8 [0, 255]
    Returns: frames tensor (1, T, 3, H, W) [-1, 1], masks tensor (1, T, 1, H, W) [0, 1]
    """
    # Frames: (T, H, W, 3) -> (T, 3, H, W) -> (1, T, 3, H, W)
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).unsqueeze(0)
    masks_tensor = torch.from_numpy(masks_np).unsqueeze(1).unsqueeze(0)

    if target_device is not None:
        # Move to GPU in target dtype, skipping float32 intermediate
        frames_tensor = frames_tensor.to(device=target_device, dtype=target_dtype)
        masks_tensor = masks_tensor.to(device=target_device, dtype=target_dtype)
    else:
        frames_tensor = frames_tensor.float()
        masks_tensor = masks_tensor.float()

    frames_tensor = frames_tensor / 255.0 * 2 - 1  # Normalize to [-1, 1]
    masks_tensor = masks_tensor / 255.0  # Normalize to [0, 1]

    return frames_tensor, masks_tensor


# MODEL_DIR = Path("release_model")
# CKPT_PATH = MODEL_DIR / "E2FGVI-HQ-CVPR22.pth"

# TODO: RuntimeError: MPS: Unsupported Border padding mode
# mps doesn't work here.....
device = get_device()
if device.type == "mps":
    logger.warning(
        f"E2FGVI_HQ Cleaner doesn't support MPS, using CPU instead. But it is very very slow!!"
    )
    device = torch.device("cpu")


class E2FGVIHDConfig(BaseModel):
    ref_length: int = 10
    num_ref: int = -1
    neighbor_stride: int = 5
    chunk_size_ratio: float = 0.2  # TODO: this can be adjust as the VRAM
    overlap_ratio: int = 0.05
    enable_torch_compile: bool = ENABLE_E2FGVI_HQ_TORCH_COMPILE
    use_bf16: bool = is_bf16_supported()  # Enable bf16 inference for faster processing


class E2FGVIHDCleaner:
    def __init__(
        self,
        ckpt_path: Path = E2FGVI_HQ_CHECKPOINT_PATH,
        config: E2FGVIHDConfig = E2FGVIHDConfig(),
    ):
        """
        Initialize the cleaner by ensuring the model checkpoint is available, loading the inpainting model onto the computation device, setting it to evaluation mode, storing the configuration, and profiling VRAM to determine the chunk size.

        Parameters:
            ckpt_path (Path): Path to the model checkpoint file. Defaults to the configured E2FGVI-HQ checkpoint path and will be downloaded if missing.
            config (E2FGVIHDConfig): Configuration for reference selection, neighbor stride, and chunking behavior used during processing.
        """
        ensure_model_downloaded(ckpt_path, E2FGVI_HQ_CHECKPOINT_REMOTE_URL)
        self.model = InpaintGenerator().to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.config = config
        self.device = device

        # Enable bf16 if configured and supported
        self.use_bf16 = config.use_bf16 and device.type == "cuda"
        if self.use_bf16:
            logger.info("Enabling bf16 inference for E2FGVI_HQ cleaner")
            # Convert model to bfloat16 for better performance with torch.compile
            self.model = self.model.to(dtype=torch.bfloat16)

        # Flag to track if artifacts have been saved
        self._artifacts_saved = False
        
        self.profiling_chunk_size()
        self.auto_compile()

    def auto_compile(self):
        if self.config.enable_torch_compile:
            try:
                # Use different cache files for bf16 and fp32 modes
                self.artifacts_path = (
                    E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS_BF16
                    if self.use_bf16
                    else E2FGVI_HQ_TORCH_COMPILE_ARTIFACTS
                )

                if self.artifacts_path.exists():
                    logger.info(
                        f"Loading cached torch compile artifacts from {self.artifacts_path}"
                    )
                    artifact_bytes = self.artifacts_path.read_bytes()
                    torch.compiler.load_cache_artifacts(artifact_bytes)
                    # Use default mode for better stability with bf16
                    compile_mode = "default"  # if self.use_bf16 else "max-autotune"
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self._artifacts_saved = True  # Already have cached artifacts

                else:
                    logger.info(
                        f"Compiling model on first inference. Artifacts will be saved to {self.artifacts_path}"
                    )
                    # Use default mode for bf16 to avoid compilation issues
                    compile_mode = "default"  # if self.use_bf16 else "max-autotune"
                    logger.info(f"Using torch.compile mode: {compile_mode}")
                    self.model = torch.compile(self.model, mode=compile_mode)
                    # Will save artifacts after first inference

            except Exception as e:
                logger.warning(
                    f"Device not support torch compile, or compile failed: {e}. using the original model instead."
                )
        else:
            self.artifacts_path = None  # No artifacts path when compile is disabled

    def profiling_chunk_size(self):
        # memory_profiling
        # 1GB can process about 5 frames in chunk size
        """
        Set the instance's adapted_chunk_size based on available GPU VRAM.

        Profiles free device memory and computes adapted_chunk_size by multiplying the measured free memory (in GB) with CHUNK_SIZE_PER_GB_VRAM; stores the result in `self.adapted_chunk_size` and logs the chosen chunk size.
        """
        if self.device.type != "cuda":
            # For non-CUDA devices, use a default chunk size that is reasonable for CPU processing
            self.adapted_chunk_size = CHUNK_SIZE_PER_GB_VRAM  
            logger.debug(
                f"Non-CUDA device detected. Using default chunk size: {self.adapted_chunk_size}"
            )
        else:
            memory_profiling_results = memory_profiling()
            adapted_chunk_size = int(
                memory_profiling_results.free_memory * CHUNK_SIZE_PER_GB_VRAM
            )
            self.adapted_chunk_size = min(adapted_chunk_size, MAX_ADAPTED_CHUNK_SIZE)
            logger.debug(
                # keep two digit
                f"Chunk size is set to {self.adapted_chunk_size} based on the free VRAM {round(memory_profiling_results.free_memory, 2)}GB"
            )

    @property
    def chunk_size(self):
        """
        Configured chunk size used for processing frames.

        Returns:
            chunk_size (int): The adapted number of frames per chunk computed from available VRAM.
        """
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
        h_pad: int,
        w_pad: int,
    ) -> List[np.ndarray]:
        """
        Compose and return cleaned frames for a chunk by inpainting masked regions using neighboring and reference frames.

        Parameters:
            chunk_length (int): Number of frames in the chunk.
            neighbor_stride (int): Step between neighbor anchor frames processed within the chunk.
            imgs_chunk (torch.Tensor): Tensor of input frames with shape (1, T_chunk, 3, H, W), normalized for the model (typically in [-1, 1]).
            masks_chunk (torch.Tensor): Corresponding masks with shape (1, T_chunk, 1, H, W), values in [0, 1] where 1 indicates masked pixels.
            binary_masks_chunk (np.ndarray): Binary masks used to blend predictions with originals; shape (chunk_length, H, W, 3) or (chunk_length, H, W) broadcastable to frames, dtype uint8 or {0,1}.
            frames_np_chunk (np.ndarray): Original frames as uint8 numpy arrays with shape (chunk_length, H, W, 3).
            h (int): Frame height.
            w (int): Frame width.
            h_pad (int): Pre-computed height padding to align to mod 60.
            w_pad (int): Pre-computed width padding to align to mod 108.

        Returns:
            List[np.ndarray]: List of length `chunk_length` containing the composited uint8 frames (H x W x 3) for processed indices; entries may be `None` for frames that were not covered/updated.
        """
        comp_frames_chunk = [None] * chunk_length

        for f in tqdm(
            range(0, chunk_length, neighbor_stride),
            desc=f"    Frame progress",
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

            with torch.inference_mode():
                nvtx.range_push("prepare_input")
                masked_imgs = selected_imgs * (1 - selected_masks)
                if h_pad > 0 or w_pad > 0:
                    # pad last 3 dims: (C_left, C_right, W_left, W_right, H_left, H_right)
                    masked_imgs = torch.nn.functional.pad(
                        masked_imgs, (0, w_pad, 0, h_pad, 0, 0), mode="reflect"
                    )
                nvtx.range_pop()  # prepare_input
                nvtx.range_push("model_forward")
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                nvtx.range_pop()  # model_forward
                nvtx.range_push("postprocess")
                # Crop and take only neighbor frames (model outputs all frames including refs)
                pred_imgs = pred_imgs[:len(neighbor_ids), :, :h, :w]
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
                        )
                nvtx.range_pop()  # postprocess

        return comp_frames_chunk

    def clean(self, frames: np.ndarray, masks: np.ndarray) -> List[np.ndarray]:
        """
        Clean a video by processing frames and masks in overlapping, memory-adaptive chunks and merging the inpainted results.

        Parameters:
            frames (np.ndarray): Video frames as a (T, H, W, 3) uint8 array (RGB) where T is the number of frames.
            masks (np.ndarray): Corresponding masks as a (T, H, W) uint8 array; non-zero values indicate masked pixels to be inpainted.

        Returns:
            List[np.ndarray]: List of T cleaned frames as (H, W, 3) uint8 arrays (RGB), where each entry is the composited result for the corresponding input frame.
        """
        video_length = len(frames)
        chunk_size = int(self.config.chunk_size_ratio * video_length)
        overlap_size = int(self.config.overlap_ratio * video_length)
        num_chunks = int(np.ceil(video_length / (chunk_size - overlap_size)))
        h, w = frames[0].shape[:2]
        # Pre-compute padding once for the whole video
        h_pad = (60 - h % 60) % 60
        w_pad = (108 - w % 108) % 108
        # Determine target dtype for GPU tensors
        target_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        # Convert directly to GPU tensors, skipping float32 intermediate on CPU
        imgs_all, masks_all = numpy_to_tensor(frames, masks, target_device=device, target_dtype=target_dtype)
        # Prepare binary masks for compositing
        binary_masks = np.expand_dims(masks > 0, axis=-1).astype(
            np.uint8
        )  # (T, H, W, 1)
        comp_frames = [None] * video_length
        logger.debug(
            f"Processing {video_length} frames in {num_chunks} chunks (chunk_size={chunk_size}, overlap={overlap_size})"
        )

        for chunk_idx in tqdm(
            range(num_chunks), desc="  Chunk", position=1, leave=False
        ):
            start_idx = chunk_idx * (chunk_size - overlap_size)
            end_idx = min(start_idx + chunk_size, video_length)
            actual_chunk_size = end_idx - start_idx
            # Slice from already-GPU tensors (already correct dtype)
            imgs_chunk = imgs_all[:, start_idx:end_idx, :, :, :]
            masks_chunk = masks_all[:, start_idx:end_idx, :, :, :]
            frames_np_chunk = frames[start_idx:end_idx]
            binary_masks_chunk = binary_masks[start_idx:end_idx]
            # Process chunk
            comp_frames_chunk = self.process_frames_chunk(
                actual_chunk_size,
                self.config.neighbor_stride,
                imgs_chunk,
                masks_chunk,
                binary_masks_chunk,
                frames_np_chunk,
                h,
                w,
                h_pad,
                w_pad,
            )
            # Merge results with blending in overlap region
            comp_frames = merge_frames_with_overlap(
                result_frames=comp_frames,
                chunk_frames=comp_frames_chunk,
                start_idx=start_idx,
                overlap_size=overlap_size,
                is_first_chunk=(chunk_idx == 0),
            )
            del imgs_chunk, masks_chunk, comp_frames_chunk

        # Save torch compile artifacts after first inference
        if self.config.enable_torch_compile and not self._artifacts_saved:
            try:
                artifacts = torch.compiler.save_cache_artifacts()
                if artifacts is not None:
                    artifact_bytes, _ = artifacts
                    self.artifacts_path.write_bytes(artifact_bytes)
                    logger.info(
                        f"Saved torch compile artifacts to {self.artifacts_path}"
                    )
                    self._artifacts_saved = True
            except Exception as e:
                logger.warning(f"Failed to save torch compile artifacts: {e}")

        return comp_frames


if __name__ == "__main__":
    #       --frames examples/extract_frame_and_mask_frames.npy \
    #   --masks examples/extract_frame_and_mask_masks.npy \
    import os

    import cv2

    frames_path = Path("examples/extract_frame_and_mask_frames.npy")
    masks_path = Path("examples/extract_frame_and_mask_masks.npy")
    frames_np = np.load(frames_path)
    masks_np = np.load(masks_path)
    # Convert BGR to RGB if frames are saved in BGR format
    frames_np = frames_np[:, :, :, ::-1].copy()
    cleaner = E2FGVIHDCleaner()
    comp_frames = cleaner.clean(frames_np, masks_np)

    # Save the result as video
    fps = 30
    output_video_path = "results/output.mp4"
    h, w = frames_np[0].shape[:2]

    os.makedirs("results", exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in comp_frames:
        # Convert RGB to BGR for OpenCV
        writer.write(frame.astype(np.uint8)[:, :, ::-1])
    writer.release()
    logger.info(f"Video saved to: {output_video_path}")
