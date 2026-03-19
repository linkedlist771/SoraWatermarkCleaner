from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from sorawm.configs import DEFAULT_WATERMARK_REMOVE_MODEL
from sorawm.iopaint.const import DEFAULT_MODEL_DIR
from sorawm.iopaint.download import cli_download_model, scan_models
from sorawm.iopaint.helper import norm_img
from sorawm.iopaint.model_manager import ModelManager
from sorawm.iopaint.schema import InpaintRequest
from sorawm.utils.devices_utils import get_device

# This codebase is from https://github.com/Sanster/IOPaint#, thanks for their amazing work!

LAMA_INFER_BATCH_SIZE = 4  # Optimal batch size for LAMA on 704x1280 frames


class LamaCleaner:
    def __init__(self):
        self.model = DEFAULT_WATERMARK_REMOVE_MODEL
        self.device = get_device()

        scanned_models = scan_models()
        if self.model not in [it.name for it in scanned_models]:
            logger.info(
                f"{self.model} not found in {DEFAULT_MODEL_DIR}, try to downloading"
            )
            cli_download_model(self.model)
        self.model_manager = ModelManager(name=self.model, device=self.device)
        self.inpaint_request = InpaintRequest()
        # Direct reference to the JIT model for batched inference
        self._jit_model = self.model_manager.model.model

    def clean(self, input_image: np.array, watermark_mask: np.array) -> np.array:
        inpaint_result = self.model_manager(
            input_image, watermark_mask, self.inpaint_request
        )
        inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
        return inpaint_result

    def clean_batch(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Process a batch of frames through LAMA simultaneously.

        frames: list of (H, W, 3) uint8 BGR images
        masks: list of (H, W) uint8 masks (255 = inpaint area)
        Returns: list of (H, W, 3) uint8 BGR cleaned images
        """
        # Build batch tensors: (N, C, H, W) float32
        imgs_np = np.stack([norm_img(f) for f in frames], axis=0)   # (N, 3, H, W)
        msks_np = np.stack([norm_img(m) for m in masks], axis=0)     # (N, 1, H, W)
        img_t = torch.from_numpy(imgs_np).to(self.device)
        msk_t = (torch.from_numpy(msks_np).to(self.device) > 0).float()

        with torch.inference_mode():
            result = self._jit_model(img_t, msk_t)  # (N, 3, H, W) float

        result_np = result.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        # Convert RGB -> BGR (LAMA outputs RGB, pipeline expects BGR)
        return [cv2.cvtColor(r, cv2.COLOR_RGB2BGR) for r in result_np]
