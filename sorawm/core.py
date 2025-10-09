from pathlib import Path
from typing import Callable
from collections import deque

import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm

from sorawm.utils.video_utils import VideoLoader
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector


class SoraWM:
    def __init__(self, temporal_smooth_window: int = 3, keyframe_interval: int = 5):
        """
        Args:
            temporal_smooth_window: 时序平滑窗口大小，用于平滑inpainting结果（奇数）
            keyframe_interval: 关键帧间隔，每N帧重新inpainting一次
        """
        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner()
        self.temporal_smooth_window = temporal_smooth_window
        self.keyframe_interval = keyframe_interval

    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 30.0) -> bool:
        """检测两帧之间是否发生场景切换"""
        diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
        return diff > threshold

    def temporal_smooth_inpainted_region(
        self, 
        current_frame: np.ndarray, 
        bbox: tuple, 
        inpainted_frames_buffer: deque
    ) -> np.ndarray:
        """
        对inpainting区域进行时序平滑
        
        Args:
            current_frame: 当前帧
            bbox: 水印区域的bbox (x1, y1, x2, y2)
            inpainted_frames_buffer: 存储最近几帧inpainting结果的缓冲区
        
        Returns:
            平滑后的帧
        """
        if len(inpainted_frames_buffer) == 0:
            return current_frame
        
        x1, y1, x2, y2 = bbox
        result_frame = current_frame.copy()
        
        # 对水印区域进行时序平滑（加权平均）
        region_sum = np.zeros((y2 - y1, x2 - x1, 3), dtype=float)
        weights_sum = 0
        
        for i, frame in enumerate(inpainted_frames_buffer):
            # 使用高斯权重，越近的帧权重越大
            weight = np.exp(-0.5 * ((i - len(inpainted_frames_buffer) // 2) ** 2))
            region_sum += frame[y1:y2, x1:x2].astype(float) * weight
            weights_sum += weight
        
        # 加入当前帧
        weight_current = 1.0
        region_sum += current_frame[y1:y2, x1:x2].astype(float) * weight_current
        weights_sum += weight_current
        
        # 计算加权平均
        smoothed_region = (region_sum / weights_sum).astype(np.uint8)
        result_frame[y1:y2, x1:x2] = smoothed_region
        
        return result_frame

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames
        

        
        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"
        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",  
        }
        
        if input_video_loader.original_bitrate:
            output_options["video_bitrate"] = str(int(int(input_video_loader.original_bitrate) * 1.2))
        else:
            output_options["crf"] = "18"
        
        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        frame_and_mask = {}
        detect_missed = []

        logger.debug(
            f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}"
        )
        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect watermarks")
        ):
            detection_result = self.detector.detect(frame)
            if detection_result["detected"]:
                frame_and_mask[idx] = {"frame": frame, "bbox": detection_result["bbox"]}
            else:
                frame_and_mask[idx] = {"frame": frame, "bbox": None}
                detect_missed.append(idx)

            # 10% - 50%
            if progress_callback and idx % 10 == 0:
                progress = 10 + int((idx / total_frames) * 40)
                progress_callback(progress)

        logger.debug(f"detect missed frames: {detect_missed}")

        for missed_idx in detect_missed:
            before = max(missed_idx - 1, 0)
            after = min(missed_idx + 1, total_frames - 1)
            before_box = frame_and_mask[before]["bbox"]
            after_box = frame_and_mask[after]["bbox"]
            if before_box:
                frame_and_mask[missed_idx]["bbox"] = before_box
            elif after_box:
                frame_and_mask[missed_idx]["bbox"] = after_box

        # 使用关键帧策略和时序平滑
        inpainted_frames_buffer = deque(maxlen=self.temporal_smooth_window)
        last_keyframe_inpainted_region = None
        last_keyframe_bbox = None
        prev_frame = None
        scene_change_detected = False
        
        for idx in tqdm(range(total_frames), desc="Remove watermarks"):
            frame_info = frame_and_mask[idx]
            frame = frame_info["frame"]
            bbox = frame_info["bbox"]
            
            # 检测场景变化
            if prev_frame is not None and idx > 0:
                scene_change_detected = self.detect_scene_change(prev_frame, frame)
                if scene_change_detected:
                    logger.debug(f"Scene change detected at frame {idx}")
                    inpainted_frames_buffer.clear()
                    last_keyframe_inpainted_region = None
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                
                # 判断是否需要重新inpainting（关键帧或场景切换）
                is_keyframe = (idx % self.keyframe_interval == 0) or scene_change_detected or (last_keyframe_inpainted_region is None)
                
                if is_keyframe:
                    # 关键帧：执行完整的inpainting
                    cleaned_frame = self.cleaner.clean(frame, mask)
                    last_keyframe_inpainted_region = cleaned_frame[y1:y2, x1:x2].copy()
                    last_keyframe_bbox = bbox
                    logger.debug(f"Keyframe inpainting at frame {idx}")
                else:
                    # 非关键帧：复用上一个关键帧的inpainting结果
                    cleaned_frame = frame.copy()
                    if last_keyframe_inpainted_region is not None and bbox == last_keyframe_bbox:
                        # bbox位置相同，直接复用
                        cleaned_frame[y1:y2, x1:x2] = last_keyframe_inpainted_region
                    else:
                        # bbox位置变化了，需要重新inpainting
                        cleaned_frame = self.cleaner.clean(frame, mask)
                        last_keyframe_inpainted_region = cleaned_frame[y1:y2, x1:x2].copy()
                        last_keyframe_bbox = bbox
                
                # 应用时序平滑
                if len(inpainted_frames_buffer) > 0:
                    cleaned_frame = self.temporal_smooth_inpainted_region(
                        cleaned_frame, bbox, inpainted_frames_buffer
                    )
                
                # 将当前帧加入缓冲区
                inpainted_frames_buffer.append(cleaned_frame.copy())
            else:
                cleaned_frame = frame
            
            process_out.stdin.write(cleaned_frame.tobytes())
            prev_frame = frame.copy()
            scene_change_detected = False

            # 50% - 95%
            if progress_callback and idx % 10 == 0:
                progress = 50 + int((idx / total_frames) * 45)
                progress_callback(progress)

        process_out.stdin.close()
        process_out.wait()

        # 95% - 99%
        if progress_callback:
            progress_callback(95)

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            progress_callback(99)

    def merge_audio_track(
        self, input_video_path: Path, temp_output_path: Path, output_video_path: Path
    ):
        logger.info("Merging audio track...")
        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        # Clean up temporary file
        temp_output_path.unlink()
        logger.info(f"Saved no watermark video with audio at: {output_video_path}")


if __name__ == "__main__":
    from pathlib import Path

    input_video_path = Path(
        "resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
