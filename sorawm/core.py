from pathlib import Path
from typing import Callable
import threading

import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm

from sorawm.utils.video_utils import VideoLoader
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.pipeline import PipelineManager


class SoraWM:
    def __init__(self):
        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner()

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

        for idx in tqdm(range(total_frames), desc="Remove watermarks"):
            frame_info = frame_and_mask[idx]
            frame = frame_info["frame"]
            bbox = frame_info["bbox"]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                cleaned_frame = self.cleaner.clean(frame, mask)
            else:
                cleaned_frame = frame
            process_out.stdin.write(cleaned_frame.tobytes())

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

    def run_overlap(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ):
        """
        Run with pipeline architecture for overlapping detection and cleaning to improve GPU utilization.
        
        Args:
            input_video_path: Input video path
            output_video_path: Output video path
            progress_callback: Progress callback function
        """
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames

        logger.debug(
            f"Total frames: {total_frames}, FPS: {fps}, Width: {width}, Height: {height}"
        )

        # Create pipeline manager
        pipeline = PipelineManager(width=width, height=height, queue_size=30)

        # Prepare video output
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

        try:
            # Use threading to overlap input and output operations
            frame_buffer = {}  # Buffer for out-of-order frames
            next_frame_idx = 0
            received_count = 0
            input_error = None
            output_error = None
            
            def input_worker():
                """Thread to feed frames into pipeline"""
                nonlocal input_error
                try:
                    logger.info("[Pipeline] Starting input worker thread...")
                    for idx, frame in enumerate(input_video_loader):
                        pipeline.put_frame(idx, frame, timeout=30.0)
                        if progress_callback and idx % 10 == 0:
                            progress = 10 + int((idx / total_frames) * 20)
                            progress_callback(progress)
                    
                    # Signal end of input
                    pipeline.signal_end()
                    logger.info("[Pipeline] Input worker: All frames fed")
                except Exception as e:
                    input_error = e
                    logger.error(f"[Pipeline] Input worker error: {e}")
            
            def output_worker():
                """Thread to receive cleaned frames and write to video"""
                nonlocal next_frame_idx, received_count, output_error
                try:
                    logger.info("[Pipeline] Starting output worker thread...")
                    with tqdm(total=total_frames, desc="Processing frames") as pbar:
                        while received_count < total_frames:
                            result = pipeline.get_cleaned_frame(timeout=30.0)
                            if result is None:  # End signal
                                break

                            idx, cleaned_frame = result
                            frame_buffer[idx] = cleaned_frame
                            received_count += 1
                            
                            # Write frames in order
                            while next_frame_idx in frame_buffer:
                                process_out.stdin.write(frame_buffer[next_frame_idx].tobytes())
                                del frame_buffer[next_frame_idx]
                                next_frame_idx += 1
                                pbar.update(1)

                                if progress_callback and next_frame_idx % 10 == 0:
                                    progress = 30 + int((next_frame_idx / total_frames) * 65)
                                    progress_callback(progress)
                    
                    logger.info(f"[Pipeline] Output worker: Processed {next_frame_idx} frames total")
                except Exception as e:
                    output_error = e
                    logger.error(f"[Pipeline] Output worker error: {e}")
            
            # Start both threads
            input_thread = threading.Thread(target=input_worker, name="InputWorker")
            output_thread = threading.Thread(target=output_worker, name="OutputWorker")
            
            input_thread.start()
            output_thread.start()
            
            # Wait for both threads to complete
            input_thread.join()
            output_thread.join()
            
            # Check for errors
            if input_error:
                raise input_error
            if output_error:
                raise output_error

            process_out.stdin.close()
            process_out.wait()

            if progress_callback:
                progress_callback(95)

            # Stage 3: Merge audio track (95% - 99%)
            self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

            if progress_callback:
                progress_callback(99)

        finally:
            # Ensure pipeline is properly shutdown
            pipeline.shutdown()

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
