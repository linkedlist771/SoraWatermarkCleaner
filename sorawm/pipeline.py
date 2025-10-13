"""
Streaming pipeline module for overlapping detection and cleaning.
Improves GPU utilization by parallelizing detection and inpainting.
"""

import multiprocessing as mp
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from loguru import logger

from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.watermark_cleaner import WaterMarkCleaner

# Set spawn method for CUDA compatibility
# Must use 'spawn' instead of 'fork' when using CUDA in subprocesses
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


@dataclass
class FrameData:
    """Frame data structure for pipeline processing"""
    idx: int
    frame: np.ndarray
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) or None


class DetectorWorker:
    """Detector worker process - reads frames from input queue and outputs detection results"""
    
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detector = None
    
    @staticmethod
    def worker_main(input_queue: mp.Queue, output_queue: mp.Queue):
        """Detector worker main loop"""
        logger.debug("[DetectorWorker] Starting detector process")
        worker = DetectorWorker(input_queue, output_queue)
        
        # Initialize detector in subprocess to avoid CUDA context issues
        worker.detector = SoraWaterMarkDetector()
        
        while True:
            try:
                # Get frame from input queue
                frame_data = input_queue.get(timeout=1.0)
                
                if frame_data is None:  # Shutdown signal
                    logger.debug("[DetectorWorker] Received shutdown signal")
                    output_queue.put(None)  # Pass shutdown signal downstream
                    break
                
                # Perform detection
                detection_result = worker.detector.detect(frame_data.frame)
                
                # Update bbox
                if detection_result["detected"]:
                    frame_data.bbox = detection_result["bbox"]
                else:
                    frame_data.bbox = None
                
                # Put result into output queue
                output_queue.put(frame_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[DetectorWorker] Error: {e}")
                import traceback
                traceback.print_exc()
        
        logger.debug("[DetectorWorker] Exiting")


class CleanerWorker:
    """Cleaner worker process - reads detection results and outputs cleaned frames"""
    
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, width: int, height: int):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.width = width
        self.height = height
        self.cleaner = None
    
    @staticmethod
    def worker_main(input_queue: mp.Queue, output_queue: mp.Queue, width: int, height: int):
        """Cleaner worker main loop"""
        logger.debug("[CleanerWorker] Starting cleaner process")
        worker = CleanerWorker(input_queue, output_queue, width, height)
        
        # Initialize cleaner in subprocess to avoid CUDA context issues
        worker.cleaner = WaterMarkCleaner()
        
        # Buffer for handling frames with failed detections and out-of-order processing
        # Store recent frames for bbox interpolation
        recent_frames = {}  # idx -> {"frame": frame, "bbox": bbox, "cleaned": cleaned_frame}
        last_valid_bbox = None
        processed_count = 0
        
        while True:
            try:
                # Get detection result from input queue
                frame_data = input_queue.get(timeout=1.0)
                
                if frame_data is None:  # Shutdown signal
                    logger.debug(f"[CleanerWorker] Received shutdown signal, processed {processed_count} frames")
                    
                    # Send shutdown signal
                    output_queue.put(None)
                    break
                
                # Process frame immediately (streaming approach)
                frame = frame_data.frame
                bbox = frame_data.bbox
                
                # If no bbox detected, use last valid bbox or next valid bbox
                if bbox is None:
                    # Try to use last valid bbox
                    if last_valid_bbox is not None:
                        bbox = last_valid_bbox
                        logger.debug(f"[CleanerWorker] Frame {frame_data.idx}: Using last valid bbox")
                    else:
                        # Store for later processing when we have a valid bbox
                        recent_frames[frame_data.idx] = {"frame": frame, "bbox": None, "cleaned": None}
                        continue
                else:
                    last_valid_bbox = bbox
                
                # Clean the frame
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    mask = np.zeros((worker.height, worker.width), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 255
                    cleaned_frame = worker.cleaner.clean(frame, mask)
                else:
                    cleaned_frame = frame
                
                # Output cleaned frame immediately
                output_queue.put((frame_data.idx, cleaned_frame))
                processed_count += 1
                
                # Process any pending frames that were waiting for a valid bbox
                if last_valid_bbox is not None and recent_frames:
                    pending_indices = sorted(recent_frames.keys())
                    for pending_idx in pending_indices:
                        pending_info = recent_frames[pending_idx]
                        if pending_info["cleaned"] is None:
                            # Clean with the current valid bbox
                            x1, y1, x2, y2 = last_valid_bbox
                            mask = np.zeros((worker.height, worker.width), dtype=np.uint8)
                            mask[y1:y2, x1:x2] = 255
                            cleaned = worker.cleaner.clean(pending_info["frame"], mask)
                            output_queue.put((pending_idx, cleaned))
                            processed_count += 1
                            del recent_frames[pending_idx]
                            logger.debug(f"[CleanerWorker] Frame {pending_idx}: Processed with interpolated bbox")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[CleanerWorker] Error: {e}")
                import traceback
                traceback.print_exc()
        
        logger.debug("[CleanerWorker] Exiting")


class PipelineManager:
    """Pipeline manager - coordinates detector and cleaner processes"""
    
    def __init__(self, width: int, height: int, queue_size: int = 30):
        """
        Args:
            width: Video width
            height: Video height
            queue_size: Maximum queue size (controls memory usage)
        """
        self.width = width
        self.height = height
        self.queue_size = queue_size
        
        # Create inter-process communication queues
        self.detector_input_queue = mp.Queue(maxsize=queue_size)
        self.detector_output_queue = mp.Queue(maxsize=queue_size)
        self.cleaner_output_queue = mp.Queue(maxsize=queue_size)
        
        # Start worker processes
        self.detector_process = mp.Process(
            target=DetectorWorker.worker_main,
            args=(self.detector_input_queue, self.detector_output_queue),
            name="DetectorWorker",
            daemon=True
        )
        
        self.cleaner_process = mp.Process(
            target=CleanerWorker.worker_main,
            args=(self.detector_output_queue, self.cleaner_output_queue, width, height),
            name="CleanerWorker",
            daemon=True
        )
        
        self.detector_process.start()
        self.cleaner_process.start()
        
        logger.info("[PipelineManager] Pipeline started")
    
    def put_frame(self, idx: int, frame: np.ndarray, timeout: float = 5.0):
        """Put a frame into the pipeline"""
        frame_data = FrameData(idx=idx, frame=frame)
        self.detector_input_queue.put(frame_data, timeout=timeout)
    
    def signal_end(self):
        """Signal end of input"""
        self.detector_input_queue.put(None)
    
    def get_cleaned_frame(self, timeout: float = 5.0):
        """Get cleaned frame from pipeline
        
        Returns:
            (idx, cleaned_frame) or None (end signal)
        """
        return self.cleaner_output_queue.get(timeout=timeout)
    
    def shutdown(self, timeout: float = 10.0):
        """Shutdown the pipeline"""
        logger.info("[PipelineManager] Shutting down pipeline...")
        
        # Wait for processes to finish
        self.detector_process.join(timeout=timeout)
        self.cleaner_process.join(timeout=timeout)
        
        # Force terminate processes that didn't finish
        if self.detector_process.is_alive():
            logger.warning("[PipelineManager] Force terminating detector process")
            self.detector_process.terminate()
        
        if self.cleaner_process.is_alive():
            logger.warning("[PipelineManager] Force terminating cleaner process")
            self.cleaner_process.terminate()
        
        logger.info("[PipelineManager] Pipeline shutdown complete")

