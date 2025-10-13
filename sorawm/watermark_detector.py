from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO

from sorawm.configs import WATER_MARK_DETECT_YOLO_WEIGHTS
from sorawm.utils.download_utils import download_detector_weights
from sorawm.utils.devices_utils import get_device
from sorawm.utils.video_utils import VideoLoader

# based on the sora tempalte to detect the whole, and then got the icon part area.


class SoraWaterMarkDetector:
    def __init__(self):
        download_detector_weights()
        logger.debug(f"Begin to load yolo water mark detet model.")
        self.model = YOLO(WATER_MARK_DETECT_YOLO_WEIGHTS)
        self.model.to(str(get_device())) 
        logger.debug(f"Yolo water mark detet model loaded.")

        self.model.eval()

    def detect(self, input_image: np.array):
        # Run YOLO inference
        results = self.model(input_image, verbose=False)
        # Extract predictions from the first (and only) result
        result = results[0]

        # Check if any detections were made
        if len(result.boxes) == 0:
            return {"detected": False, "bbox": None, "confidence": None, "center": None}

        # Get the first detection (highest confidence)
        box = result.boxes[0]

        # Extract bounding box coordinates (xyxy format)
        # Convert tensor to numpy, then to python float, finally to int
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        # Extract confidence score
        confidence = float(box.conf[0].cpu().numpy())
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return {
            "detected": True,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": confidence,
            "center": (int(center_x), int(center_y)),
        }


if __name__ == "__main__":
    from pathlib import Path

    import cv2
    from tqdm import tqdm

    # ========= 配置 =========
    # video_path = Path("resources/puppies.mp4") # 19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4
    video_path = Path("resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4")
    save_video = True
    out_path = Path("outputs/sora_watermark_yolo_detected.mp4")
    window = "Sora Watermark YOLO Detection"
    # =======================

    # 初始化检测器
    detector = SoraWaterMarkDetector()

    # 初始化视频加载器
    video_loader = VideoLoader(video_path)

    # 预取一帧确定尺寸/FPS
    first_frame = None
    for first_frame in video_loader:
        break
    assert first_frame is not None, "无法读取视频帧"

    H, W = first_frame.shape[:2]
    fps = getattr(video_loader, "fps", 30)

    # 输出视频设置
    writer = None
    if save_video:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        assert writer.isOpened(), "无法创建输出视频文件"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def visualize_detection(frame, detection_result, frame_idx):
        """在帧上可视化检测结果"""
        vis = frame.copy()

        if detection_result["detected"]:
            # 绘制边界框
            x1, y1, x2, y2 = detection_result["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制中心点
            cx, cy = detection_result["center"]
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # 显示置信度
            conf = detection_result["confidence"]
            label = f"Watermark: {conf:.2f}"

            # 文本背景
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), (0, 255, 0), -1
            )

            # 绘制文本
            cv2.putText(
                vis,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            status = f"Frame {frame_idx} | DETECTED | Conf: {conf:.3f}"
            status_color = (0, 255, 0)
        else:
            status = f"Frame {frame_idx} | NO WATERMARK"
            status_color = (0, 0, 255)

        # 显示帧信息
        cv2.putText(
            vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )

        return vis

    # 处理第一帧
    print("开始处理视频...")
    detection = detector.detect(first_frame)
    vis_frame = visualize_detection(first_frame, detection, 0)
    cv2.imshow(window, vis_frame)
    if writer is not None:
        writer.write(vis_frame)

    # 处理剩余帧
    total_frames = 0
    detected_frames = 0

    for idx, frame in enumerate(
        tqdm(video_loader, desc="Processing frames", initial=1, unit="f"), start=1
    ):
        # YOLO 检测
        detection = detector.detect(frame)

        # 可视化
        vis_frame = visualize_detection(frame, detection, idx)

        # 统计
        total_frames += 1
        if detection["detected"]:
            detected_frames += 1

        # 显示
        cv2.imshow(window, vis_frame)

        # 保存
        if writer is not None:
            writer.write(vis_frame)

        # 按键控制
        key = cv2.waitKey(max(1, int(1000 / max(1, int(fps))))) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # 空格暂停
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k in (ord(" "), ord("q")):
                    if k == ord("q"):
                        idx = 10**9
                    break
            if idx >= 10**9:
                break

    # 清理
    if writer is not None:
        writer.release()
        print(f"\n[完成] 可视化视频已保存: {out_path}")

    # 打印统计信息
    total_frames += 1  # 包括第一帧
    if detection["detected"]:
        detected_frames += 1

    print(f"\n=== 检测统计 ===")
    print(f"总帧数: {total_frames}")
    print(f"检测到水印: {detected_frames} 帧")
    print(f"检测率: {detected_frames/total_frames*100:.2f}%")

    cv2.destroyAllWindows()
