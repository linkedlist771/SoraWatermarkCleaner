from pathlib import Path

import cv2
from tqdm import tqdm
from sorawm.watermark_detector import SoraWaterMarkDetector
from sorawm.configs import ROOT

videos_dir = ROOT / "videos"
datasets_dir = ROOT / "datasets"
images_dir = datasets_dir / "images"
images_dir.mkdir(exist_ok=True, parents=True)
detector = SoraWaterMarkDetector()


if __name__ == "__main__":
    fps_save_interval = 1  # Save every 5th frame

    video_idx = 0
    image_idx = 0  # 全局图片索引
    total_failed = 0  # 检测失败的总数
    
    for video_path in tqdm(list(videos_dir.rglob("*.mp4"))):
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        video_name = video_path.name
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            continue

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                # Break if no more frames
                if not ret:
                    break

                # Save frame at the specified interval
                if frame_count % fps_save_interval == 0:
                    if not detector.detect(frame)["detected"]:
                        # Create filename: image_idx_framecount.jpg
                        image_filename = f"{video_name}_failed_image_frame_{frame_count:06d}.jpg"
                        image_path = images_dir / image_filename
                        # Save the frame
                        cv2.imwrite(str(image_path), frame)
                        image_idx += 1
                        total_failed += 1

                frame_count += 1

        finally:
            # Release the video capture object
            cap.release()
        
        video_idx += 1

    print(f"Processed {video_idx} videos, extracted {total_failed} failed detection frames to {images_dir}")
