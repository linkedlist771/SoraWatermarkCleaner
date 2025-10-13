from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from sorawm.configs import DEFAULT_WATERMARK_REMOVE_MODEL
from sorawm.iopaint.const import DEFAULT_MODEL_DIR
from sorawm.iopaint.download import cli_download_model, scan_models
from sorawm.iopaint.model_manager import ModelManager
from sorawm.iopaint.schema import InpaintRequest
from sorawm.utils.devices_utils import get_device

# This codebase is from https://github.com/Sanster/IOPaint#, thanks for their amazing work!


class WaterMarkCleaner:
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

    def clean(self, input_image: np.array, watermark_mask: np.array) -> np.array:
        inpaint_result = self.model_manager(
            input_image, watermark_mask, self.inpaint_request
        )
        inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
        return inpaint_result


if __name__ == "__main__":
    from pathlib import Path

    import cv2
    import numpy as np
    from tqdm import tqdm

    # ========= 配置 =========
    video_path = Path("resources/puppies.mp4")
    save_video = True
    out_path = Path("outputs/dog_vs_sam_detected.mp4")
    window = "Sora watermark (threshold+morph+shape + tracking)"

    # 追踪/回退策略参数
    PREV_ROI_EXPAND = 2.2  # 上一框宽高的膨胀倍数（>1）
    AREA1 = (1000, 2000)  # 主检测面积范围
    AREA2 = (600, 4000)  # 回退阶段面积范围
    # =======================

    cleaner = SoraWaterMarkCleaner(video_path, video_path)

    # 预取一帧确定尺寸/FPS
    first_frame = None
    for first_frame in cleaner.input_video_loader:
        break
    assert first_frame is not None, "无法读取视频帧"
    H, W = first_frame.shape[:2]
    fps = getattr(cleaner.input_video_loader, "fps", 30)

    # 输出视频（原 | bw | all-contours | vis 四联画）
    writer = None
    if save_video:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W * 4, H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W * 4, H))
        assert writer.isOpened(), "无法创建输出视频文件"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # ---- 工具函数 ----
    def _clip_rect(x0, y0, x1, y1, w_img, h_img):
        x0 = max(0, min(x0, w_img - 1))
        x1 = max(0, min(x1, w_img))
        y0 = max(0, min(y0, h_img - 1))
        y1 = max(0, min(y1, h_img))
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        return x0, y0, x1, y1

    def _cnt_bbox(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, x + w, y + h)

    def _bbox_center(b):
        x0, y0, x1, y1 = b
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    def detect_flower_like(image, prev_bbox=None):
        """
        识别流程：
        灰度范围 → 自适应阈值 → 仅在 3 个区域 + (可选)上一帧膨胀ROI 内找轮廓
        三个区域：1) 左上20%  2) 左下20%  3) 中间水平带 y∈[0.4H, 0.6H], x∈[0,W]
        返回: bw_region, best_cnt, contours_region, region_boxes, prev_roi_box
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 208 ± 20% 亮度范围
        low, high = int(round(208 * 0.9)), int(round(208 * 1.1))
        mask = ((gray >= low) & (gray <= high)).astype(np.uint8) * 255

        # 自适应阈值并限制到亮度范围
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -5
        )
        bw = cv2.bitwise_and(bw, mask)

        # -------- 三个区域：左上/左下/中间带 --------
        h_img, w_img = gray.shape[:2]
        r_top_left = (0, 0, int(0.2 * w_img), int(0.2 * h_img))
        r_bot_left = (0, int(0.8 * h_img), int(0.2 * w_img), h_img)
        y0, y1 = int(0.40 * h_img), int(0.60 * h_img)  # 中间带
        r_mid_band = (0, y0, w_img, y1)

        region_mask = np.zeros_like(bw, dtype=np.uint8)
        for x0, ys, x1, ye in (r_top_left, r_bot_left):
            region_mask[ys:ye, x0:x1] = 255
        region_mask[y0:y1, :] = 255

        # -------- 追加：上一帧膨胀ROI --------
        prev_roi_box = None
        if prev_bbox is not None:
            px0, py0, px1, py1 = prev_bbox
            pw, ph = (px1 - px0), (py1 - py0)
            cx, cy = _bbox_center(prev_bbox)
            rw = int(pw * PREV_ROI_EXPAND)
            rh = int(ph * PREV_ROI_EXPAND)
            rx0, ry0 = cx - rw // 2, cy - rh // 2
            rx1, ry1 = cx + rw // 2, cy + rh // 2
            rx0, ry0, rx1, ry1 = _clip_rect(rx0, ry0, rx1, ry1, w_img, h_img)
            region_mask[ry0:ry1, rx0:rx1] = 255
            prev_roi_box = (rx0, ry0, rx1, ry1)

        bw_region = cv2.bitwise_and(bw, region_mask)

        # -------- 轮廓 + 形状筛选 --------
        def select_candidates(bw_bin, area_rng):
            contours, _ = cv2.findContours(
                bw_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cand = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < area_rng[0] or area > area_rng[1]:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circularity = 4.0 * np.pi * area / (peri * peri)
                if 0.55 <= circularity <= 0.95:
                    cand.append(cnt)
            return contours, cand

        contours_region, cand1 = select_candidates(bw_region, AREA1)

        best_cnt = None
        if cand1:
            # 若有上一帧，用“离上一帧中心最近”优先；否则取面积最大
            if prev_bbox is None:
                best_cnt = max(cand1, key=lambda c: cv2.contourArea(c))
            else:
                pcx, pcy = _bbox_center(prev_bbox)
                best_cnt = max(
                    cand1,
                    key=lambda c: -(
                        (((_cnt_bbox(c)[0] + _cnt_bbox(c)[2]) // 2 - pcx) ** 2)
                        + (((_cnt_bbox(c)[1] + _cnt_bbox(c)[3]) // 2 - pcy) ** 2)
                    ),
                )
        else:
            # 回退1：仅在上一帧 ROI 内放宽面积
            if prev_roi_box is not None:
                rx0, ry0, rx1, ry1 = prev_roi_box
                roi = np.zeros_like(bw_region)
                roi[ry0:ry1, rx0:rx1] = bw_region[ry0:ry1, rx0:rx1]
                _, cand2 = select_candidates(roi, AREA2)
                if cand2:
                    if prev_bbox is None:
                        best_cnt = max(cand2, key=lambda c: cv2.contourArea(c))
                    else:
                        pcx, pcy = _bbox_center(prev_bbox)
                        best_cnt = max(
                            cand2,
                            key=lambda c: -(
                                (((_cnt_bbox(c)[0] + _cnt_bbox(c)[2]) // 2 - pcx) ** 2)
                                + (
                                    ((_cnt_bbox(c)[1] + _cnt_bbox(c)[3]) // 2 - pcy)
                                    ** 2
                                )
                            ),
                        )
                else:
                    # 回退2：全区域 cand，选最近中心
                    if prev_bbox is not None:
                        _, cand3 = select_candidates(bw_region, AREA2)
                        if cand3:
                            pcx, pcy = _bbox_center(prev_bbox)
                            best_cnt = max(
                                cand3,
                                key=lambda c: -(
                                    (
                                        ((_cnt_bbox(c)[0] + _cnt_bbox(c)[2]) // 2 - pcx)
                                        ** 2
                                    )
                                    + (
                                        ((_cnt_bbox(c)[1] + _cnt_bbox(c)[3]) // 2 - pcy)
                                        ** 2
                                    )
                                ),
                            )

        region_boxes = (r_top_left, r_bot_left, r_mid_band, (y0, y1))
        return bw_region, best_cnt, contours_region, region_boxes, prev_roi_box

    # ---- 时序追踪状态（用字典避免 nonlocal/global） ----
    state = {"bbox": None}  # 保存上一帧外接框 (x0,y0,x1,y1)

    def process_and_show(frame, idx):
        img = frame.copy()
        bw, best, contours, region_boxes, prev_roi_box = detect_flower_like(
            img, state["bbox"]
        )
        r_top_left, r_bot_left, r_mid_band, (y0, y1) = region_boxes

        # 所有轮廓（黄）
        allc = img.copy()
        if contours:
            cv2.drawContours(allc, contours, -1, (0, 255, 255), 1)

        # 画三个区域：红框 + 中间带上下红线
        def draw_rect(im, rect, color=(0, 0, 255), th=2):
            x0, y0r, x1, y1r = rect
            cv2.rectangle(im, (x0, y0r), (x1, y1r), color, th)

        draw_rect(allc, r_top_left)
        draw_rect(allc, r_bot_left)
        draw_rect(allc, (r_mid_band[0], r_mid_band[1], r_mid_band[2], r_mid_band[3]))
        cv2.line(allc, (0, y0), (img.shape[1], y0), (0, 0, 255), 2)
        cv2.line(allc, (0, y1), (img.shape[1], y1), (0, 0, 255), 2)

        # 画上一帧的膨胀 ROI（青色）
        if prev_roi_box is not None:
            x0, y0r, x1, y1r = prev_roi_box
            cv2.rectangle(allc, (x0, y0r), (x1, y1r), (255, 255, 0), 2)

        # 最终检测
        vis = img.copy()
        title = "no-detect"
        if best is not None:
            cv2.drawContours(vis, [best], -1, (0, 255, 0), 2)
            x0, y0r, x1, y1r = _cnt_bbox(best)
            state["bbox"] = (x0, y0r, x1, y1r)  # 更新追踪状态
            M = cv2.moments(best)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            title = "detected"
        else:
            # 若仍未检测，维持上一状态
            cv2.putText(
                vis,
                "No detection (kept last state)",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            if state["bbox"] is not None:
                x0, y0r, x1, y1r = state["bbox"]
                cv2.rectangle(vis, (x0, y0r), (x1, y1r), (255, 255, 0), 2)

        # 四联画：原图 | 区域内bw | 所有轮廓 | 最终检测
        panel = np.hstack([img, cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), allc, vis])
        cv2.putText(
            panel,
            f"Frame {idx} | {title}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window, panel)
        if writer is not None:
            if panel.shape[:2] != (H, W * 4):
                panel = cv2.resize(panel, (W * 4, H), interpolation=cv2.INTER_AREA)
            writer.write(panel)

    # 先处理已取出的第一帧
    process_and_show(first_frame, 0)

    # 按你的遍历方式继续
    for idx, frame in enumerate(
        tqdm(cleaner.input_video_loader, desc="Processing frames", initial=1, unit="f")
    ):
        process_and_show(frame, idx)
        key = cv2.waitKey(max(1, int(1000 / max(1, int(fps))))) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k in (ord(" "), ord("q")):
                    if k == ord("q"):
                        idx = 10**9
                    break
            if idx >= 10**9:
                break

    if writer is not None:
        writer.release()
        print(f"[OK] 可视化视频已保存: {out_path}")

    cv2.destroyAllWindows()
