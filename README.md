# SoraWatermarkCleaner

English | [中文](README-zh.md)

This project provides an elegant way to remove the sora watermark in the sora2 generated videos.



<table>
  <tr>
    <td width="50%">
      <h3 align="center">Watermark removed</h3>
      <video src="https://github.com/user-attachments/assets/8cdc075e-7d15-4d04-8fa2-53dd287e5f4c" width="100%"></video>
    </td>
    <td width="50%">
      <h3 align="center">Original</h3>
      <video src="https://github.com/user-attachments/assets/4f032fc7-97da-471b-9a54-9de2a434fa57" width="100%"></video>
    </td>
  </tr>
</table>








⭐️: 

1. **Yolo weights has been updated, try the new version watermark detect model, it should work better. **

2. **We have uploaded the labelled datasets into huggingface, check this [dataset](https://huggingface.co/datasets/LLinked/sora-watermark-dataset) out. Free free to train your custom detector model or improve our model!**
3. **One-click portable build is available** — [Download here](#3-one-click-portable-version) for Windows users! No installation required.


## 1. Method

The SoraWatermarkCleaner(we call it `SoraWm` later) is composed of two parsts:

- SoraWaterMarkDetector: We trained a yolov11s version to detect the sora watermark. (Thank you yolo!)

- WaterMarkCleaner: We refer iopaint's implementation for watermark removal using the lama model.

  (This codebase is from https://github.com/Sanster/IOPaint#, thanks for their amazing work!)

Our SoraWm is purely deeplearning driven and yields good results in many generated videos.



## 2. Installation

[FFmpeg](https://ffmpeg.org/) is needed for video processing, please install it first.  We highly recommend using the `uv` to install the environments:

1. installation:

```bash
uv sync
```

> now the envs will be installed at the `.venv`, you can activate the env using:
>
> ```bash
> source .venv/bin/activate
> ```

2. Downloaded the pretrained models:

The trained yolo weights will be stored in the `resources` dir as the `best.pt`.  And it will be automatically download from https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/best.pt . The `Lama` model is downloaded from https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt, and will be stored in the torch cache dir. Both downloads are automatic, if you fail, please check your internet status.

## 3. One-Click Portable Version

For users who prefer a ready-to-use solution without manual installation, we provide a **one-click portable distribution** that includes all dependencies pre-configured.

### Download Links

**Google Drive:**
- [Download from Google Drive](https://drive.google.com/file/d/1ujH28aHaCXGgB146g6kyfz3Qxd-wHR1c/view?usp=share_link)

**Baidu Pan (百度网盘) - For users in China:**
- Link: https://pan.baidu.com/s/1i4exYsPvXv0evnGs5MWcYA?pwd=3jr6
- Extract Code (提取码): `3jr6`

### Features
- ✅ No installation required
- ✅ All dependencies included
- ✅ Pre-configured ironment
- ✅ Ready to use out of the box

Simply download, extract, and run!

## 4.  Demo

To have a basic usage, just try the `example.py`:

```python

from pathlib import Path
from sorawm.core import SoraWM


if __name__ == "__main__":
    input_video_path = Path(
        "resources/dog_vs_sam.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)

```

We also provide you with a `streamlit` based interactive web page, try it with:

```bash
streamlit run app.py
```

<img src="resources/app.png" style="zoom: 25%;" />

## 5. WebServer

Here, we provide a **FastAPI-based web server** that can quickly turn this watermark remover into a service.

Simply run:

```
python start_server.py
```

The web server will start on port **5344**.

You can view the FastAPI [documentation](http://localhost:5344/docs) for more details.

There are three routes available:

1. **submit_remove_task**

   > After uploading a video, a task ID will be returned, and the video will begin processing immediately.

<img src="resources/53abf3fd-11a9-4dd7-a348-34920775f8ad.png" alt="image" style="zoom: 25%;" />

2. **get_results**

You can use the task ID obtained above to check the task status.

It will display the percentage of video processing completed.

Once finished, the returned data will include a **download URL**.

3. **download**

You can use the **download URL** from step 2 to retrieve the cleaned video.

## 6. Datasets

We have uploaded the labelled datasets into huggingface, check this out https://huggingface.co/datasets/LLinked/sora-watermark-dataset. Free free to train your custom detector model or improve our model!

## 7. API

Packaged as a Cog and [published to Replicate](https://replicate.com/uglyrobot/sora2-watermark-remover) for simple API based usage.

## 8. License

 Apache License


## 9. Citation

If you use this project, please cite:

```bibtex
@misc{sorawatermarkcleaner2025,
  author = {linkedlist771},
  title = {SoraWatermarkCleaner},
  year = {2025},
  url = {https://github.com/linkedlist771/SoraWatermarkCleaner}
}
```

## 10. Acknowledgments

- [IOPaint](https://github.com/Sanster/IOPaint) for the LAMA implementation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
