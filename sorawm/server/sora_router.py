from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from sorawm.server.sora_scraper import sora_scraper
from sorawm.server.worker import worker
from sorawm.schemas import CleanerType
from uuid import uuid4
import shutil
import os

router = APIRouter()

class SoraDownloadRequest(BaseModel):
    url: str

@router.post("/login")
async def login_to_sora():
    """Launches the browser for the user to login."""
    try:
        await sora_scraper.login()
        return {"status": "success", "message": "Browser opened for login. Please sign in to Sora."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download")
async def download_sora_video(request: SoraDownloadRequest):
    """Downloads a video from a Sora URL and adds it to the watermark removal queue."""
    try:
        # Create a temporary path for the download
        task_uuid = str(uuid4())
        upload_filename = f"{task_uuid}_sora_download.mp4"
        video_path = worker.upload_dir / upload_filename

        success = await sora_scraper.download_video(request.url, video_path)

        if not success:
             raise HTTPException(status_code=400, detail="Failed to download video. Please check the URL or your login status.")

        # Add to the worker queue automatically
        # Create task in DB
        task_id = await worker.create_task(CleanerType.LAMA) # Default to LAMA for now

        # Queue it
        await worker.queue_task(task_id, video_path)

        return {
            "status": "success",
            "message": "Video downloaded and queued for processing",
            "task_id": task_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
