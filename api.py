import os
import shutil
import uuid
import json
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import run_inference

app = FastAPI(
    title="Mudra Analysis API",
    description="API for analyzing Bharatanatyam dance videos for Mudras and Steps.",
    version="1.1.0"
)

# Configuration
BASE_DATA_DIR = Path("data/jobs")
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
RESULTS_DIR = BASE_DATA_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory status tracking (will be lost on restart, but we have file persistence for results)
# Statuses: "pending", "processing", "completed", "failed"
job_status: Dict[str, str] = {}

def process_video_task(task_id: str, temp_video_path: Path, output_json_path: Path, use_mudra: bool):
    """Background task to run inference and update status."""
    global job_status
    job_status[task_id] = "processing"
    
    try:
        logger.info(f"Task {task_id}: Starting inference...")
        run_inference(str(temp_video_path), str(output_json_path), use_mudra_model=use_mudra)
        
        if output_json_path.exists():
            job_status[task_id] = "completed"
            logger.info(f"Task {task_id}: Inference completed successfully.")
        else:
            job_status[task_id] = "failed"
            logger.error(f"Task {task_id}: Inference failed to output results.")
            
    except Exception as e:
        job_status[task_id] = "failed"
        logger.error(f"Task {task_id}: Critical error during processing: {str(e)}")
        
    finally:
        # Cleanup video file but keep JSON result
        if temp_video_path.exists():
            temp_video_path.unlink()

@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"status": "online", "message": "Mudra Analysis API is active.", "docs": "/docs"}

@app.get("/status")
async def get_total_status():
    """Check the health of the API."""
    return {"status": "healthy", "jobs_tracked": len(job_status)}

@app.post("/analyze")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), use_mudra: bool = True):
    """
    Upload a video file for background analysis.
    Returns a task_id immediately.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    task_id = str(uuid.uuid4())
    temp_video_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    output_json_path = RESULTS_DIR / f"{task_id}_result.json"

    try:
        # Save uploaded file
        with temp_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        job_status[task_id] = "pending"
        
        # Queue the background task
        background_tasks.add_task(
            process_video_task, 
            task_id, 
            temp_video_path, 
            output_json_path, 
            use_mudra
        )

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Video uploaded and queued for processing."
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """
    Get the result or status of a specific task.
    """
    status = job_status.get(task_id)
    
    # Check if result file exists (even if not in memory)
    output_json_path = RESULTS_DIR / f"{task_id}_result.json"
    
    if output_json_path.exists():
        with output_json_path.open("r") as f:
            result_data = json.load(f)
        return {
            "task_id": task_id,
            "status": "completed",
            "data": result_data
        }
    
    if status:
        return {
            "task_id": task_id,
            "status": status,
            "message": "Job is still in progress or failed." if status != "completed" else "Completed"
        }
    
    raise HTTPException(status_code=404, detail="Task ID not found.")

if __name__ == "__main__":
    import uvicorn
    # Respect PORT env var for Render deployment
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
