import os
import shutil
import uuid
import json
import logging
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pipeline components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import run_inference, StepPredictor
from src.extraction import FeatureExtractor

app = FastAPI(
    title="Mudra Analysis API",
    description="API for analyzing Bharatanatyam dance videos with memory optimization.",
    version="1.2.0"
)

# Configuration
BASE_DATA_DIR = Path("data/jobs")
UPLOAD_DIR = BASE_DATA_DIR / "uploads"
RESULTS_DIR = BASE_DATA_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global State
job_status: Dict[str, str] = {}
processing_lock = threading.Lock()

# Singleton Model Instances (Lazy Loaded)
_predictor: Optional[StepPredictor] = None
_extractor: Optional[FeatureExtractor] = None

def log_memory_usage(context: str):
    """Log current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"MEMORY [{context}]: {mem_info.rss / (1024 * 1024):.2f} MB RSS")

def get_models():
    """Get or initialize singleton model instances."""
    global _predictor, _extractor
    if _predictor is None or _extractor is None:
        logger.info("Initializing models for the first time (Lazy Load)...")
        log_memory_usage("BEFORE MODEL LOAD")
        if _extractor is None:
            _extractor = FeatureExtractor(use_static_image_mode=False)
        if _predictor is None:
            _predictor = StepPredictor(use_mudra_model=True, extractor=_extractor)
        log_memory_usage("AFTER MODEL LOAD")
    return _predictor, _extractor

def process_video_task(task_id: str, temp_video_path: Path, output_json_path: Path, use_mudra: bool):
    """Background task with concurrency lock and singleton models."""
    global job_status
    
    # Check if another job is running
    if processing_lock.locked():
        logger.info(f"Task {task_id}: Waiting for processing lock...")
        job_status[task_id] = "waiting"
    
    with processing_lock:
        job_status[task_id] = "processing"
        logger.info(f"Task {task_id}: Starting inference (Lock Acquired)...")
        
        try:
            log_memory_usage(f"TASK START {task_id}")
            
            # Use pre-loaded singletons
            predictor, extractor = get_models()
            
            run_inference(
                str(temp_video_path), 
                str(output_json_path), 
                use_mudra_model=use_mudra,
                predictor=predictor,
                extractor=extractor
            )
            
            if output_json_path.exists():
                job_status[task_id] = "completed"
                logger.info(f"Task {task_id}: Completed.")
            else:
                job_status[task_id] = "failed"
                logger.error(f"Task {task_id}: Failed to generate output file.")
                
        except Exception as e:
            job_status[task_id] = "failed"
            logger.error(f"Task {task_id}: Error: {str(e)}", exc_info=True)
            
        finally:
            log_memory_usage(f"TASK END {task_id}")
            if temp_video_path.exists():
                temp_video_path.unlink()

@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"status": "online", "message": "Mudra Analysis API is active.", "docs": "/docs"}

@app.get("/status")
async def get_total_status():
    """Check health and memory usage."""
    process = psutil.Process(os.getpid())
    return {
        "status": "healthy",
        "jobs_tracked": len(job_status),
        "app_memory_mb": process.memory_info().rss / (1024 * 1024),
        "lock_active": processing_lock.locked()
    }

@app.post("/analyze")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), use_mudra: bool = True):
    """Upload a video for background processing."""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    task_id = str(uuid.uuid4())
    temp_video_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    output_json_path = RESULTS_DIR / f"{task_id}_result.json"

    try:
        with temp_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        job_status[task_id] = "pending"
        background_tasks.add_task(process_video_task, task_id, temp_video_path, output_json_path, use_mudra)

        return {"task_id": task_id, "status": "pending"}

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Retrieve job status or final result."""
    status = job_status.get(task_id)
    output_json_path = RESULTS_DIR / f"{task_id}_result.json"
    
    if output_json_path.exists():
        with output_json_path.open("r") as f:
            return {"task_id": task_id, "status": "completed", "data": json.load(f)}
    
    if status:
        return {"task_id": task_id, "status": status}
    
    raise HTTPException(status_code=404, detail="Task ID not found.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
