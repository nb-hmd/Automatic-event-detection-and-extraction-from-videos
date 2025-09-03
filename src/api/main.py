from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import shutil
from pathlib import Path
from ..services.video_processor import VideoProcessor
from ..utils.config import settings

app = FastAPI(title="Video Event Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_processor = VideoProcessor()

class QueryRequest(BaseModel):
    video_id: str
    query: str
    mode: str = "mvp"
    top_k: Optional[int] = None
    threshold: Optional[float] = None

class QueryResponse(BaseModel):
    task_id: str
    status: str
    results: List[dict]
    total_found: int

class VideoUploadResponse(BaseModel):
    video_id: str
    status: str
    filename: str
    path: str
    format: Optional[str] = None
    size: Optional[int] = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Video Event Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/api/upload": "POST - Upload video file",
            "/api/query": "POST - Process event detection query",
            "/api/download/{clip_filename}": "GET - Download extracted clip",
            "/api/health": "GET - Health check"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "video-event-detection"}

@app.post("/api/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload video file."""
    try:
        # Validate file format
        if not file.filename:
            raise HTTPException(400, "No filename provided")
            
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported format: {file_extension}. Supported: {settings.SUPPORTED_FORMATS}")
        
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        video_path = settings.DATA_DIR / "videos" / f"{video_id}.{file_extension}"
        
        # Ensure videos directory exists
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate the saved video
        validation_result = video_processor.validate_video(str(video_path))
        
        if not validation_result['valid']:
            # Clean up invalid file
            video_path.unlink(missing_ok=True)
            raise HTTPException(400, f"Invalid video file: {validation_result['error']}")
        
        return VideoUploadResponse(
            video_id=video_id,
            status="success",
            filename=file.filename,
            path=str(video_path),
            format=validation_result.get('format'),
            size=validation_result.get('size')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process event detection query."""
    try:
        # Find video file
        video_path = None
        for fmt in settings.SUPPORTED_FORMATS:
            potential_path = settings.DATA_DIR / "videos" / f"{request.video_id}.{fmt}"
            if potential_path.exists():
                video_path = potential_path
                break
        
        if not video_path:
            raise HTTPException(404, f"Video not found: {request.video_id}")
        
        # Process query
        result = video_processor.process_query(
            str(video_path),
            request.query,
            request.mode,
            request.top_k,
            request.threshold
        )
        
        if result['status'] == 'error':
            raise HTTPException(500, result['error'])
        
        return QueryResponse(
            task_id=str(uuid.uuid4()),
            status="completed",
            results=result['results'],
            total_found=result['total_found']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Query processing failed: {str(e)}")

@app.get("/api/download/{clip_filename}")
async def download_clip(clip_filename: str):
    """Download extracted clip."""
    try:
        clip_path = settings.DATA_DIR / "clips" / clip_filename
        
        if not clip_path.exists():
            raise HTTPException(404, "Clip not found")
        
        return FileResponse(
            clip_path, 
            media_type="video/mp4",
            filename=clip_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")

@app.get("/api/videos")
async def list_videos():
    """List all uploaded videos."""
    try:
        videos_dir = settings.DATA_DIR / "videos"
        if not videos_dir.exists():
            return {"videos": []}
        
        videos = []
        for video_file in videos_dir.glob("*"):
            if video_file.is_file() and video_file.suffix.lower().lstrip('.') in settings.SUPPORTED_FORMATS:
                videos.append({
                    "video_id": video_file.stem,
                    "filename": video_file.name,
                    "format": video_file.suffix.lower().lstrip('.'),
                    "size": video_file.stat().st_size,
                    "created": video_file.stat().st_ctime
                })
        
        return {"videos": videos}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to list videos: {str(e)}")

@app.get("/api/clips")
async def list_clips():
    """List all extracted clips."""
    try:
        clips_dir = settings.DATA_DIR / "clips"
        if not clips_dir.exists():
            return {"clips": []}
        
        clips = []
        for clip_file in clips_dir.glob("*.mp4"):
            if clip_file.is_file():
                clips.append({
                    "clip_id": clip_file.stem,
                    "filename": clip_file.name,
                    "size": clip_file.stat().st_size,
                    "created": clip_file.stat().st_ctime
                })
        
        return {"clips": clips}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to list clips: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)