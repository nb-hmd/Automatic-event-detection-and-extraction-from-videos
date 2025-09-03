import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # Video processing - High performance settings
    MAX_VIDEO_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB for complete video processing
    SUPPORTED_FORMATS: list = ["mp4", "avi", "mov", "mkv"]
    FRAME_SAMPLE_RATE: int = 1  # Extract every frame for maximum detail
    WINDOW_SIZE: int = 16  # More frames per window for better context
    WINDOW_STRIDE: int = 8  # Optimal stride for comprehensive coverage
    
    # Frame processing - High quality settings
    MAX_FRAME_WIDTH: int = 512  # Higher resolution for better accuracy
    MAX_FRAME_HEIGHT: int = 512  # Higher resolution for better accuracy
    FRAME_QUALITY: int = 95  # High quality for maximum detail
    MAX_WINDOWS_PER_BATCH: int = 32  # Larger batches for efficient processing
    
    # Model settings
    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"
    BLIP_MODEL: str = "Salesforce/blip-image-captioning-base"  # Smallest BLIP model for limited memory
    UNIVTG_MODEL: str = "univtg_qvhighlights"
    
    # Processing - High performance settings
    BATCH_SIZE: int = 32  # Optimal batch size for maximum throughput
    TOP_K_RESULTS: int = 15  # More results for comprehensive analysis
    CONFIDENCE_THRESHOLD: float = 0.25  # Balanced threshold for quality results
    CLIP_DURATION: int = 30  # Longer clips for complete event coverage
    
    # Memory management
    ENABLE_MEMORY_MONITORING: bool = True
    MIN_AVAILABLE_MEMORY_MB: int = 100  # Minimum memory before fallback
    MEMORY_CLEANUP_INTERVAL: int = 5  # Clean memory every N windows
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Redis (for Celery)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"

settings = Settings()