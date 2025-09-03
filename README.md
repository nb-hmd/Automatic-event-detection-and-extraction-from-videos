# 🎥 Automatic Video Event Detection and Extraction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent AI-powered video analysis system that automatically detects and extracts specific events from long-form videos using natural language queries. The system eliminates the need for manual video review by providing precise timestamps and extracted clips of relevant events.

![Project UI](assets/ui_screenshot.png)
*Web interface showing video upload and natural language event query functionality*

## 🌟 Key Features

- **🗣️ Natural Language Queries**: Describe events in plain English ("Two cars hit a man", "Person wearing red shirt")
- **🤖 Multi-Phase AI Processing**: Three processing modes with different accuracy/speed trade-offs
- **🎯 State-of-the-Art AI Models**: Integrates OpenCLIP, BLIP-2, and UniVTG for maximum accuracy
- **✂️ Automatic Clip Extraction**: Generates precise video clips of detected events with timestamps
- **🌐 Web Interface**: User-friendly Streamlit interface for easy interaction
- **🔌 REST API**: FastAPI backend for programmatic access and integration
- **📹 Multiple Video Formats**: Supports MP4, AVI, MOV, MKV files
- **⚡ Memory Optimization**: Efficient processing for large video files
- **🔍 Debug Mode**: Detailed analysis and frame-by-frame debugging

## 🏗️ Architecture

### Processing Phases

#### Phase 1 (MVP) - Fast Detection
- **Technology**: OpenCLIP for image-text similarity
- **Speed**: ⚡ Fastest processing
- **Use Case**: Quick candidate identification
- **Process**: Frame extraction → Sliding windows → CLIP similarity matching

#### Phase 2 (Re-ranked) - Enhanced Accuracy
- **Technology**: OpenCLIP + BLIP-2 image captioning
- **Speed**: ⚖️ Balanced accuracy/speed
- **Use Case**: Production-ready results
- **Process**: Phase 1 + Image captioning → Semantic similarity scoring → Combined ranking

#### Phase 3 (Advanced) - Maximum Precision
- **Technology**: Full pipeline with UniVTG
- **Speed**: 🎯 Most accurate but slower
- **Use Case**: Critical applications requiring highest accuracy
- **Process**: Phase 2 + Temporal boundary refinement → Precise event localization

### System Components

```
video_event_detection/
├── src/
│   ├── models/          # AI model wrappers (OpenCLIP, BLIP-2, UniVTG)
│   ├── services/        # Core processing services
│   ├── pipeline/        # Three-phase processing pipeline
│   ├── api/            # FastAPI backend
│   ├── web/            # Streamlit web interface
│   └── utils/          # Configuration and utilities
├── data/               # Video storage and outputs
├── models/             # Pre-trained model cache
├── assets/             # Documentation assets
└── requirements.txt    # Python dependencies
```

## 🚀 Technology Stack

### AI/ML Framework
- **PyTorch**: Deep learning framework
- **OpenCLIP**: Vision-language model for image-text similarity
- **BLIP-2**: Advanced image captioning model
- **UniVTG**: Video temporal grounding model
- **Transformers**: Hugging Face model library
- **FAISS**: Vector similarity search

### Computer Vision
- **OpenCV**: Video processing and frame extraction
- **FFmpeg**: Video format handling and conversion
- **Pillow**: Image processing utilities

### Web Framework
- **Streamlit**: Interactive web interface
- **FastAPI**: High-performance REST API
- **Uvicorn**: ASGI server

### Backend & Processing
- **Python 3.8+**: Core programming language
- **Celery**: Async task processing
- **Redis**: Task queue and caching
- **SQLAlchemy**: Database ORM

## 📋 Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended for optimal performance)
- **FFmpeg** (for video processing)
- **8GB+ RAM** (16GB+ recommended for large videos)
- **10GB+ free disk space** (for model cache and video processing)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nb-hmd/Automatic-Event-Detection-and-Extraction-from-Videos.git
cd "Automatic Event Detection and Extraction from Video"
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

**Windows:**
1. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your PATH

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### 5. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
ffmpeg -version
```

## 🚀 Quick Start

### Web Interface (Recommended)

#### Windows
```bash
# Using PowerShell script (recommended)
.\start_server.ps1

# Or manually
streamlit run src/web/streamlit_app.py
```

#### Linux/Mac
```bash
# Using Python script
python start_server.py

# Or manually
streamlit run src/web/streamlit_app.py
```

Open your browser to **http://localhost:8501**

### API Server

```bash
python -m src.api.main
```

API documentation available at **http://localhost:8000/docs**

## 📖 Usage Examples

### Web Interface Usage

1. **📁 Upload Video**: Drag and drop or select a video file (MP4, AVI, MOV, MKV)
2. **🔍 Enter Query**: Describe the event you want to find:
   - "Two cars colliding"
   - "A person wearing a red shirt"
   - "Someone falling down"
   - "People shaking hands"
3. **⚙️ Select Mode**: Choose processing mode (MVP/Reranked/Advanced)
4. **🎯 Adjust Settings**: Set confidence threshold and max results
5. **🚀 Process**: Click "Find Events" to start detection
6. **📊 Review Results**: View timestamps, confidence scores, and download clips

### API Usage Examples

#### Upload Video
```python
import requests

with open("sample_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload",
        files={"file": f}
    )
video_id = response.json()["video_id"]
print(f"Video uploaded with ID: {video_id}")
```

#### Query for Events
```python
query_data = {
    "video_id": video_id,
    "query": "A person walking a dog",
    "mode": "reranked",  # mvp, reranked, or advanced
    "top_k": 5,
    "threshold": 0.4
}

response = requests.post(
    "http://localhost:8000/api/query",
    json=query_data
)

results = response.json()
print(f"Found {results['total_found']} events")

for i, result in enumerate(results['results']):
    print(f"{i+1}. Timestamp: {result['timestamp']:.2f}s, Confidence: {result['confidence']:.3f}")
```

#### Download Generated Clips
```python
for result in results['results']:
    if 'clip_path' in result:
        clip_filename = result['clip_path'].split('/')[-1]
        clip_response = requests.get(
            f"http://localhost:8000/api/download/{clip_filename}"
        )
        
        with open(f"downloaded_{clip_filename}", "wb") as f:
            f.write(clip_response.content)
        print(f"Downloaded: {clip_filename}")
```

### Direct Python Usage

```python
from src.services.video_processor import VideoProcessor

# Initialize processor
processor = VideoProcessor()

# Process video with different modes
video_path = "path/to/your/video.mp4"
query = "A person walking a dog"

# MVP mode (fastest)
results_mvp = processor.process_query(
    video_path, query, 
    mode="mvp", 
    top_k=10, 
    threshold=0.3
)

# Re-ranked mode (balanced)
results_reranked = processor.process_query(
    video_path, query, 
    mode="reranked", 
    top_k=5, 
    threshold=0.4
)

# Advanced mode (most accurate)
results_advanced = processor.process_query(
    video_path, query, 
    mode="advanced", 
    top_k=3, 
    threshold=0.5
)

# Print results
for result in results_reranked['results']:
    print(f"Found event at {result['timestamp']:.2f}s with confidence {result['confidence']:.3f}")
```

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

### Video Processing Settings
```python
# Frame extraction
FRAME_SAMPLE_RATE = 1.0  # Extract every N seconds
WINDOW_SIZE = 5          # Frames per sliding window
WINDOW_STRIDE = 2        # Window overlap

# Supported formats
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
```

### Model Settings
```python
# Model configurations
CLIP_MODEL_NAME = "ViT-B/32"
BLIP_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.3
```

### API Settings
```python
# Server configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
STREAMLIT_PORT = 8501

# CORS settings
ALLOWED_ORIGINS = ["*"]
```

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Cache
MODEL_CACHE_DIR=./models

# Data Directories
DATA_DIR=./data
VIDEO_DIR=./data/videos
CLIP_DIR=./data/clips

# Processing
MAX_VIDEO_SIZE_MB=2048
MAX_PROCESSING_TIME_MINUTES=30

# Debug
DEBUG_MODE=false
LOG_LEVEL=INFO
```

## 💡 Use Cases

### 🔒 Security & Surveillance
- Automated incident detection in security footage
- Suspicious activity identification
- Crowd behavior analysis
- Perimeter breach detection

### 🏃 Sports & Entertainment
- Highlight reel generation
- Specific play detection
- Player performance analysis
- Event moment extraction

### 📚 Education & Training
- Lecture content indexing
- Training video analysis
- Skill assessment
- Educational content curation

### 🏢 Business & Marketing
- Content moderation
- Brand mention detection
- Customer behavior analysis
- Marketing campaign analysis

## 🔧 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce batch size in config.py
BATCH_SIZE = 4  # Default: 8
MAX_WINDOWS_PER_BATCH = 10  # Default: 20
```

#### FFmpeg Not Found
```bash
# Windows: Add to PATH or install via chocolatey
choco install ffmpeg

# Linux: Install via package manager
sudo apt install ffmpeg

# Mac: Install via Homebrew
brew install ffmpeg
```

#### CUDA Out of Memory
```python
# Enable CPU-only mode in config.py
FORCE_CPU = True
DEVICE = "cpu"
```

#### Model Download Issues
```bash
# Clear model cache and retry
rm -rf models/
python -c "from src.models.openclip_model import OpenCLIPModel; OpenCLIPModel()"
```

### Performance Optimization

#### For Large Videos (>1GB)
- Use MVP mode for initial screening
- Increase `FRAME_SAMPLE_RATE` to 2.0 or higher
- Process in smaller segments
- Enable debug mode to monitor memory usage

#### For Real-time Processing
- Use GPU acceleration
- Optimize `WINDOW_SIZE` and `WINDOW_STRIDE`
- Consider using the API for batch processing

## 📊 Performance Benchmarks

| Mode | Speed | Accuracy | Memory Usage | Best For |
|------|-------|----------|--------------|----------|
| MVP | ⚡⚡⚡ | ⭐⭐⭐ | Low | Quick screening |
| Reranked | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Production use |
| Advanced | ⚡ | ⭐⭐⭐⭐⭐ | High | Critical applications |

*Benchmarks on RTX 3080, 32GB RAM, 10-minute 1080p video*

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/nb-hmd/Automatic-Event-Detection-and-Extraction-from-Videos.git
cd "Automatic Event Detection and Extraction from Video"
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Running Tests

```bash
pytest tests/
python -m pytest tests/ --cov=src
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** - Vision-language understanding
- **Salesforce BLIP-2** - Advanced image captioning
- **UniVTG** - Video temporal grounding
- **Streamlit** - Web interface framework
- **FastAPI** - High-performance API framework


## 🔮 Future Roadmap

- [ ] Real-time video stream processing
- [ ] Multi-language query support
- [ ] Advanced temporal relationship detection
- [ ] Cloud deployment templates
- [ ] Mobile app interface
- [ ] Integration with popular video platforms

---

*Star ⭐ this repository if you find it helpful!*
