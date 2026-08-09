# Movie Review Tool

A desktop AI-powered application for automatic movie review generation.

## Features
- Video ingestion and thumbnail generation
- Multithreaded processing (non‑blocking UI)
- Configurable via `config/config.json`
- Logging with Loguru
- Built with Python 3.12, CustomTkinter, FFmpeg, OpenCV, PySceneDetect, Faster Whisper, Gemini API, SQLite, Requests.

## Project Structure
```
.
├─ .env.example         # Environment variable template
├─ config/
│   └─ config.json      # Default configuration
├─ src/
│   ├─ core/
│   │   └─ video_loader.py
│   └─ utils/
│       ├─ logger.py
│       └─ thread_pool.py
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your API keys.
3. Run the application (to be implemented).

---
*This repository follows the design specifications defined in the architecture phase.*
