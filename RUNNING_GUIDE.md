# User Execution & Deployment Manual: USD Multimodal Vector Search

This guide provides step-by-step instructions on how to run, test, and use the **USD Multimodal Vector Search Engine & Explorer Web UI** on your local machine.

---

## Architecture Overview

The system runs locally within a Python virtual environment (`.venv`):

- **FastAPI Service (`main.py`)**: Runs on `http://localhost:8000` and handles multimodal SigLIP2 vector encoding and search requests.
- **Explorer Web UI (`index.html`)**: A lightweight visual interface allowing natural language text queries and reference image dropzone file uploads.
- **SigLIP2 Vector Model (`embedder.py`)**: Generates 1152-dimensional L2-normalized embeddings via HuggingFace `google/siglip2-so400m-patch14-384`.
- **OpenUSD Renderer (`usd_renderer.py`)**: Inspects 3D `.usd` stage bounding boxes and renders snapshot previews.

---

## Step 1: Activate Virtual Environment

Open your terminal in the root workspace directory (`d:\___tmp\usd_search_alambre`) and activate the virtual environment:

### PowerShell (Windows):
```powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt (CMD):
```cmd
.\.venv\Scripts\activate.bat
```

---

## Step 2: Start the FastAPI Backend Server

Run the FastAPI backend server using Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

You should see startup logs indicating model loading:
```text
Initializing SigLIP2 Embedder model...
Loading SigLIP2 model 'google/siglip2-so400m-patch14-384' on device: cpu
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Step 3: Open & Use the Explorer Web UI

1. Open `index.html` directly in any web browser, or serve it using Python's simple HTTP server:
   ```powershell
   .\.venv\Scripts\python.exe -m http.server 8080
   ```
2. Navigate to:
   ```text
   http://localhost:8080
   ```
3. **Run a Search Query**:
   - **Text Prompt**: Type e.g., `"industrial robot arm"` or `"heavy duty storage shelf"` and click **Search**.
   - **Image Reference**: Click **Ref Image**, pick a reference image file, and click **Search** to execute visual vector similarity search.

---

## Step 4: Access API Documentation (OpenAPI / Swagger)

With the server running on port 8000, inspect interactive REST API documentation at:

```text
http://localhost:8000/docs
```

---

## Quick Testing Commands via Terminal

### Health Check Endpoint:
```powershell
curl http://127.0.0.1:8000/health
```

### Multimodal Search API Test (Text Query):
```powershell
curl -X POST "http://127.0.0.1:8000/api/search" -F "query=robotic manipulator arm"
```

### Test SigLIP2 Vector Generator Directly:
```powershell
.\.venv\Scripts\python.exe embedder.py
```
