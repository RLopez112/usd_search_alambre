import os
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional, List

from embedder import SigLIP2Embedder
from asset_manager import AssetManager

app = FastAPI(
    title="USD Multimodal Vector Search API",
    description="Multimodal vector search microservice for local industrial USD assets.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder: Optional[SigLIP2Embedder] = None
asset_manager: Optional[AssetManager] = None
asset_db = []

@app.on_event("startup")
def startup_event():
    global embedder, asset_manager, asset_db
    print("Initializing SigLIP2 Embedder model & Scanning Assets...")
    embedder = SigLIP2Embedder()
    asset_manager = AssetManager()
    raw_assets = asset_manager.scan_assets()
    
    # Compute vector embeddings for scanned USD assets
    for asset in raw_assets:
        vector = embedder.embed_text(asset["caption"])
        asset_db.append({
            **asset,
            "vector": np.array(vector)
        })
    print(f"Loaded and indexed {len(asset_db)} industrial USD assets into memory vector DB.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "total_assets": len(asset_db)}

@app.get("/api/thumbnail")
def get_thumbnail(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Thumbnail not found")

@app.post("/api/search")
async def search_assets(
    query: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None)
):
    if not query and not image:
        raise HTTPException(status_code=400, detail="Must provide either text query or reference image.")
    
    if image:
        temp_dir = "scratch/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, image.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        query_vec = np.array(embedder.embed_image(temp_file_path))
    else:
        query_vec = np.array(embedder.embed_text(query))
    
    # Compute Cosine Similarity against indexed asset vectors
    results = []
    for asset in asset_db:
        # Vectors are L2 normalized, inner product equals cosine similarity
        score = float(np.dot(query_vec, asset["vector"]))
        results.append({
            "asset_id": asset["asset_id"],
            "usd_path": asset["rel_path"],
            "full_path": asset["usd_path"],
            "thumbnail_url": f"http://127.0.0.1:8000/api/thumbnail?path={asset['thumbnail_path']}" if asset.get("thumbnail_path") else "",
            "category": asset["category"],
            "caption": asset["caption"],
            "score": max(0.0, score)
        })
        
    # Sort by top vector similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "status": "success",
        "query_type": "image" if image else "text",
        "query_text": query,
        "results": results[:18]
    }

