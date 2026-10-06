import os
import glob
from embedder import SigLIP2Embedder
from usd_renderer import render_usd_thumbnail

ASSETS_DIR = r"D:\___tmp\Industrial_NVD@10012\Assets\ArchVis\Industrial"

class AssetManager:
    """Manages indexing and vector search over local industrial USD assets."""
    def __init__(self, base_dir: str = ASSETS_DIR):
        self.base_dir = base_dir
        self.assets = []
        self.indexed = False

    def scan_assets(self):
        """Scans directory for USD assets and extracts captions/paths and thumbnails."""
        self.assets = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith(('.usd', '.usda', '.usdc', '.usdz')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.base_dir)
                    
                    # 1. Look for pre-existing Omniverse thumbnail in .thumbs folder
                    thumb_dir = os.path.join(root, ".thumbs", "256x256")
                    possible_thumb = os.path.join(thumb_dir, f"{file}.png")
                    
                    if os.path.exists(possible_thumb):
                        thumbnail_path = possible_thumb
                    else:
                        # 2. Render thumbnail snapshot using usd_renderer
                        os.makedirs("scratch/generated_thumbs", exist_ok=True)
                        gen_thumb = os.path.join("scratch/generated_thumbs", f"{file}.png")
                        try:
                            render_usd_thumbnail(full_path, gen_thumb)
                            thumbnail_path = gen_thumb
                        except Exception:
                            thumbnail_path = ""

                    parts = rel_path.replace("\\", "/").split("/")
                    category = parts[0] if len(parts) > 1 else "Industrial"
                    subcategory = parts[1] if len(parts) > 2 else ""
                    name = os.path.splitext(file)[0].replace("_", " ").replace("-", " ")
                    
                    caption = f"Industrial 3D asset: {category} {subcategory} {name}".strip()
                    
                    self.assets.append({
                        "asset_id": rel_path.replace("\\", "_").replace("/", "_"),
                        "usd_path": full_path,
                        "rel_path": rel_path,
                        "thumbnail_path": thumbnail_path,
                        "category": category,
                        "caption": caption
                    })
        print(f"Scanned {len(self.assets)} USD industrial assets with thumbnails from {self.base_dir}")
        return self.assets

if __name__ == "__main__":
    manager = AssetManager()
    assets = manager.scan_assets()
    print("Sample asset:", assets[0])
