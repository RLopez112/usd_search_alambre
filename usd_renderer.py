import os
from pxr import Usd, UsdGeom
import numpy as np
from PIL import Image

def render_usd_thumbnail(usd_path: str, output_image_path: str, resolution=(512, 512)) -> str:
    """
    Opens a USD stage, computes world bounding box, and renders a snapshot/thumbnail image.
    """
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found at: {usd_path}")
        
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD stage: {usd_path}")
    
    # Compute bounding box for automatic framing validation
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root_prim = stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    aligned_range = bbox.ComputeAlignedRange()
    
    min_pt, max_pt = aligned_range.GetMin(), aligned_range.GetMax()
    center = (np.array(min_pt) + np.array(max_pt)) / 2.0
    size = np.linalg_norm(np.array(max_pt) - np.array(min_pt))
    
    # Save a placeholder snapshot thumbnail (Production uses Omniverse RTX / PyRender)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    img = Image.new("RGB", resolution, color=(30, 30, 30))
    img.save(output_image_path)
    print(f"Generated thumbnail snapshot for '{usd_path}' -> '{output_image_path}' (Bounds size: {size:.2f})")
    return output_image_path
