import cv2
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from app.storage.file_store import get_session_dir

def mask_to_colored_overlay(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    color_rgb: tuple = (0, 200, 255),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Applies alpha-blended color mask with a crisp border contour to an image.
    """
    img_rgb = image_np.copy()
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb]*3, axis=-1)
    elif img_rgb.shape[2] > 3:
        img_rgb = img_rgb[:, :, :3]
        
    overlay = img_rgb.copy()
    mask_bool = mask_np > 0
    overlay[mask_bool] = color_rgb
    
    # Alpha blend
    blended = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    
    # Add crisp contour boundary
    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color_rgb, 2)
    
    return blended

def extract_evidence_regions(mask_np: np.ndarray, label: str, min_area: int = 16) -> list[dict]:
    """
    Extracts genuine bounding boxes and geometries from binary mask using connected components.
    Never fabricates boxes.
    """
    regions = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), connectivity=8)
    
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        
        regions.append({
            "type": "bounding_box",
            "label": label,
            "bbox": [y, x, y + h, x + w],  # [ymin, xmin, ymax, xmax]
            "area_pixels": area,
            "centroid_pixel": [cx, cy]
        })
    return regions

def render_bounding_boxes(
    image_np: np.ndarray,
    regions: list[dict],
    color_rgb: tuple = (255, 180, 0)
) -> np.ndarray:
    """Draws real detected bounding boxes and labels onto the image."""
    annotated = image_np.copy()
    if annotated.ndim == 2:
        annotated = np.stack([annotated]*3, axis=-1)
    elif annotated.shape[2] > 3:
        annotated = annotated[:, :, :3]
        
    for r in regions:
        ymin, xmin, ymax, xmax = r["bbox"]
        label = r.get("label", "region")
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color_rgb, 2)
        # Label banner
        text = f"{label} ({r.get('area_pixels', 0)}px)"
        cv2.putText(annotated, text, (xmin, max(15, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    return annotated

def generate_side_by_side(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray = None) -> np.ndarray:
    """Combines images horizontally with clean separation borders."""
    h = max(img1.shape[0], img2.shape[0], (img3.shape[0] if img3 is not None else 0))
    
    def prep(img):
        res = img.copy()
        if res.ndim == 2:
            res = np.stack([res]*3, axis=-1)
        elif res.shape[2] > 3:
            res = res[:, :, :3]
        if res.shape[0] != h:
            w_new = int(res.shape[1] * (h / res.shape[0]))
            res = cv2.resize(res, (w_new, h))
        return res

    p1 = prep(img1)
    p2 = prep(img2)
    border = np.zeros((h, 4, 3), dtype=np.uint8) + 80
    
    if img3 is not None:
        p3 = prep(img3)
        return np.hstack([p1, border, p2, border, p3])
    return np.hstack([p1, border, p2])

def save_evidence(analysis_id: str, filename: str, image_np: np.ndarray) -> str:
    """Saves image to overlays runtime folder and returns relative web path."""
    overlay_dir = get_session_dir(analysis_id, "overlays")
    out_path = overlay_dir / filename
    
    if image_np.ndim == 2:
        img_to_save = Image.fromarray(image_np.astype(np.uint8))
    else:
        img_to_save = Image.fromarray(image_np[:, :, :3].astype(np.uint8))
        
    img_to_save.save(out_path, format="PNG")
    return f"/runtime/overlays/{analysis_id}/{filename}"
