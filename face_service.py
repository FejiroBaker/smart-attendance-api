# python_backend/face_service.py

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import numpy as np
from PIL import Image
import io
from typing import Optional, List, Tuple
from models import DetectedFace, StoredEmbedding

RECOGNITION_THRESHOLD = 0.5   # cosine distance; lower = stricter match

# InsightFace app — loaded once at module level
_face_app = None


def _get_face_app():
    """Lazy-load InsightFace so import errors surface clearly."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_sc",                    # small but accurate model (~85 MB)
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace model loaded.")
    return _face_app


def warmup():
    """Call at startup to pre-load the model and avoid first-request lag."""
    try:
        app = _get_face_app()
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        app.get(dummy)
        print("InsightFace warmup complete.")
    except Exception as e:
        print(f"InsightFace warmup warning: {e}")


# ─────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────

def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode JPEG/PNG bytes to BGR numpy array (OpenCV convention)."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # InsightFace expects BGR
        rgb = np.array(pil_image)
        return rgb[:, :, ::-1].copy()   # RGB to BGR
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def detect_faces(image_bytes: bytes) -> Tuple[List[DetectedFace], Optional[np.ndarray]]:
    """
    Detect all faces in the image.
    Returns (list of DetectedFace, bgr image array).
    """
    image_array = decode_image(image_bytes)
    if image_array is None:
        return [], None

    try:
        faces = _get_face_app().get(image_array)
        h, w = image_array.shape[:2]
        detected = []

        for face in faces:
            score = float(face.det_score)
            if score < 0.4:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            detected.append(DetectedFace(
                left=float(x1),
                top=float(y1),
                right=float(x2),
                bottom=float(y2),
                confidence=round(score, 4),
            ))

        return detected, image_array
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return [], image_array


def register_face(image_bytes: bytes, student_id: int, student_name: str) -> dict:
    """
    Detect face and extract embedding for registration.
    Does NOT write to the database — the caller (main.py) does that.
    """
    image_array = decode_image(image_bytes)
    if image_array is None:
        return {"success": False, "message": "Failed to decode image"}

    try:
        faces = _get_face_app().get(image_array)
        valid = [f for f in faces if float(f.det_score) >= 0.5]

        if not valid:
            return {"success": False, "message": "No face detected. Face the camera directly."}
        if len(valid) > 1:
            return {"success": False, "message": f"Multiple faces detected ({len(valid)}). One person only."}

        face = valid[0]
        quality = _estimate_quality(image_array, face.bbox.astype(int))

        if quality < 0.3:
            return {"success": False, "message": "Face quality too low. Improve lighting."}

        embedding: List[float] = face.embedding.tolist()

        return {
            "success": True,
            "message": "Face detected and embedding extracted",
            "embedding": embedding,
            "quality": round(quality, 4),
            "embedding_size": len(embedding),
        }
    except Exception as e:
        print(f"Register face error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


def recognize_face(
    image_bytes: bytes,
    stored_embeddings: List[StoredEmbedding],
    exclude_student_ids: List[int] = None,
) -> dict:
    """
    Recognize the face in the image against all stored embeddings.
    Returns matched student info if found.
    """
    if not stored_embeddings:
        return {"success": False, "message": "No registered faces in database"}

    image_array = decode_image(image_bytes)
    if image_array is None:
        return {"success": False, "message": "Failed to decode image"}

    try:
        faces = _get_face_app().get(image_array)
        valid = [f for f in faces if float(f.det_score) >= 0.5]

        if not valid:
            return {"success": False, "message": "No face detected"}
        if len(valid) > 1:
            return {"success": False, "message": "Multiple faces detected. One at a time."}

        face = valid[0]
        quality = _estimate_quality(image_array, face.bbox.astype(int))

        if quality < 0.25:
            return {"success": False, "message": "Face not clear enough. Adjust lighting."}

        query_vec = face.embedding   # already L2-normalised by InsightFace

        candidates = stored_embeddings
        if exclude_student_ids:
            candidates = [e for e in stored_embeddings if e.student_id not in exclude_student_ids]

        if not candidates:
            return {"success": False, "message": "All registered students already marked"}

        best_distance = float("inf")
        best_match = None

        for stored in candidates:
            dist = _cosine_distance(query_vec, np.array(stored.embedding, dtype=np.float32))
            if dist < best_distance:
                best_distance = dist
                best_match = stored

        confidence = round(max(0.0, 1.0 - best_distance), 4)

        if best_distance <= RECOGNITION_THRESHOLD and best_match is not None:
            return {
                "success": True,
                "message": "Face recognized",
                "matched_student_id": best_match.student_id,
                "matched_student_name": best_match.student_name,
                "confidence": confidence,
                "face_quality": round(quality, 4),
            }
        else:
            return {
                "success": False,
                "message": "Face not recognized",
                "confidence": confidence,
                "face_quality": round(quality, 4),
            }
    except Exception as e:
        print(f"Recognize face error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine distance between two vectors (0 = identical, 1 = opposite)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    return float(1.0 - np.dot(v1, v2) / (n1 * n2))


def _estimate_quality(bgr: np.ndarray, bbox: np.ndarray) -> float:
    """
    Heuristic face quality score in [0, 1] based on size, sharpness, brightness.
    Pure numpy — no cv2 needed.
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
    face_region = bgr[y1:y2, x1:x2]

    if face_region.size == 0:
        return 0.0

    h, w = face_region.shape[:2]
    size_score = min(1.0, (w * h) / (80 * 80))

    gray = np.mean(face_region, axis=2).astype(float)
    lap = (
        np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=0) +
        np.roll(gray, -1, axis=1) + np.roll(gray, 1, axis=1) -
        4 * gray
    )
    sharpness_score = min(1.0, float(np.var(lap)) / 500.0)

    mean_brightness = float(np.mean(gray))
    if 60 <= mean_brightness <= 200:
        brightness_score = 1.0
    elif mean_brightness < 60:
        brightness_score = mean_brightness / 60.0
    else:
        brightness_score = max(0.0, 1.0 - (mean_brightness - 200) / 55.0)

    return round(min(1.0, size_score * 0.3 + sharpness_score * 0.5 + brightness_score * 0.2), 4)