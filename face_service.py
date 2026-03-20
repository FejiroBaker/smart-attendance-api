# python_backend/face_service.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

import numpy as np
from PIL import Image
import io
from typing import Optional, List, Tuple
from deepface import DeepFace
from models import DetectedFace, StoredEmbedding

RECOGNITION_THRESHOLD = 0.4
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to numpy array (RGB)."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return np.array(pil_image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def _get_cv2():
    """Lazy import cv2 to avoid startup crash."""
    import cv2
    return cv2


def detect_faces(image_bytes: bytes) -> Tuple[List[DetectedFace], Optional[np.ndarray]]:
    """Detect all faces in an image."""
    image_array = decode_image(image_bytes)
    if image_array is None:
        return [], None

    try:
        faces = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )

        detected = []
        h, w = image_array.shape[:2]

        for face in faces:
            if face.get("confidence", 0) < 0.5:
                continue
            region = face.get("facial_area", {})
            left = max(0, region.get("x", 0))
            top = max(0, region.get("y", 0))
            width = region.get("w", 0)
            height = region.get("h", 0)
            right = min(w, left + width)
            bottom = min(h, top + height)
            face_region = image_array[top:bottom, left:right]
            quality = _estimate_face_quality(face_region)
            detected.append(DetectedFace(
                left=float(left), top=float(top),
                right=float(right), bottom=float(bottom),
                confidence=round(quality, 4),
            ))

        return detected, image_array
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return [], image_array


def extract_embedding(image_array: np.ndarray) -> Optional[List[float]]:
    """Extract face embedding using DeepFace + Facenet."""
    try:
        result = DeepFace.represent(
            img_path=image_array,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
        if result and len(result) > 0:
            return result[0]["embedding"]
        return None
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


def register_face(image_bytes: bytes, student_id: int, student_name: str) -> dict:
    """Detect face and extract embedding for registration."""
    image_array = decode_image(image_bytes)
    if image_array is None:
        return {"success": False, "message": "Failed to decode image"}

    try:
        faces = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
        valid_faces = [f for f in faces if f.get("confidence", 0) >= 0.5]

        if not valid_faces:
            return {"success": False, "message": "No face detected. Please face the camera directly."}
        if len(valid_faces) > 1:
            return {"success": False, "message": f"Multiple faces detected ({len(valid_faces)}). Only one person allowed."}

        face = valid_faces[0]
        region = face.get("facial_area", {})
        left, top = region.get("x", 0), region.get("y", 0)
        w, h = region.get("w", 0), region.get("h", 0)
        face_region = image_array[top:top+h, left:left+w]
        quality = _estimate_face_quality(face_region)

        if quality < 0.35:
            return {"success": False, "message": "Face quality too low. Ensure good lighting."}

        embedding = extract_embedding(image_array)
        if embedding is None:
            return {"success": False, "message": "Failed to extract face features"}

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
    """Recognize a face against stored embeddings."""
    if not stored_embeddings:
        return {"success": False, "message": "No registered faces in database"}

    image_array = decode_image(image_bytes)
    if image_array is None:
        return {"success": False, "message": "Failed to decode image"}

    try:
        faces = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
        valid_faces = [f for f in faces if f.get("confidence", 0) >= 0.5]

        if not valid_faces:
            return {"success": False, "message": "No face detected"}
        if len(valid_faces) > 1:
            return {"success": False, "message": "Multiple faces detected. One at a time."}

        face = valid_faces[0]
        region = face.get("facial_area", {})
        left, top = region.get("x", 0), region.get("y", 0)
        w, h = region.get("w", 0), region.get("h", 0)
        face_region = image_array[top:top+h, left:left+w]
        quality = _estimate_face_quality(face_region)

        if quality < 0.3:
            return {"success": False, "message": "Face not clear enough. Adjust lighting."}

        query_embedding = extract_embedding(image_array)
        if query_embedding is None:
            return {"success": False, "message": "Failed to process face"}

        query_array = np.array(query_embedding)

        candidates = stored_embeddings
        if exclude_student_ids:
            candidates = [e for e in stored_embeddings if e.student_id not in exclude_student_ids]

        if not candidates:
            return {"success": False, "message": "All registered students already marked"}

        best_distance = float("inf")
        best_match = None

        for stored in candidates:
            distance = _cosine_distance(query_array, np.array(stored.embedding))
            if distance < best_distance:
                best_distance = distance
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


def _cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine distance between two vectors (0=identical)."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return float(1.0 - np.dot(vec1, vec2) / (norm1 * norm2))


def _estimate_face_quality(face_region: np.ndarray) -> float:
    """Estimate face quality using PIL only — no cv2 needed."""
    if face_region is None or face_region.size == 0:
        return 0.0

    height, width = face_region.shape[:2]

    # Size score
    size_score = min(1.0, (width * height) / (80 * 80))

    # Sharpness using numpy (no cv2)
    if len(face_region.shape) == 3:
        gray = np.mean(face_region, axis=2)
    else:
        gray = face_region.astype(float)

    # Simple Laplacian variance using numpy
    laplacian = (
        np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=0) +
        np.roll(gray, -1, axis=1) + np.roll(gray, 1, axis=1) -
        4 * gray
    )
    sharpness_score = min(1.0, float(np.var(laplacian)) / 500.0)

    # Brightness score
    mean_brightness = float(np.mean(gray))
    if 60 <= mean_brightness <= 200:
        brightness_score = 1.0
    elif mean_brightness < 60:
        brightness_score = mean_brightness / 60.0
    else:
        brightness_score = max(0.0, 1.0 - (mean_brightness - 200) / 55.0)

    quality = size_score * 0.3 + sharpness_score * 0.5 + brightness_score * 0.2
    return round(min(1.0, quality), 4)