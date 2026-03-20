# python_backend/main.py

import os
import sys

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional

import database
import face_service
from models import (
    DetectFacesResponse,
    RegisterFaceResponse,
    RecognizeFaceResponse,
    DeleteFaceResponse,
    HealthResponse,
)

app = FastAPI(
    title="Smart Attendance Face Recognition API",
    description="Python backend for face detection, recognition and embedding storage",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database and warm up InsightFace models on startup."""
    print("Starting Smart Attendance Face Recognition API...")

    # Init SQLite
    database.init_db()

    # Warm up InsightFace — downloads buffalo_sc (~85 MB) on first run,
    # then caches it. Subsequent starts are instant.
    try:
        print("Loading InsightFace models (first run downloads ~85 MB)...")
        face_service.warmup()
    except Exception as e:
        print(f"Warning: InsightFace warmup failed: {e}", file=sys.stderr)

    print("API startup complete.")


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    registered_count = database.get_registered_count()
    return HealthResponse(
        status="ok",
        message=f"API is running. {registered_count} face(s) registered."
    )


# ─────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────

@app.post("/detect", response_model=DetectFacesResponse)
async def detect_faces(image: UploadFile = File(...)):
    """Detect all faces in an uploaded image."""
    try:
        image_bytes = await image.read()
        detected_faces, _ = face_service.detect_faces(image_bytes)
        return DetectFacesResponse(
            success=True,
            faces=detected_faces,
            face_count=len(detected_faces),
            message=f"Detected {len(detected_faces)} face(s)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


# ─────────────────────────────────────────────
# FACE REGISTRATION
# ─────────────────────────────────────────────

@app.post("/register", response_model=RegisterFaceResponse)
async def register_face(
    image: UploadFile = File(...),
    student_id: int = Form(...),
    student_name: str = Form(...)
):
    """Register a student's face — extracts and stores the embedding."""
    try:
        image_bytes = await image.read()
        result = face_service.register_face(image_bytes, student_id, student_name)

        if not result["success"]:
            return RegisterFaceResponse(success=False, message=result["message"])

        saved = database.save_embedding(
            student_id=student_id,
            student_name=student_name,
            embedding=result["embedding"]
        )

        if not saved:
            return RegisterFaceResponse(
                success=False,
                message="Failed to save face data to database"
            )

        return RegisterFaceResponse(
            success=True,
            message=f"Face registered successfully for {student_name}",
            student_id=student_id,
            embedding_size=result["embedding_size"],
            quality=result["quality"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")


# ─────────────────────────────────────────────
# FACE RECOGNITION
# ─────────────────────────────────────────────

@app.post("/recognize", response_model=RecognizeFaceResponse)
async def recognize_face(
    image: UploadFile = File(...),
    exclude_ids: Optional[str] = Form(default=None)
):
    """Recognize a face against all stored embeddings."""
    try:
        image_bytes = await image.read()

        excluded = []
        if exclude_ids:
            try:
                excluded = [int(x.strip()) for x in exclude_ids.split(",") if x.strip()]
            except ValueError:
                pass

        stored_embeddings = database.get_all_embeddings()

        if not stored_embeddings:
            return RecognizeFaceResponse(
                success=False,
                message="No registered faces in database"
            )

        result = face_service.recognize_face(image_bytes, stored_embeddings, excluded)

        return RecognizeFaceResponse(
            success=result["success"],
            message=result["message"],
            matched_student_id=result.get("matched_student_id"),
            matched_student_name=result.get("matched_student_name"),
            confidence=result.get("confidence"),
            face_quality=result.get("face_quality")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")


# ─────────────────────────────────────────────
# FACE MANAGEMENT
# ─────────────────────────────────────────────

@app.delete("/face/{student_id}", response_model=DeleteFaceResponse)
async def delete_face(student_id: int):
    """Delete a student's registered face embedding."""
    deleted = database.delete_embedding(student_id)
    if deleted:
        return DeleteFaceResponse(
            success=True,
            message=f"Face data deleted for student {student_id}"
        )
    return DeleteFaceResponse(
        success=False,
        message=f"No face data found for student {student_id}"
    )


@app.get("/face/{student_id}")
async def check_face_registered(student_id: int):
    """Check if a student has a registered face."""
    embedding = database.get_embedding(student_id)
    return {
        "registered": embedding is not None,
        "student_id": student_id,
        "student_name": embedding.student_name if embedding else None
    }


@app.get("/faces/count")
async def get_registered_count():
    """Get total number of registered faces."""
    return {"count": database.get_registered_count()}


# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )