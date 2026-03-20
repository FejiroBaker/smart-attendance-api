# python_backend/models.py

from pydantic import BaseModel
from typing import Optional, List


class DetectedFace(BaseModel):
    left: float
    top: float
    right: float
    bottom: float
    confidence: float


class DetectFacesResponse(BaseModel):
    success: bool
    faces: List[DetectedFace]
    face_count: int
    message: str


class RegisterFaceRequest(BaseModel):
    student_id: int
    student_name: str


class RegisterFaceResponse(BaseModel):
    success: bool
    message: str
    student_id: Optional[int] = None
    embedding_size: Optional[int] = None
    quality: Optional[float] = None


class RecognizeFaceResponse(BaseModel):
    success: bool
    message: str
    matched_student_id: Optional[int] = None
    matched_student_name: Optional[str] = None
    confidence: Optional[float] = None
    face_quality: Optional[float] = None


class StoredEmbedding(BaseModel):
    student_id: int
    student_name: str
    embedding: List[float]


class DeleteFaceResponse(BaseModel):
    success: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    message: str
