from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List


class DocumentPayload(BaseModel):
    base64: str
    mimeType: str
    filename: str
    extension: str = ""


class ChatRequest(BaseModel):
    query: str
    businessId: str = "aurika-campus"
    studentId: str
    documents: List[DocumentPayload] = []
    sessionId: Optional[str] = None


class CampusLearningRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str
    level: str
    medium: str
    course_type: str = Field(alias="courseType")
    student_id: str = Field(alias="studentId")
    student_name: str = Field(alias="studentName")
    learning_preferences: str = Field(alias="learningPreferences")
    business_id: str = Field(default="aurika-campus", alias="businessId")
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    documents: List[DocumentPayload] = Field(default_factory=list)
    file_base64: Optional[str] = Field(default=None, alias="fileBase64")
    file_mime_type: Optional[str] = Field(default=None, alias="fileMimeType")
    file_name: Optional[str] = Field(default=None, alias="fileName")
    board: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None


class ResponseContent(BaseModel):
    text: str
    studentName: str = "Student"
    subject: str = "N/A"
    class_: str = Field(default="N/A", serialization_alias="class")
    chapter: str = "N/A"


class ChatResponse(BaseModel):
    success: bool = True
    sessionId: str
    studentId: str
    businessId: str
    detectedSubject: Optional[str] = None
    response: ResponseContent
    metadata: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
