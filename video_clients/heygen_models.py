# file: video_clients/heygen_models.py

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# --- Request/Input Models ---

class HeyGenVideoRequest(BaseModel):
    """Model for submitting a new video generation job to HeyGen."""
    avatar_id: str = Field(..., description="The unique ID of the character avatar.")
    audio_url: str = Field(..., description="Public URL of the synthesized audio file.")
    scene_prompt: str = Field(..., description="Visual background and action prompt.")
    ratio: str = Field(..., description="Aspect ratio (e.g., '16:9' or '9:16').")
    scene_duration: float = Field(..., description="Target duration in seconds.")
    video_quality: str = Field("medium", description="Desired video quality ('medium', 'high').")
    
    # Optional Consistency Parameter (Conceptual feature based on Veo)
    ref_image_url: Optional[str] = Field(None, description="Reference image for visual consistency.")

# --- Response/Status Models ---

class HeyGenJobStatus(BaseModel):
    """Model for polling the status of a video generation job."""
    job_id: str = Field(..., description="The unique ID returned upon job submission.")
    status: str = Field(..., description="The current status (e.g., 'pending', 'processing', 'completed', 'failed').")
    video_url: Optional[str] = Field(None, description="The final URL of the video upon completion.")
    error_message: Optional[str] = Field(None, description="Error details if status is 'failed'.")

class HeyGenAvatarInfo(BaseModel):
    """Model for storing or retrieving avatar information."""
    name: str
    avatar_id: str
    image_url: Optional[str] = None
    voice_id: Optional[str] = None
