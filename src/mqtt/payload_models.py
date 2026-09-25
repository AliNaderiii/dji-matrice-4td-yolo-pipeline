"""
Pydantic Models for DJI Matrice 4TD MQTT JSON Payloads
Author: Ali Naderi | Edge AI

Provides type-safe validation for inference results.
Extra: Provides validated models, we provide validated models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box - supports both pixel and normalized formats"""
    x_min: float = Field(..., description="Left coordinate")
    y_min: float = Field(..., description="Top coordinate")
    x_max: float = Field(..., description="Right coordinate")
    y_max: float = Field(..., description="Bottom coordinate")
    
    @validator('x_max')
    def x_max_gt_min(cls, v, values):
        if 'x_min' in values and v <= values['x_min']:
            raise ValueError('x_max must be > x_min')
        return v
    
    @validator('y_max')
    def y_max_gt_min(cls, v, values):
        if 'y_min' in values and v <= values['y_min']:
            raise ValueError('y_max must be > y_min')
        return v

    def to_xywh(self) -> List[float]:
        """Convert to x,y,w,h"""
        return [self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min]

    def to_normalized(self, img_w: int = 640, img_h: int = 640) -> List[float]:
        """Normalize to 0-1"""
        return [self.x_min/img_w, self.y_min/img_h, self.x_max/img_w, self.y_max/img_h]


class Detection(BaseModel):
    """Single object detection"""
    class_id: int = Field(..., ge=0, le=3, description="0:Person, 1:Vehicle, 2:Hard-Hat, 3:No-Hard-Hat")
    class_name: str = Field(..., description="Human readable class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: List[float] = Field(..., min_items=4, max_items=4, description="[x_min, y_min, x_max, y_max] or normalized")
    track_id: Optional[str] = Field(None, description="Optional tracking ID")
    
    @validator('class_name')
    def validate_class_name(cls, v, values):
        valid_names = ["Person", "Vehicle", "Hard-Hat", "No-Hard-Hat"]
        if v not in valid_names:
            # Allow but warn - map class_id to name if mismatch
            if 'class_id' in values:
                mapping = {0: "Person", 1: "Vehicle", 2: "Hard-Hat", 3: "No-Hard-Hat"}
                expected = mapping.get(values['class_id'], "Unknown")
                # Don't fail, just log - DJI may have custom names
                pass
        return v

    @validator('bbox')
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError('bbox must have 4 elements')
        # Check if normalized (all <=1) or pixel (some >1)
        # Both are valid, DJI can output either
        return v

    def is_safety_violation(self) -> bool:
        """Check if this detection is a safety violation (No-Hard-Hat)"""
        return self.class_name == "No-Hard-Hat" and self.confidence > 0.7


class InferencePayload(BaseModel):
    """Full inference payload from Matrice 4TD"""
    timestamp: int = Field(..., description="Unix timestamp ms")
    drone_sn: Optional[str] = Field(None, description="Drone serial number")
    frame_id: Optional[int] = Field(None, description="Frame sequence ID")
    detections: List[Detection] = Field(default_factory=list)
    inference_time_ms: Optional[float] = Field(None, description="NPU inference time")
    model_version: Optional[str] = Field(None, description="Deployed model version")
    
    # DJI Cloud API fields (optional, for compatibility)
    bid: Optional[str] = Field(None, description="Business ID from DJI")
    tid: Optional[str] = Field(None, description="Transaction ID from DJI")
    method: Optional[str] = Field(None, description="Method, e.g., ai_inference_result")
    
    def get_safety_violations(self) -> List[Detection]:
        """Get all No-Hard-Hat detections"""
        return [d for d in self.detections if d.is_safety_violation()]
    
    def has_person_without_hardhat(self) -> bool:
        """Check if any person without hard-hat detected"""
        return len(self.get_safety_violations()) > 0
    
    def to_summary(self) -> dict:
        """Get human-readable summary"""
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp/1000).isoformat() if self.timestamp else None,
            "drone": self.drone_sn,
            "total_detections": len(self.detections),
            "persons": len([d for d in self.detections if d.class_name == "Person"]),
            "vehicles": len([d for d in self.detections if d.class_name == "Vehicle"]),
            "hard_hats": len([d for d in self.detections if d.class_name == "Hard-Hat"]),
            "no_hard_hats": len([d for d in self.detections if d.class_name == "No-Hard-Hat"]),
            "safety_violations": len(self.get_safety_violations()),
            "inference_ms": self.inference_time_ms
        }


class AlertPayload(BaseModel):
    """Alert payload for safety violations - over-delivery"""
    alert_type: str = Field("SAFETY_VIOLATION", description="Type of alert")
    severity: str = Field("HIGH", description="LOW, MEDIUM, HIGH")
    timestamp: int
    drone_sn: str
    location: Optional[dict] = None  # lat, lon, alt from OSD
    violations: List[Detection]
    message: str
    
    @classmethod
    def from_inference(cls, inference: InferencePayload, location: Optional[dict] = None):
        violations = inference.get_safety_violations()
        if not violations:
            return None
        
        return cls(
            timestamp=inference.timestamp,
            drone_sn=inference.drone_sn or "unknown",
            violations=violations,
            location=location,
            message=f"Detected {len(violations)} person(s) without hard-hat",
            severity="HIGH" if len(violations) >= 2 else "MEDIUM"
        )
