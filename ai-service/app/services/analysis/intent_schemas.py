"""
Unified Intent Schemas - Production Ready
Fixed all enum and validation issues
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum


class IntentType(str, Enum):
    """Types of user intents - Unified across all modules"""
    CREATE_APP = "create_app"
    MODIFY_APP = "modify_app"
    EXTEND_APP = "extend_app"
    BUG_FIX = "bug_fix"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    CLARIFICATION = "clarification"
    HELP = "help"
    OTHER = "other"


class ComplexityLevel(str, Enum):
    """
    Unified complexity levels - Works with both simple and advanced systems
    Maps to both backend processing and LLM responses
    """
    # Simple 3-tier system (used by LLMs)
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    
    # Advanced 6-tier system (used internally)
    SIMPLE_UI = "simple_ui"
    DATA_DRIVEN = "data_driven"
    INTEGRATED = "integrated"
    ENTERPRISE = "enterprise"
    HARDWARE = "hardware"
    AI_ML = "ai_ml"
    
    @classmethod
    def normalize(cls, value: str) -> "ComplexityLevel":
        """
        Normalize any complexity value to valid enum
        Maps advanced types to simple types for LLM compatibility
        """
        mapping = {
            # Direct mappings
            "simple": cls.SIMPLE,
            "medium": cls.MEDIUM,
            "complex": cls.COMPLEX,
            "simple_ui": cls.SIMPLE_UI,
            "data_driven": cls.DATA_DRIVEN,
            "integrated": cls.INTEGRATED,
            "enterprise": cls.ENTERPRISE,
            "hardware": cls.HARDWARE,
            "ai_ml": cls.AI_ML,
            
            # Fallback mappings (advanced -> simple)
            "simple_ui_fallback": cls.SIMPLE,
            "data_driven_fallback": cls.MEDIUM,
            "integrated_fallback": cls.MEDIUM,
            "enterprise_fallback": cls.COMPLEX,
            "hardware_fallback": cls.COMPLEX,
            "ai_ml_fallback": cls.COMPLEX,
        }
        
        # Try direct lookup
        normalized = value.lower().strip()
        if normalized in mapping:
            return mapping[normalized]
        
        # Default to MEDIUM
        return cls.MEDIUM
    
    def to_simple(self) -> "ComplexityLevel":
        """Convert advanced complexity to simple 3-tier"""
        mapping = {
            ComplexityLevel.SIMPLE: ComplexityLevel.SIMPLE,
            ComplexityLevel.MEDIUM: ComplexityLevel.MEDIUM,
            ComplexityLevel.COMPLEX: ComplexityLevel.COMPLEX,
            ComplexityLevel.SIMPLE_UI: ComplexityLevel.SIMPLE,
            ComplexityLevel.DATA_DRIVEN: ComplexityLevel.MEDIUM,
            ComplexityLevel.INTEGRATED: ComplexityLevel.MEDIUM,
            ComplexityLevel.ENTERPRISE: ComplexityLevel.COMPLEX,
            ComplexityLevel.HARDWARE: ComplexityLevel.COMPLEX,
            ComplexityLevel.AI_ML: ComplexityLevel.COMPLEX,
        }
        return mapping.get(self, ComplexityLevel.MEDIUM)


class SafetyStatus(str, Enum):
    """Safety classification"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    UNSAFE = "unsafe"


class ActionRecommendation(str, Enum):
    """Recommended action"""
    PROCEED = "proceed"
    CLARIFY = "clarify"
    BLOCK_MODIFY = "block_modify"
    BLOCK_EXTEND = "block_extend"
    REJECT = "reject"


class AppDomain(str, Enum):
    """Application domains"""
    PRODUCTIVITY = "productivity"
    ENTERTAINMENT = "entertainment"
    UTILITY = "utility"
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTH_FITNESS = "health_fitness"
    FINANCE = "finance"
    DEVELOPMENT = "development"
    IOT_HARDWARE = "iot_hardware"
    CREATIVE_MEDIA = "creative_media"
    DATA_SCIENCE = "data_science"
    CUSTOM = "custom"


class ExtractedEntities(BaseModel):
    """Entities extracted from prompt"""
    components: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    data_types: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    screens: List[str] = Field(default_factory=list)
    integrations: List[str] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    """Confidence scoring"""
    overall: float = Field(default=0.7, ge=0.0, le=1.0)
    intent_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    complexity_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    safety_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TechnicalRequirements(BaseModel):
    """Technical requirements for the app"""
    needs_hardware: bool = False
    needs_ai_ml: bool = False
    needs_real_time: bool = False
    needs_3d: bool = False
    special_apis: List[str] = Field(default_factory=list)
    complex_components: List[str] = Field(default_factory=list)
    permissions_required: List[str] = Field(default_factory=list)


class IntentAnalysisResult(BaseModel):
    """
    Unified intent analysis result
    Compatible with all analyzers and LLM providers
    """
    
    # Core classification
    intent_type: IntentType
    complexity: ComplexityLevel
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    
    # Extracted information
    extracted_entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    technical_requirements: Optional[TechnicalRequirements] = None
    
    # Decision making
    action_recommendation: ActionRecommendation = ActionRecommendation.PROCEED
    safety_status: SafetyStatus = SafetyStatus.SAFE
    requires_context: bool = False
    multi_turn: bool = False
    
    # User communication
    user_message: Optional[str] = None
    suggested_clarifications: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    
    # Domain information
    domain: Optional[AppDomain] = None
    specific_type: Optional[str] = None
    
    # Metadata
    source: str = "unknown"  # llama3, heuristic, claude, etc.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    
    @field_validator('complexity', mode='before')
    @classmethod
    def normalize_complexity(cls, v):
        """Normalize complexity value"""
        if isinstance(v, str):
            return ComplexityLevel.normalize(v)
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "intent_type": self.intent_type.value,
            "complexity": self.complexity.value,
            "confidence": {
                "overall": self.confidence.overall,
                "intent_confidence": self.confidence.intent_confidence,
                "complexity_confidence": self.confidence.complexity_confidence,
                "entity_confidence": self.confidence.entity_confidence,
                "safety_confidence": self.confidence.safety_confidence,
            },
            "extracted_entities": {
                "components": self.extracted_entities.components,
                "actions": self.extracted_entities.actions,
                "data_types": self.extracted_entities.data_types,
                "features": self.extracted_entities.features,
            },
            "action_recommendation": self.action_recommendation.value,
            "safety_status": self.safety_status.value,
            "requires_context": self.requires_context,
            "user_message": self.user_message,
            "domain": self.domain.value if self.domain else None,
            "specific_type": self.specific_type,
            "source": self.source,
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "latency_ms": self.total_latency_ms,
                "cost_usd": self.total_cost_usd,
            }
        }


class ClassificationRequest(BaseModel):
    """Request for classification"""
    prompt: str = Field(..., min_length=1, max_length=5000)
    user_id: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    
    @field_validator('prompt')
    @classmethod
    def clean_prompt(cls, v: str) -> str:
        return v.strip()


__all__ = [
    "IntentType",
    "ComplexityLevel",
    "SafetyStatus",
    "ActionRecommendation",
    "AppDomain",
    "ExtractedEntities",
    "ConfidenceBreakdown",
    "TechnicalRequirements",
    "IntentAnalysisResult",
    "ClassificationRequest",
]