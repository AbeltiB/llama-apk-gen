"""
Production-Ready Intent Analyzer - Backward Compatible
Exports global 'intent_analyzer' instance for existing code
"""
import json
import re
import time
import hashlib
import asyncio
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import httpx

from app.services.analysis.intent_schemas import (
    IntentAnalysisResult,
    IntentType,
    ComplexityLevel,
    SafetyStatus,
    ActionRecommendation,
    AppDomain,
    ExtractedEntities,
    ConfidenceBreakdown,
    TechnicalRequirements,
    ClassificationRequest,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ProductionIntentAnalyzer:
    """
    Production-ready intent analyzer with:
    - Llama3 primary classification
    - Robust heuristic fallback
    - Comprehensive error handling
    - Request caching
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analyzer"""
        # Llama3 configuration
        self.api_url = config.get('llama3_api_url', '')
        self.api_key = config.get('llama3_api_key', '')
        self.model = config.get('llama3_model', 'llama-3-70b-instruct')
        self.timeout = config.get('timeout', 60.0)
        self.max_retries = config.get('max_retries', 3)
        
        # Caching configuration
        self.enable_caching = config.get('enable_caching', True)
        self.cache_ttl = config.get('cache_ttl', 300)
        self.cache: Dict[str, tuple] = {}
        
        # Llama3 availability
        self.llama3_available = bool(self.api_url)
        if not self.llama3_available:
            logger.warning("Llama3 API URL not configured - using heuristic fallback only")
        elif not self.api_key:
            logger.warning("Llama3 API key not configured - API calls may fail")
        
        # Circuit breaker for Llama3
        self.llama3_failure_count = 0
        self.llama3_max_failures = 3
        self.llama3_circuit_open = False
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'llama3_success': 0,
            'llama3_failures': 0,
            'heuristic_fallback': 0,
            'total_latency_ms': 0,
        }
        
        logger.info(
            "intent_analyzer.initialized",
            extra={
                "llama3_available": self.llama3_available,
                "api_url": self.api_url[:50] if self.api_url else "Not configured"
            }
        )
    
    async def analyze(
        self,
        prompt: str,
        user_id: str = "unknown",
        session_id: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysisResult:
        """
        Main analysis method - GUARANTEED to return result
        
        Args:
            prompt: User's natural language request
            user_id: User identifier
            session_id: Session identifier
            context: Optional context information
            
        Returns:
            IntentAnalysisResult - always returns, never raises
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Create request object
            request = ClassificationRequest(
                prompt=prompt,
                user_id=user_id,
                session_id=session_id,
                context=context
            )
            
            # Check cache
            if self.enable_caching:
                cached = self._check_cache(request)
                if cached:
                    self.stats['cache_hits'] += 1
                    logger.debug("intent_analyzer.cache_hit")
                    return cached
            
            # Try Llama3 first (if available and circuit not open)
            if self.llama3_available and not self.llama3_circuit_open:
                try:
                    result = await self._analyze_with_llama3(request)
                    
                    # Success - reset failure count
                    self.llama3_failure_count = 0
                    self.stats['llama3_success'] += 1
                    
                    # Cache result
                    if self.enable_caching:
                        self._cache_result(request, result)
                    
                    # Update latency
                    latency_ms = int((time.time() - start_time) * 1000)
                    result.total_latency_ms = latency_ms
                    self.stats['total_latency_ms'] += latency_ms
                    
                    logger.info(
                        "intent_analyzer.llama3_success",
                        extra={
                            "intent": result.intent_type.value,
                            "latency_ms": latency_ms
                        }
                    )
                    return result
                    
                except Exception as e:
                    # Llama3 failed - log and continue to fallback
                    self._handle_llama3_failure(e)
            
            # Fallback to heuristic
            result = self._analyze_with_heuristic(request)
            self.stats['heuristic_fallback'] += 1
            
            # Cache result
            if self.enable_caching:
                self._cache_result(request, result)
            
            # Update latency
            latency_ms = int((time.time() - start_time) * 1000)
            result.total_latency_ms = latency_ms
            self.stats['total_latency_ms'] += latency_ms
            
            logger.info(
                "intent_analyzer.heuristic_success",
                extra={"intent": result.intent_type.value}
            )
            return result
            
        except Exception as e:
            # Critical error - return safe fallback
            logger.error(
                "intent_analyzer.critical_error",
                extra={"error": str(e)},
                exc_info=True
            )
            return self._create_safe_fallback(prompt)
    
    # ================== LLAMA3 INTEGRATION ==================
    
    async def _analyze_with_llama3(
        self,
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Analyze using Llama3 API with robust error handling"""
        
        # Build prompts
        system_prompt = self._build_llama3_system_prompt()
        user_prompt = self._build_llama3_user_prompt(request)
        
        # Call API with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                response_data = await self._call_llama3_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    attempt=attempt
                )
                
                # Parse response
                result = self._parse_llama3_response(response_data, request)
                return result
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise Exception(f"Llama3 API authentication failed - check API key")
                elif e.response.status_code == 429:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
                else:
                    raise
            
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(1 * attempt)
                    continue
                raise
        
        raise Exception(f"Llama3 API failed after {self.max_retries} attempts")
    
    async def _call_llama3_api(
        self,
        system_prompt: str,
        user_prompt: str,
        attempt: int = 1
    ) -> Dict[str, Any]:
        """Make API call to Llama3"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
    
    def _build_llama3_system_prompt(self) -> str:
        """Build system prompt for Llama3"""
        return """You are an expert intent classifier for a mobile app generation system.

Classify user requests into these categories:

INTENT TYPES:
- create_app: Create new application
- modify_app: Modify existing app  
- extend_app: Add features to existing app
- bug_fix: Report bug or issue
- optimize_performance: Performance improvements
- clarification: Unclear request needing more info
- help: General question or help request
- other: Other requests

COMPLEXITY (use ONLY these 3 values):
- simple: Basic app, 1-3 components, single screen
- medium: Multiple screens, 3-8 components
- complex: Advanced features, API integration, complex logic

SAFETY:
- safe: Normal request
- suspicious: Potentially problematic
- unsafe: Malicious or harmful

CRITICAL: Respond with ONLY valid JSON, no markdown, no explanations.

Format:
{
  "intent_type": "create_app",
  "complexity": "simple",
  "confidence": {
    "overall": 0.85,
    "intent_confidence": 0.9,
    "complexity_confidence": 0.8,
    "entity_confidence": 0.85,
    "safety_confidence": 1.0
  },
  "extracted_entities": {
    "components": ["Button", "Text"],
    "actions": ["click"],
    "data_types": ["counter"],
    "features": ["increment"]
  },
  "safety_status": "safe",
  "requires_context": false,
  "reasoning": "User wants simple counter app"
}

"""
    
    def _build_llama3_user_prompt(
        self,
        request: ClassificationRequest
    ) -> str:
        """Build user prompt"""
        prompt = f'Classify this request:\n\n"{request.prompt}"'
        
        if request.context:
            if request.context.get('has_existing_project'):
                prompt += "\n\nNote: User has existing project in session."
        
        return prompt
    
    def _parse_llama3_response(
        self,
        response_data: Dict[str, Any],
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Parse Llama3 response to IntentAnalysisResult"""
        
        # Extract content
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            raise ValueError("Invalid Llama3 response: no choices")
        
        content = response_data['choices'][0]['message']['content']
        
        # Clean markdown
        content = content.strip()
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError("Could not parse Llama3 response as JSON")
        
        # Map to enums
        intent_map = {
            "create_app": IntentType.CREATE_APP,
            "modify_app": IntentType.MODIFY_APP,
            "extend_app": IntentType.EXTEND_APP,
            "bug_fix": IntentType.BUG_FIX,
            "optimize_performance": IntentType.OPTIMIZE_PERFORMANCE,
            "clarification": IntentType.CLARIFICATION,
            "help": IntentType.HELP,
            "other": IntentType.OTHER,
        }
        
        intent_type = intent_map.get(
            data.get('intent_type', 'other'),
            IntentType.OTHER
        )
        
        # Normalize complexity
        complexity_str = data.get('complexity', 'medium').lower()
        complexity = ComplexityLevel.normalize(complexity_str)
        
        # Safety
        safety_map = {
            "safe": SafetyStatus.SAFE,
            "suspicious": SafetyStatus.SUSPICIOUS,
            "unsafe": SafetyStatus.UNSAFE,
        }
        safety = safety_map.get(
            data.get('safety_status', 'safe'),
            SafetyStatus.SAFE
        )
        
        # Confidence
        conf_data = data.get('confidence', {})
        confidence = ConfidenceBreakdown(
            overall=conf_data.get('overall', 0.75),
            intent_confidence=conf_data.get('intent_confidence', 0.75),
            complexity_confidence=conf_data.get('complexity_confidence', 0.75),
            entity_confidence=conf_data.get('entity_confidence', 0.75),
            safety_confidence=conf_data.get('safety_confidence', 1.0),
        )
        
        # Entities
        entities_data = data.get('extracted_entities', {})
        entities = ExtractedEntities(
            components=entities_data.get('components', []),
            actions=entities_data.get('actions', []),
            data_types=entities_data.get('data_types', []),
            features=entities_data.get('features', []),
        )
        
        # Determine action
        action = self._determine_action(intent_type, safety, confidence)
        
        # Estimate cost
        tokens = response_data.get('usage', {}).get('total_tokens', 0)
        cost_usd = (tokens / 1000) * 0.001
        
        return IntentAnalysisResult(
            intent_type=intent_type,
            complexity=complexity,
            confidence=confidence,
            extracted_entities=entities,
            action_recommendation=action,
            safety_status=safety,
            requires_context=data.get('requires_context', False),
            reasoning=data.get('reasoning', ''),
            source="llama3",
            total_cost_usd=cost_usd,
        )
    
    def _handle_llama3_failure(self, error: Exception):
        """Handle Llama3 failure with circuit breaker"""
        self.llama3_failure_count += 1
        self.stats['llama3_failures'] += 1
        
        logger.warning(
            "intent_analyzer.llama3_failed",
            extra={"error": str(error), "failures": self.llama3_failure_count}
        )
        
        if self.llama3_failure_count >= self.llama3_max_failures:
            self.llama3_circuit_open = True
            logger.error("intent_analyzer.circuit_breaker_open")
    
    # ================== HEURISTIC FALLBACK ==================
    
    def _analyze_with_heuristic(
        self,
        request: ClassificationRequest
    ) -> IntentAnalysisResult:
        """Heuristic analysis using pattern matching"""
        
        prompt_lower = request.prompt.lower()
        words = prompt_lower.split()
        
        # Classify intent
        intent_type, intent_conf = self._heuristic_classify_intent(prompt_lower)
        
        # Classify complexity
        complexity, complex_conf = self._heuristic_classify_complexity(
            prompt_lower, len(words)
        )
        
        # Extract entities
        entities, entity_conf = self._heuristic_extract_entities(prompt_lower)
        
        # Check safety
        safety, safety_conf = self._heuristic_check_safety(prompt_lower)
        
        # Calculate overall confidence
        overall = (
            intent_conf * 0.35 +
            complex_conf * 0.20 +
            entity_conf * 0.25 +
            safety_conf * 0.20
        )
        
        confidence = ConfidenceBreakdown(
            overall=overall,
            intent_confidence=intent_conf,
            complexity_confidence=complex_conf,
            entity_confidence=entity_conf,
            safety_confidence=safety_conf,
        )
        
        action = self._determine_action(intent_type, safety, confidence)
        
        return IntentAnalysisResult(
            intent_type=intent_type,
            complexity=complexity,
            confidence=confidence,
            extracted_entities=entities,
            action_recommendation=action,
            safety_status=safety,
            requires_context=intent_type in [IntentType.MODIFY_APP, IntentType.EXTEND_APP],
            source="heuristic",
        )
    
    def _heuristic_classify_intent(self, prompt: str) -> tuple:
        prompt_lower = prompt.lower()

        from app.services.analysis.intent_config import HARDWARE_PATTERNS, AI_PATTERNS
    
        # Check hardware patterns
        for hw_type, patterns in HARDWARE_PATTERNS.items():
            if any(kw in prompt_lower for kw in patterns.get("keywords", [])):
                return IntentType.CREATE_APP, 0.9  # Hardware apps are create_app
        
        # Check AI patterns  
        for ai_type, patterns in AI_PATTERNS.items():
            if any(kw in prompt_lower for kw in patterns.get("keywords", [])):
                return IntentType.CREATE_APP, 0.85
            
        """Classify intent using keywords"""
            
        patterns = {
            IntentType.CREATE_APP: ["create", "build", "make", "generate", "new"],
            IntentType.MODIFY_APP: ["change", "modify", "update", "edit", "fix"],
            IntentType.EXTEND_APP: ["add", "also", "include", "plus", "extend"],
            IntentType.HELP: ["help", "how", "what", "why", "explain"],
            IntentType.CLARIFICATION: ["?", "unclear", "not sure"],
        }
        
        scores = {intent: 0 for intent in patterns}
        
        for intent, keywords in patterns.items():
            for keyword in keywords:
                if keyword in prompt:
                    scores[intent] += 1
        
        if sum(scores.values()) == 0:
            return IntentType.CREATE_APP, 0.5
        
        best = max(scores.items(), key=lambda x: x[1])
        confidence = min(0.85, 0.5 + (best[1] * 0.15))
        
        return best[0], confidence
    
    def _heuristic_classify_complexity(
        self, 
        prompt: str, 
        word_count: int
    ) -> tuple:
        """Classify complexity"""
        
        if any(w in prompt for w in ["simple", "basic", "just"]):
            return ComplexityLevel.SIMPLE, 0.8
        
        if any(w in prompt for w in ["complete", "full", "complex", "advanced"]):
            return ComplexityLevel.COMPLEX, 0.8
        
        if word_count <= 10:
            return ComplexityLevel.SIMPLE, 0.7
        elif word_count <= 30:
            return ComplexityLevel.MEDIUM, 0.7
        else:
            return ComplexityLevel.COMPLEX, 0.6
    
    def _heuristic_extract_entities(self, prompt: str) -> tuple:
        """Extract entities"""
        
        components = []
        actions = []
        features = []
        
        comp_map = {
            "button": ["button", "btn"],
            "text": ["text", "label"],
            "input": ["input", "textbox", "field"],
            "list": ["list", "items"],
        }
        
        for comp, aliases in comp_map.items():
            if any(a in prompt for a in aliases):
                components.append(comp.capitalize())
        
        action_words = ["click", "tap", "press", "type", "input"]
        actions = [a for a in action_words if a in prompt]
        
        feature_words = ["login", "search", "filter", "notification"]
        features = [f for f in feature_words if f in prompt]
        
        entities = ExtractedEntities(
            components=components,
            actions=actions,
            features=features,
        )
        
        total = len(components) + len(actions) + len(features)
        conf = min(0.8, 0.4 + (total * 0.1))
        
        return entities, conf
    
    def _heuristic_check_safety(self, prompt: str) -> tuple:
        """Check for unsafe content"""
        
        unsafe_keywords = [
            "hack", "malware", "virus", "exploit", "crack",
            "steal", "password", "credit card"
        ]
        
        for keyword in unsafe_keywords:
            if keyword in prompt:
                return SafetyStatus.UNSAFE, 0.95
        
        return SafetyStatus.SAFE, 0.9
    
    # ================== COMMON UTILITIES ==================
    
    def _determine_action(
        self,
        intent: IntentType,
        safety: SafetyStatus,
        confidence: ConfidenceBreakdown
    ) -> ActionRecommendation:
        """Determine recommended action"""
        
        if safety == SafetyStatus.UNSAFE:
            return ActionRecommendation.REJECT
        
        if intent == IntentType.MODIFY_APP and confidence.overall < 0.7:
            return ActionRecommendation.BLOCK_MODIFY
        
        if intent == IntentType.EXTEND_APP and confidence.overall < 0.7:
            return ActionRecommendation.BLOCK_EXTEND
        
        if confidence.overall < 0.6:
            return ActionRecommendation.CLARIFY
        
        return ActionRecommendation.PROCEED
    
    def _create_safe_fallback(self, prompt: str) -> IntentAnalysisResult:
        """Create safe fallback result"""
        return IntentAnalysisResult(
            intent_type=IntentType.CLARIFICATION,
            complexity=ComplexityLevel.MEDIUM,
            confidence=ConfidenceBreakdown(
                overall=0.3,
                intent_confidence=0.3,
                complexity_confidence=0.4,
                entity_confidence=0.2,
                safety_confidence=0.8,
            ),
            extracted_entities=ExtractedEntities(),
            action_recommendation=ActionRecommendation.CLARIFY,
            safety_status=SafetyStatus.SAFE,
            user_message="I need more information to understand your request.",
            source="fallback",
        )
    
    # ================== CACHING ==================
    
    def _check_cache(self, request: ClassificationRequest) -> Optional[IntentAnalysisResult]:
        """Check cache for result"""
        key = self._cache_key(request)
        if key in self.cache:
            result, cached_time = self.cache[key]
            age = time.time() - cached_time
            if age < self.cache_ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def _cache_result(self, request: ClassificationRequest, result: IntentAnalysisResult):
        """Cache result"""
        key = self._cache_key(request)
        self.cache[key] = (result, time.time())
    
    def _cache_key(self, request: ClassificationRequest) -> str:
        """Generate cache key"""
        data = f"{request.prompt.lower().strip()}|{request.user_id}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'cache_hit_rate': (self.stats['cache_hits'] / total) * 100,
            'llama3_success_rate': (self.stats['llama3_success'] / total) * 100,
            'heuristic_rate': (self.stats['heuristic_fallback'] / total) * 100,
            'avg_latency_ms': self.stats['total_latency_ms'] / total,
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'llama3_success': 0,
            'llama3_failures': 0,
            'heuristic_fallback': 0,
            'total_latency_ms': 0,
        }


# ================== FACTORY FUNCTION ==================

def create_intent_analyzer(config: Dict[str, Any]) -> ProductionIntentAnalyzer:
    """Factory function to create analyzer instance"""
    return ProductionIntentAnalyzer(config)


# ================== GLOBAL INSTANCE (for backward compatibility) ==================

def _create_global_instance():
    """Create global instance from environment/settings"""
    try:
        from app.config import settings
        
        config = {
            'llama3_api_url': getattr(settings, 'LLAMA3_API_URL', os.getenv('LLAMA3_API_URL', 'settings.llama3_api_url')),
            'llama3_api_key': getattr(settings, 'LLAMA3_API_KEY', os.getenv('LLAMA3_API_KEY', 'settings.llama3_api_key')),
            'llama3_model': getattr(settings, 'LLAMA3_MODEL', os.getenv('LLAMA3_MODEL', 'llama-3-70b-instruct')),
            'timeout': float(getattr(settings, 'LLAMA3_TIMEOUT', os.getenv('LLAMA3_TIMEOUT', '60.0'))),
            'max_retries': int(getattr(settings, 'LLAMA3_MAX_RETRIES', os.getenv('LLAMA3_MAX_RETRIES', '3'))),
            'enable_caching': True,
            'cache_ttl': 300,
        }
        
        return ProductionIntentAnalyzer(config)
        
    except Exception as e:
        logger.warning(
            "Failed to create global intent analyzer from settings, using fallback config",
            extra={"error": str(e)}
        )
        
        # Fallback config - will use heuristic only
        fallback_config = {
            'llama3_api_url': os.getenv('LLAMA3_API_URL', ''),
            'llama3_api_key': os.getenv('LLAMA3_API_KEY', ''),
            'llama3_model': 'llama-3-70b-instruct',
            'timeout': 60.0,
            'max_retries': 3,
            'enable_caching': True,
            'cache_ttl': 300,
        }
        
        return ProductionIntentAnalyzer(fallback_config)


# Create global instance
intent_analyzer = _create_global_instance()


# ================== EXPORTS ==================

__all__ = [
    "ProductionIntentAnalyzer",
    "create_intent_analyzer",
    "intent_analyzer",  # Global instance for backward compatibility
]