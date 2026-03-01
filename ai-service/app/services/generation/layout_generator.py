"""
Layout Generator - Phase 3
Uses LLM Orchestrator (Llama3 → Heuristic fallback)
"""
import json
import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from difflib import get_close_matches

from app.config import settings
from app.models.schemas.architecture import ArchitectureDesign, ScreenDefinition
from app.models.schemas.components import EnhancedComponentDefinition
from app.models.schemas.layout import EnhancedLayoutDefinition
from app.models.schemas.core import PropertyValue
from app.models.schemas.component_catalog import (
    normalize_component_type,
    get_component_default_dimensions,
    get_component_default_properties,
    has_component_event,
    is_input_component,
)
from app.models.prompts import prompts
from app.services.generation.layout_validator import layout_validator
from app.llm.orchestrator import LLMOrchestrator
from app.llm.base import LLMMessage
from app.utils.logging import get_logger, log_context, trace_async

logger = get_logger(__name__)





class LayoutGenerationError(Exception):
    """Base exception for layout generation errors"""
    pass


class CollisionError(Exception):
    """Raised when UI elements collide during layout generation."""
    pass


class LayoutGenerator:
    """
    Phase 3 Layout Generator using LLM Orchestrator.
    
    Generation Flow:
    1. Try Llama3 via orchestrator
    2. Retry with corrections if needed
    3. Fall back to heuristic if all retries fail
    4. Validate and resolve collisions
    
    Features:
    - Llama3 as primary LLM
    - Automatic heuristic template fallback
    - ✅ Component type normalization I should make this from a central hub
    - Collision detection and resolution
    - Touch target validation
    """
    
    def __init__(self, orchestrator: Optional[LLMOrchestrator] = None):
        # Initialize LLM orchestrator
        if orchestrator:
            self.orchestrator = orchestrator
        else:
            config = {
                "failure_threshold": 3,
                "failure_window_minutes": 5,
                "llama3_api_url": settings.llama3_api_url,
                "llama3_api_key": settings.llama3_api_key
            }
            self.orchestrator = LLMOrchestrator(config)
        
        # Canvas constraints
        self.canvas_width = settings.canvas_width
        self.canvas_height = settings.canvas_height
        self.safe_area_top = settings.canvas_safe_area_top
        self.safe_area_bottom = settings.canvas_safe_area_bottom
        
        # ✅ Available components from settings
        self.available_components = settings.available_components
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'collisions_resolved': 0,
            'heuristic_fallbacks': 0,
            'llama3_successes': 0,
            'components_normalized': 0,
            'json_fixes_applied': 0
        }
        
        logger.info(
            "layout.generator.initialized",
            extra={
                "llm_provider": "llama3",
                "canvas": f"{self.canvas_width}x{self.canvas_height}",
                "heuristic_fallback_enabled": True,
                "component_normalization_enabled": True,
                "robust_json_parsing": True
            }
        )
    
    def _normalize_component_type(self, component_type: str) -> str:
        """
        ✅ FIX #3: Normalize component type to match available types
        
        Handles:
        - Direct matches (Button → Button)
        - Alias lookup (button → Button, text → Text)
        - Case-insensitive matching (BUTTON → Button)
        - Fuzzy matching for typos (Buton → Button)
        - Default fallback (Unknown → Text)
        """
        
        if not component_type:
            logger.warning("layout.component.empty_type")
            return "Text"
        
        original = component_type
        normalized = component_type.strip()

        resolved = normalize_component_type(normalized, fallback="Text")
        if resolved != normalized:
            self.stats['components_normalized'] += 1
            logger.debug(
                "layout.component.normalized",
                extra={"original": original, "normalized": resolved}
            )
            return resolved

        matches = get_close_matches(
            normalized,
            self.available_components,
            n=1,
            cutoff=0.6
        )
        
        if matches:
            result = matches[0]
            logger.warning(
                "layout.component.fuzzy_match",
                extra={
                    "original": original,
                    "matched": result,
                    "confidence": "medium"
                }
            )
            self.stats['components_normalized'] += 1
            return result
        
        # Strategy 5: Default fallback
        logger.warning(
            "layout.component.unknown_type",
            extra={
                "original": original,
                "defaulting_to": "Text"
            }
        )
        self.stats['components_normalized'] += 1
        return "Text"
    
    @trace_async("layout.generation")
    async def generate(
        self,
        architecture: ArchitectureDesign,
        screen_id: str
    ) -> Tuple[EnhancedLayoutDefinition, Dict[str, Any]]:
        """
        Generate layout for a specific screen.
        
        Args:
            architecture: Complete architecture design
            screen_id: Screen to generate layout for
            
        Returns:
            Tuple of (EnhancedLayoutDefinition, metadata)
            
        Raises:
            LayoutGenerationError: If generation fails
        """
        self.stats['total_requests'] += 1
        
        # Find the screen
        screen = None
        for s in architecture.screens:
            if s.id == screen_id:
                screen = s
                break
        
        if not screen:
            raise LayoutGenerationError(f"Screen '{screen_id}' not found in architecture")
        
        with log_context(operation="layout_generation", screen_id=screen_id):
            logger.info(
                f"📐 layout.generation.started",
                extra={
                    "screen_name": screen.name,
                    "components": len(screen.components)
                }
            )
            
            # Try LLM first
            layout = None
            metadata = {}
            used_heuristic = False
            
            try:
                layout_data, metadata = await self._generate_with_llm(
                    screen=screen,
                    architecture=architecture
                )
                
                # ✅ Convert to enhanced components with normalization
                components = await self._convert_to_enhanced_components(
                    layout_data.get('components', []),
                    screen_id
                )
                
                self.stats['llama3_successes'] += 1
                logger.info(
                    "✅ layout.llm.success",
                    extra={
                        "components": len(components),
                        "provider": metadata.get('provider', 'llama3')
                    }
                )
                
            except Exception as llm_error:
                logger.warning(
                    "⚠️ layout.llm.failed",
                    extra={"error": str(llm_error)},
                    exc_info=llm_error
                )
                
                # Fall back to heuristic
                logger.info("🛡️ layout.fallback.initiating")
                
                try:
                    components = await self._generate_heuristic_layout(screen)
                    metadata = {
                        'generation_method': 'heuristic',
                        'fallback_reason': str(llm_error),
                        'provider': 'heuristic',
                        'tokens_used': 0,
                        'api_duration_ms': 0
                    }
                    
                    used_heuristic = True
                    self.stats['heuristic_fallbacks'] += 1
                    
                    logger.info(
                        "✅ layout.heuristic.success",
                        extra={"components": len(components)}
                    )
                    
                except Exception as heuristic_error:
                    logger.error(
                        "❌ layout.heuristic.failed",
                        extra={"error": str(heuristic_error)},
                        exc_info=heuristic_error
                    )
                    
                    self.stats['failed'] += 1
                    raise LayoutGenerationError(
                        f"Both LLM and heuristic generation failed. "
                        f"LLM: {llm_error}, Heuristic: {heuristic_error}"
                    )
            
            # Resolve collisions
            components = await self._resolve_collisions(components)
            
            # Create layout definition
            layout = EnhancedLayoutDefinition(
                screen_id=screen_id,
                canvas=self._get_default_canvas(),
                components=components,
                layout_metadata=metadata
            )
            
            # Validate layout
            logger.info("🔍 layout.validation.starting")
            
            try:
                is_valid, warnings = await layout_validator.validate(layout)
                
                error_count = sum(1 for w in warnings if w.level == "error")
                warning_count = sum(1 for w in warnings if w.level == "warning")
                
                if not is_valid:
                    logger.error(
                        "❌ layout.validation.failed",
                        extra={
                            "errors": error_count,
                            "warnings": warning_count
                        }
                    )
                    # Don't fail - validation warnings are informational
                
                logger.info(
                    "✅ layout.validation.completed",
                    extra={
                        "warnings": warning_count,
                        "used_heuristic": used_heuristic
                    }
                )
                
            except Exception as validation_error:
                logger.warning(
                    "⚠️ layout.validation.error",
                    extra={"error": str(validation_error)}
                )
            
            # Update metadata
            metadata.update({
                'used_heuristic': used_heuristic,
                'components_normalized': self.stats['components_normalized'],
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z",
                'json_fixes_applied': self.stats['json_fixes_applied']
            })
            
            self.stats['successful'] += 1
            
            logger.info(
                "🎉 layout.generation.completed",
                extra={
                    "screen": screen.name,
                    "components": len(components),
                    "used_heuristic": used_heuristic,
                    "components_normalized": self.stats['components_normalized']
                }
            )
            
            return layout, metadata
    
    async def _generate_with_llm(
        self,
        screen: ScreenDefinition,
        architecture: ArchitectureDesign
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate layout using LLM orchestrator with retries"""
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"🔄 layout.llm.attempt",
                    extra={
                        "attempt": attempt,
                        "screen": screen.name
                    }
                )
                
                # Determine primary action
                primary_action = "view content"
                if any(has_component_event(comp, 'onPress') for comp in screen.components):
                    primary_action = "button interaction"
                elif any(is_input_component(comp) for comp in screen.components):
                    primary_action = "text input"
                
                # Format prompt
                screen_architecture_payload = self._sanitize_for_json({
                    'id': screen.id,
                    'name': screen.name,
                    'purpose': screen.purpose
                })
                system_prompt, user_prompt = prompts.LAYOUT_GENERATE.format(
                    components=", ".join(self.available_components),
                    prompt=f"Layout for {screen.name}",
                    screen_architecture=json.dumps(screen_architecture_payload, indent=2),
                    required_components=", ".join(screen.components),
                    primary_action=primary_action
                )
                             
                # Create messages
                messages = [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt)
                ]
                
                # Call LLM via orchestrator
                start_time = asyncio.get_event_loop().time()
                
                response = await self.orchestrator.generate(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096
                )

                dump_path = None
                if getattr(settings, "LAYOUT_LLM_DEBUG", False):
                    if hasattr(self, '_dump_raw_llm_response') and callable(getattr(self, '_dump_raw_llm_response')):
                        try:
                            debug_dir = getattr(settings, 'LAYOUT_LLM_DEBUG_DIR', './debug/layout_dumps')
                            if debug_dir:
                                dump_path = self._dump_raw_llm_response(
                                    screen_id=screen.id,
                                    screen_name=screen.name,
                                    attempt=attempt,
                                    provider=response.provider.value,
                                    content=response.content
                                )
                            else:
                                logger.warning(
                                    "⚠️ layout.llm.debug_dir_not_set",
                                    extra={"note": "LAYOUT_LLM_DEBUG_DIR not configured"}
                                )
                        except Exception as dump_error:
                            logger.warning(
                                "⚠️ layout.llm.dump_failed",
                                extra={
                                "error": str(dump_error),
                                "error_type": type(dump_error).__name__
                                }
                            )
                            dump_path = None
                    
                    else:
                        logger.warning(
                            "⚠️ layout.llm.dump_method_missing",
                            extra={"note": "_dump_raw_llm_response not available on this instance"}
                        )
                    
                api_duration = int((asyncio.get_event_loop().time() - start_time) * 1000)
                
                logger.debug(
                    "layout.llm.response_received",
                    extra={
                        "response_length": len(response.content),
                        "api_duration_ms": api_duration,
                        "provider": response.provider.value
                    }
                )
                
                # Parse response with robust JSON handling
                layout_data = await self._robust_parse_layout_json(response.content)
                
                # Validate layout data structure
                if not isinstance(layout_data, dict):
                    raise LayoutGenerationError(f"Expected dict but got {type(layout_data)}")
                
                if 'components' not in layout_data:
                    logger.warning("Layout data missing 'components' key, creating empty list")
                    layout_data['components'] = []
                
                # Build metadata
                metadata = {
                    'generation_method': 'llm',
                    'provider': response.provider.value,
                    'tokens_used': response.tokens_used,
                    'api_duration_ms': api_duration,
                    'screen_id': screen.id,
                    'screen_name': screen.name,
                    'raw_response_dump': dump_path
                }
                
                return layout_data, metadata
                
            except Exception as e:
                last_error = e
                
                logger.warning(
                    f"⚠️ layout.llm.retry",
                    extra={
                        "attempt": attempt,
                        "error": str(e)[:200],
                        "will_retry": attempt < self.max_retries
                    }
                )
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "❌ layout.llm.exhausted",
                        extra={
                            "total_attempts": attempt,
                            "final_error": str(last_error)
                        }
                    )
                    raise last_error
        
        raise last_error or LayoutGenerationError("All retries failed")
    
    async def _robust_parse_layout_json(self, response_text: str) -> Dict[str, Any]:
        """
        🔧 FIXED: Robust JSON parsing with multiple fallback strategies
        """
        original_text = response_text[:500] + "..." if len(response_text) > 500 else response_text
        logger.debug(f"Raw LLM response (first 500 chars): {original_text}")
        
        # Strategy 1: Clean and parse directly
        cleaned_text = self._clean_json_response(response_text)
        
        try:
            result = json.loads(cleaned_text)
            logger.debug(f"✅ JSON parsed successfully: {len(cleaned_text)} chars")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed, trying advanced fixes: {e}")
            self.stats['json_fixes_applied'] += 1
        
        # Strategy 2: Extract JSON object from text
        extracted_json = self._extract_json_object(cleaned_text)
        if extracted_json != cleaned_text:
            try:
                result = json.loads(extracted_json)
                logger.debug(f"✅ JSON extracted and parsed: {len(extracted_json)} chars")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Extracted JSON parse failed: {e}")
        
        # Strategy 3: Try to repair common issues
        repaired_json = self._repair_json(cleaned_text)
        try:
            result = json.loads(repaired_json)
            logger.debug(f"✅ JSON repaired and parsed: {len(repaired_json)} chars")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON repair failed: {e}")
            
            # Strategy 4: Create minimal valid JSON as last resort
            logger.warning("Creating minimal valid JSON as fallback")
            return self._create_minimal_layout_json()
    
    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response from common formatting issues"""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
            # Find end of first line (language spec)
            lines = text.split('\n')
            if lines and ('```' in lines[0]):
                # Remove first line if it's just ```
                lines = lines[1:]
                text = '\n'.join(lines)
            
            # Remove trailing ```
            if text.endswith('```'):
                text = text[:-3].strip()
        
        # Remove any surrounding text before first { and after last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
        
        # Fix common Llama3 issues
        text = self._fix_llama3_json_issues(text)
        
        return text.strip()
    
    def _fix_llama3_json_issues(self, text: str) -> str:
        """Fix specific JSON issues from Llama3"""
        
        # Fix 1: Double curly braces {{ }} -> { }
        text = text.replace('{{', '{').replace('}}', '}')
        
        # Fix 2: Python-style booleans and null
        text = text.replace(': True', ': true')
        text = text.replace(': False', ': false')
        text = text.replace(': None', ': null')
        text = text.replace(': true,', ': true,')
        text = text.replace(': false,', ': false,')
        
        # Fix 3: Single quotes in JSON (careful to avoid string content)
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Only replace single quotes that appear to be JSON delimiters
            # This is a simple heuristic - not perfect but works for most cases
            if "':" in line or ": '" in line or "['" in line or ", '" in line:
                # More careful single quote replacement
                # Look for patterns like 'key': or : 'value'
                line = re.sub(r"'(\w+)':", r'"\1":', line)
                line = re.sub(r": '([^']*)'", r': "\1"', line)
                line = re.sub(r"\[('([^']*)')\]", r'["\1"]', line)
                line = re.sub(r", '([^']*)'", r', "\1"', line)
            
            fixed_lines.append(line)
        
        text = '\n'.join(fixed_lines)
        
        # Fix 4: Trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix 5: Unquoted property names (simple cases)
        # Match patterns like: property: value
        text = re.sub(r'(\s*)(\w+)(\s*:\s*)', r'\1"\2"\3', text)
        
        return text
    
    def _extract_json_object(self, text: str) -> str:
        """Extract JSON object from potentially malformed text"""
        # Look for the outermost JSON object
        depth = 0
        start = -1
        end = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    end = i
                    break
        
        if start != -1 and end != -1:
            extracted = text[start:end + 1]
            logger.debug(f"Extracted JSON object: {extracted[:100]}...")
            return extracted
        
        # If no complete object found, try to find array
        depth = 0
        start = -1
        end = -1
        
        for i, char in enumerate(text):
            if char == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0 and start != -1:
                    end = i
                    break
        
        if start != -1 and end != -1:
            extracted = text[start:end + 1]
            logger.debug(f"Extracted JSON array: {extracted[:100]}...")
            return f'{{"components": {extracted}}}'
        
        return text  # Return original if nothing extracted
    
    def _repair_json(self, text: str) -> str:
        """Attempt to repair JSON with more aggressive fixes"""
        
        # Remove any non-JSON content at beginning/end
        text = text.strip()
        
        # Ensure it starts with { or [
        if not text.startswith(('{', '[')):
            # Try to find first {
            brace_pos = text.find('{')
            bracket_pos = text.find('[')
            
            if brace_pos != -1 and (bracket_pos == -1 or brace_pos < bracket_pos):
                text = text[brace_pos:]
            elif bracket_pos != -1:
                text = text[bracket_pos:]
        
        # Ensure it ends with } or ]
        if not text.endswith(('}', ']')):
            # Try to find last }
            brace_pos = text.rfind('}')
            bracket_pos = text.rfind(']')
            
            if brace_pos != -1 and (bracket_pos == -1 or brace_pos > bracket_pos):
                text = text[:brace_pos + 1]
            elif bracket_pos != -1:
                text = text[:bracket_pos + 1]
        
        # Balance braces/brackets if needed
        open_braces = text.count('{')
        close_braces = text.count('}')
        
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            text = '{' * (close_braces - open_braces) + text
        
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            text = '[' * (close_brackets - open_brackets) + text
        
        # Final cleanup
        text = self._fix_llama3_json_issues(text)
        
        return text
    
    def _create_minimal_layout_json(self) -> Dict[str, Any]:
        """Create minimal valid JSON layout as last resort"""
        logger.warning("Creating minimal layout JSON as fallback")
        
        return {
            "components": [
                {
                    "id": "fallback_title",
                    "type": "Text",
                    "properties": {
                        "value": "Layout Generation",
                        "fontSize": 24
                    },
                    "position": {"x": 50, "y": 100},
                    "constraints": {"width": 275, "height": 40}
                },
                {
                    "id": "fallback_content",
                    "type": "Text",
                    "properties": {
                        "value": "Generated layout",
                        "fontSize": 16
                    },
                    "position": {"x": 50, "y": 160},
                    "constraints": {"width": 275, "height": 60}
                }
            ]
        }

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Sanitize objects for JSON serialization
        Converts Python types to JSON-serializable equivalents
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, type):  # Catch class types (THE BUG!)
            return obj.__name__
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                str(key): self._sanitize_for_json(value) 
                for key, value in obj.items()
            }
        elif hasattr(obj, '__dict__'):  # Pydantic models, custom classes
            return self._sanitize_for_json(obj.__dict__)
        elif hasattr(obj, 'value'):  # PropertyValue objects
            return self._sanitize_for_json(obj.value)
        else:
            return str(obj)  # Fallback to string
    
    async def _convert_to_enhanced_components(
        self,
        components_data: List[Dict[str, Any]],
        screen_id: str
    ) -> List[EnhancedComponentDefinition]:
        """
        Convert LLM's component data to enhanced definitions with type normalization
        """
        
        enhanced_components = []
        
        for idx, comp_data in enumerate(components_data):
            try:
                comp_id = comp_data.get('id', f"comp_{screen_id}_{idx}")
                raw_comp_type = comp_data.get('type', 'Text')
                
                # ✅ Normalize component type
                comp_type = self._normalize_component_type(raw_comp_type)
                
                # Log if normalization changed the type
                if raw_comp_type != comp_type:
                    logger.debug(
                        "layout.component.normalized",
                        extra={
                            "original_type": raw_comp_type,
                            "normalized_type": comp_type,
                            "component_id": comp_id
                        }
                    )
                
                # Skip if still not in available components (shouldn't happen after normalization)
                if comp_type not in self.available_components:
                    logger.warning(
                        f"Component type still unsupported after normalization: {comp_type}, skipping"
                    )
                    continue
                
                # Initialize properties dict
                properties = {}
                
                # Handle style separately
                style_data = comp_data.get('style', {})
                position = comp_data.get('position', {'x': 0, 'y': 0})
                constraints = comp_data.get('constraints', {})
                
                # Get default dimensions
                width, height = get_component_default_dimensions(comp_type)
                
                # Calculate actual dimensions
                width_value = constraints.get('width', width)
                height_value = constraints.get('height', height)
                
                # Parse width if it's a string
                if isinstance(width_value, str):
                    if width_value == 'auto' or width_value == 'fill':
                        width_value = 280
                    elif '%' in width_value:
                        try:
                            percentage = float(width_value.strip('%'))
                            width_value = int(self.canvas_width * percentage / 100)
                        except:
                            width_value = 280
                    elif 'px' in width_value:
                        try:
                            width_value = int(width_value.strip('px'))
                        except:
                            width_value = 280
                
                # Parse height if it's a string
                if isinstance(height_value, str):
                    if height_value == 'auto':
                        height_value = 44
                    elif 'px' in height_value:
                        try:
                            height_value = int(height_value.strip('px'))
                        except:
                            height_value = 44
                
                # Create style property
                style_value = {
                    'left': position.get('x', 0),
                    'top': position.get('y', 0),
                    'width': width_value,
                    'height': height_value
                }
                
                # Merge with any style from comp_data
                if isinstance(style_data, dict):
                    style_value.update(style_data)
                
                properties['style'] = PropertyValue(
                    type="literal",
                    value=style_value
                )
                
                # Add component-specific properties from comp_data
                for key, value in comp_data.items():
                    if key not in ['id', 'type', 'style', 'position', 'constraints', 'z_index']:
                        sanitized_value = self._sanitize_for_json(value)
                        properties[key] = PropertyValue(type="literal", value=sanitized_value)
                
                # Ensure required properties exist from centralized defaults
                for prop_name, default_value in get_component_default_properties(comp_type).items():
                    if prop_name not in properties:
                        properties[prop_name] = PropertyValue(type="literal", value=default_value)
                
                # Get z-index
                z_index = comp_data.get('z_index', idx)
                
                # Create enhanced component
                enhanced = EnhancedComponentDefinition(
                    component_id=comp_id,
                    component_type=comp_type,
                    properties=properties,
                    z_index=z_index,
                    parent_id=None,
                    children_ids=[]
                )
                
                enhanced_components.append(enhanced)
                logger.debug(
                    f"Converted component: {comp_type} (from {raw_comp_type}) "
                    f"at position {style_value['left']}, {style_value['top']}"
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to convert component {idx} "
                    f"(type: {comp_data.get('type', 'unknown')}): {e}"
                )
                continue
        
        if not enhanced_components:
            raise LayoutGenerationError("No components could be converted from LLM response")
        
        return enhanced_components
    
    async def _generate_heuristic_layout(
        self,
        screen: ScreenDefinition
    ) -> List[EnhancedComponentDefinition]:
        """Generate layout using heuristic templates"""
        
        logger.info(
            "🛡️ layout.heuristic.generating",
            extra={"screen": screen.name}
        )
        
        components = []
        current_y = self.safe_area_top + 20
        
        for idx, raw_comp_type in enumerate(screen.components):
            # ✅ Normalize component type in heuristic too
            comp_type = self._normalize_component_type(raw_comp_type)
            
            if comp_type not in self.available_components:
                logger.warning(
                    f"Unsupported component type in heuristic after normalization: {comp_type}"
                )
                continue
            
            width, height = get_component_default_dimensions(comp_type)
            x = (self.canvas_width - width) // 2
            
            comp_id = f"{comp_type.lower()}_{idx}"
            
            # Base properties
            properties = {
                'style': PropertyValue(
                    type="literal",
                    value={
                        'left': x,
                        'top': current_y,
                        'width': width,
                        'height': height
                    }
                )
            }
            
            # Add component-specific properties from centralized defaults
            for prop_name, default_value in get_component_default_properties(comp_type).items():
                properties[prop_name] = PropertyValue(type="literal", value=default_value)
            
            try:
                component = EnhancedComponentDefinition(
                    component_id=comp_id,
                    component_type=comp_type,
                    properties=properties,
                    z_index=idx
                )
                
                components.append(component)
                logger.debug(f"Heuristic created component: {comp_type}")
                
            except Exception as e:
                logger.error(f"Failed to create {comp_type} in heuristic: {e}")
                # Skip this component but continue with others
                continue
            
            # Move to next position
            current_y += height + 16
        
        if not components:
            raise LayoutGenerationError(
                f"No components could be generated for screen: {screen.name}"
            )
        
        logger.info(
            "✅ layout.heuristic.generated",
            extra={"components": len(components)}
        )
        
        return components
    
    async def _resolve_collisions(
        self,
        components: List[EnhancedComponentDefinition]
    ) -> List[EnhancedComponentDefinition]:
        """Detect and resolve component collisions"""
        
        if len(components) <= 1:
            return components
        
        logger.debug("Checking for collisions...")
        
        # Check for collisions
        has_collisions = False
        for i, comp1 in enumerate(components):
            bounds1 = self._get_component_bounds(comp1)
            if not bounds1:
                continue
            
            for comp2 in components[i+1:]:
                bounds2 = self._get_component_bounds(comp2)
                if not bounds2:
                    continue
                
                if self._rectangles_overlap(bounds1, bounds2):
                    has_collisions = True
                    break
            
            if has_collisions:
                break
        
        if not has_collisions:
            logger.debug("No collisions detected")
            return components
        
        logger.info(f"⚠️ Collisions detected, resolving...")
        self.stats['collisions_resolved'] += 1
        
        # Simple vertical stack layout
        current_y = self.safe_area_top + 20
        
        for component in components:
            style_prop = component.properties.get('style')
            if not style_prop or style_prop.type != "literal":
                continue
            
            style = style_prop.value
            width = style.get('width', 280)
            height = style.get('height', 44)
            
            # Center horizontally
            x = (self.canvas_width - width) // 2
            
            # Update position
            style['left'] = x
            style['top'] = current_y
            
            # Move to next position
            current_y += height + 16
        
        logger.info(f"✅ Collisions resolved: stacked vertically")
        
        return components
    
    def _get_component_bounds(
        self,
        component: EnhancedComponentDefinition
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get component bounding rectangle"""
        style_prop = component.properties.get('style')
        if not style_prop or style_prop.type != "literal":
            return None
        
        style = style_prop.value
        if not isinstance(style, dict):
            return None
        
        left = style.get('left', 0)
        top = style.get('top', 0)
        width = style.get('width', 0)
        height = style.get('height', 0)
        
        return (left, top, left + width, top + height)
    
    def _rectangles_overlap(
        self,
        rect1: Tuple[int, int, int, int],
        rect2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two rectangles overlap"""
        l1_x, l1_y, r1_x, r1_y = rect1
        l2_x, l2_y, r2_x, r2_y = rect2
        
        if r1_x <= l2_x or r2_x <= l1_x:
            return False
        if r1_y <= l2_y or r2_y <= l1_y:
            return False
        
        return True
    
    def _get_default_canvas(self) -> Dict[str, Any]:
        """Get default canvas configuration"""
        return {
            "width": self.canvas_width,
            "height": self.canvas_height,
            "background_color": "#FFFFFF",
            "safe_area_insets": {
                "top": self.safe_area_top,
                "bottom": self.safe_area_bottom,
                "left": 0,
                "right": 0
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        total = self.stats['total_requests']
        
        return {
            **self.stats,
            'success_rate': (self.stats['successful'] / total * 100) if total > 0 else 0,
            'heuristic_fallback_rate': (self.stats['heuristic_fallbacks'] / total * 100) if total > 0 else 0,
            'llama3_success_rate': (self.stats['llama3_successes'] / total * 100) if total > 0 else 0,
            'collisions_resolved': self.stats['collisions_resolved'],
            'components_normalized': self.stats['components_normalized'],
            'json_fixes_applied': self.stats['json_fixes_applied']
        }
    
    async def test_json_parsing(self):
        """Test JSON parsing with common failure cases"""
        
        test_cases = [
            # Single quotes
            "{'components': [{'type': 'Button', 'checked': true}]}",
            # Markdown with single quotes
            "```json\n{'components': []}\n```",
            # Python-style booleans
            "{components: [{type: 'Checkbox', checked: True}]}",
            # Trailing comma
            "{'components': [],}",
        ]
        
        for i, test in enumerate(test_cases):
            print(f"\nTest case {i+1}:")
            print(f"Input: {test}")
            try:
                result = await self._robust_parse_layout_json(test)
                print(f"✅ Success: {result}")
            except Exception as e:
                print(f"❌ Failed: {e}")

def _dump_raw_llm_response(
    self,
    *,
    screen_id: str,
    screen_name: str,
    attempt: int,
    provider: str,
    content: str
) -> str:
    """
    Dump raw LLM response to disk for debugging.
    Returns the file path.
    
    ✅ Enhanced with safety checks and directory creation
    """
    import os
    from pathlib import Path
    
    try:
        # ✅ Safety Check 1: Get debug directory with fallback
        debug_dir = getattr(settings, 'LAYOUT_LLM_DEBUG_DIR', './debug/layout_dumps')
        
        # ✅ Safety Check 2: Ensure directory exists (create if not)
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        
        # ✅ Safety Check 3: Verify directory is writable
        if not os.access(debug_dir, os.W_OK):
            logger.warning(
                "⚠️ layout.llm.debug_dir_not_writable",
                extra={"path": debug_dir}
            )
            # Fallback to temp directory
            import tempfile
            debug_dir = tempfile.gettempdir()
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
        
        # ✅ Safety Check 4: Generate safe filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Sanitize screen_id and screen_name for filename
        safe_screen_id = "".join(c for c in screen_id if c.isalnum() or c in '-_')
        safe_screen_name = "".join(c for c in screen_name if c.isalnum() or c in '-_')[:50]
        filename = f"layout_{safe_screen_id}_{safe_screen_name}_attempt{attempt}_{timestamp}.json.txt"
        path = os.path.join(debug_dir, filename)
        
        # ✅ Safety Check 5: Write file with error handling
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # ✅ Safety Check 6: Verify file was created
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            logger.warning(
                "🧪 layout.llm.raw_dumped",
                extra={
                    "screen_id": screen_id,
                    "screen_name": screen_name,
                    "attempt": attempt,
                    "provider": provider,
                    "path": path,
                    "file_size_bytes": file_size
                }
            )
            return path
        else:
            logger.error(
                "❌ layout.llm.dump_failed",
                extra={"reason": "File not created", "path": path}
            )
            return None
            
    except PermissionError as e:
        logger.error(
            "❌ layout.llm.permission_error",
            extra={"error": str(e), "path": debug_dir}
        )
        return None
    except OSError as e:
        logger.error(
            "❌ layout.llm.os_error",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return None
    except Exception as e:
        logger.error(
            "❌ layout.llm.dump_unexpected_error",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        return None
        
# Global layout generator instance
layout_generator = LayoutGenerator()