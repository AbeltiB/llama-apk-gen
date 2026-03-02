"""
Blockly Generator - Phase 3 - FIXED VERSION
Uses LLM Orchestrator (Llama3 → Heuristic fallback)

Generates visual programming blocks for app logic.
"""
import json
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import re

from app.config import settings
from app.models.schemas.architecture import ArchitectureDesign
from app.models.schemas.layout import EnhancedLayoutDefinition
from app.models.schemas.component_catalog import get_component_event
from app.models.prompts import prompts
from app.services.generation.blockly_validator import blockly_validator
from app.llm.orchestrator import LLMOrchestrator
from app.llm.base import LLMMessage
from app.utils.logging import get_logger, log_context, trace_async

logger = get_logger(__name__)


class BlocklyGenerationError(Exception):
    """Base exception for Blockly generation errors"""
    pass


class BlocklyGenerator:
    """
    Phase 3 Blockly Generator using LLM Orchestrator - FIXED VERSION
    
    Generation Flow:
    1. 🎯 Try Llama3 via orchestrator
    2. 🔄 Retry with corrections if needed
    3. 🛡️ Fall back to heuristic if all retries fail
    4. ✅ Validate result
    
    Features:
    - Llama3 as primary LLM
    - Automatic heuristic template fallback
    - Comprehensive validation
    - Robust JSON parsing with multiple fallback strategies
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
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'heuristic_fallbacks': 0,
            'llama3_successes': 0,
            'json_fixes_applied': 0
        }
        
        logger.info(
            "blockly.generator.initialized",
            extra={
                "llm_provider": "llama3",
                "heuristic_fallback_enabled": True,
                "robust_json_parsing": True
            }
        )
    
    @trace_async("blockly.generation")
    async def generate(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Dict[str, Any]:
        """
        Generate Blockly blocks for application.
        
        Args:
            architecture: Complete architecture design
            layouts: Map of screen_id -> layout
            
        Returns:
            Blockly definition dictionary
            
        Raises:
            BlocklyGenerationError: If generation fails
        """
        self.stats['total_requests'] += 1
        
        with log_context(operation="blockly_generation"):
            logger.info(
                "🧩 blockly.generation.started",
                extra={
                    "screens": len(architecture.screens) if architecture and hasattr(architecture, 'screens') else 0,
                    "layouts": len(layouts) if layouts else 0
                }
            )
            
            # Validate inputs
            if not architecture:
                logger.error("🚨 blockly.generation.architecture_missing")
                return self._create_empty_blockly("No architecture provided")
            
            if not layouts:
                logger.warning("⚠️ blockly.generation.no_layouts")
            
            # Initialize blockly structure with defaults
            blockly = None
            used_heuristic = False
            generation_metadata = {
                'generation_method': 'unknown',
                'provider': 'unknown',
                'tokens_used': 0,
                'api_duration_ms': 0,
                'used_heuristic': False,
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z"
            }
            
            try:
                # Try LLM first
                try:
                    blockly, llm_metadata = await self._generate_with_llm(
                        architecture=architecture,
                        layouts=layouts
                    )
                    
                    if blockly and isinstance(blockly, dict):
                        generation_metadata.update(llm_metadata)
                        generation_metadata['generation_method'] = 'llm'
                        self.stats['llama3_successes'] += 1
                        logger.info(
                            "✅ blockly.llm.success",
                            extra={
                                "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                                "provider": llm_metadata.get('provider', 'llama3')
                            }
                        )
                    else:
                        raise BlocklyGenerationError("LLM generated invalid blockly structure")
                        
                except Exception as llm_error:
                    logger.warning(
                        "⚠️ blockly.llm.failed",
                        extra={"error": str(llm_error)[:200]},
                        exc_info=False  # Don't log full traceback for expected failures
                    )
                    
                    # Fall back to heuristic
                    logger.info("🛡️ blockly.fallback.initiating")
                    
                    try:
                        blockly = await self._generate_heuristic_blockly(
                            architecture=architecture,
                            layouts=layouts
                        )
                        
                        if blockly and isinstance(blockly, dict):
                            generation_metadata.update({
                                'generation_method': 'heuristic',
                                'provider': 'heuristic',
                                'fallback_reason': str(llm_error)[:200],
                                'used_heuristic': True
                            })
                            
                            used_heuristic = True
                            self.stats['heuristic_fallbacks'] += 1
                            
                            logger.info(
                                "✅ blockly.heuristic.success",
                                extra={
                                    "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                                    "variables": len(blockly.get('variables', []))
                                }
                            )
                        else:
                            raise BlocklyGenerationError("Heuristic generated invalid blockly structure")
                            
                    except Exception as heuristic_error:
                        logger.error(
                            "❌ blockly.heuristic.failed",
                            extra={"error": str(heuristic_error)},
                            exc_info=heuristic_error
                        )
                        
                        # Create minimal fallback blockly
                        blockly = self._create_minimal_blockly(architecture, layouts)
                        generation_metadata.update({
                            'generation_method': 'fallback',
                            'provider': 'fallback',
                            'fallback_reason': f"LLM: {llm_error}, Heuristic: {heuristic_error}",
                            'used_heuristic': True
                        })
                        
                        used_heuristic = True
                        self.stats['heuristic_fallbacks'] += 1
                        
                        logger.warning(
                            "⚠️ blockly.using_minimal_fallback",
                            extra={"reason": "Both LLM and heuristic failed"}
                        )
                
                # Ensure blockly is a valid dictionary
                if not blockly or not isinstance(blockly, dict):
                    logger.error("🚨 blockly.generation.invalid_output", extra={"type": type(blockly).__name__ if blockly else "None"})
                    blockly = self._create_empty_blockly("Generation returned invalid type")
                    generation_metadata['generation_method'] = 'emergency_fallback'
                
                # Add metadata to blockly structure
                if isinstance(blockly, dict):
                    blockly['metadata'] = generation_metadata
                    
                    # Ensure required structure exists
                    if 'blocks' not in blockly:
                        blockly['blocks'] = {'blocks': []}
                    elif not isinstance(blockly['blocks'], dict):
                        blockly['blocks'] = {'blocks': []}
                    
                    if 'variables' not in blockly:
                        blockly['variables'] = []
                    elif not isinstance(blockly['variables'], list):
                        blockly['variables'] = []
                    
                    if 'custom_blocks' not in blockly:
                        blockly['custom_blocks'] = []
                
                # Validate Blockly
                logger.info("🔍 blockly.validation.starting")
                
                try:
                    is_valid, warnings = await blockly_validator.validate(blockly)
                    
                    error_count = sum(1 for w in warnings if w.level == "error")
                    warning_count = sum(1 for w in warnings if w.level == "warning")
                    
                    if not is_valid:
                        logger.error(
                            "❌ blockly.validation.failed",
                            extra={
                                "errors": error_count,
                                "warnings": warning_count
                            }
                        )
                        # Add validation warnings to metadata
                        blockly['metadata']['validation_errors'] = error_count
                        blockly['metadata']['validation_warnings'] = warning_count
                    else:
                        logger.info(
                            "✅ blockly.validation.completed",
                            extra={
                                "warnings": warning_count,
                                "errors": error_count,
                                "used_heuristic": used_heuristic
                            }
                        )
                        
                except Exception as validation_error:
                    logger.warning(
                        "⚠️ blockly.validation.error",
                        extra={"error": str(validation_error)}
                    )
                    # Add validation error to metadata but don't fail
                    blockly['metadata']['validation_error'] = str(validation_error)
                
                self.stats['successful'] += 1
                
                logger.info(
                    "🎉 blockly.generation.completed",
                    extra={
                        "blocks": len(blockly.get('blocks', {}).get('blocks', [])),
                        "variables": len(blockly.get('variables', [])),
                        "used_heuristic": used_heuristic,
                        "method": generation_metadata.get('generation_method', 'unknown')
                    }
                )
                
                return blockly
                
            except Exception as e:
                logger.error(
                    "💥 blockly.generation.critical_error",
                    extra={
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    },
                    exc_info=e
                )
                
                self.stats['failed'] += 1
                
                # Always return a valid blockly structure, even if empty
                fallback_blockly = self._create_empty_blockly(f"Critical error: {str(e)[:200]}")
                fallback_blockly['metadata'] = {
                    'generation_method': 'critical_error_fallback',
                    'provider': 'fallback',
                    'error': str(e),
                    'generated_at': datetime.now(timezone.utc).isoformat() + "Z",
                    'used_heuristic': True
                }
                
                return fallback_blockly
    
    async def _generate_with_llm(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate Blockly using LLM orchestrator with retries"""
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "🔄 blockly.llm.attempt",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.max_retries
                    }
                )
                
                # Extract component events from layouts
                component_events = self._extract_component_events(layouts)
                
                # Format prompt
                system_prompt, user_prompt = prompts.BLOCKLY_GENERATE.format(
                    architecture=json.dumps(architecture.dict(), indent=2),
                    layout=json.dumps({
                        screen_id: self._layout_to_dict(layout)
                        for screen_id, layout in (layouts or {}).items()
                    }, indent=2),
                    component_events=json.dumps(component_events, indent=2)
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
                
                api_duration = int((asyncio.get_event_loop().time() - start_time) * 1000)
                
                logger.debug(
                    "blockly.llm.response_received",
                    extra={
                        "response_length": len(response.content),
                        "api_duration_ms": api_duration,
                        "provider": response.provider.value
                    }
                )
                
                # Log first 500 chars for debugging
                if response.content and len(response.content) > 0:
                    logger.debug(
                        f"Raw LLM response (first 500 chars): {response.content[:500]}",
                        extra={"response_preview": response.content[:500]}
                    )
                
                # Parse response with robust JSON handling
                blockly_data = await self._robust_parse_blockly_json(response.content)
                
                # Validate basic structure
                if not isinstance(blockly_data, dict):
                    raise BlocklyGenerationError(f"LLM response is not a dict: {type(blockly_data)}")

                if not self._has_meaningful_logic(blockly_data):
                    raise BlocklyGenerationError("LLM returned empty Blockly logic")
                
                # Build metadata
                metadata = {
                    'generation_method': 'llm',
                    'provider': response.provider.value,
                    'tokens_used': response.tokens_used,
                    'api_duration_ms': api_duration,
                    'attempt': attempt,
                    'response_length': len(response.content)
                }
                
                return blockly_data, metadata
                
            except Exception as e:
                last_error = e
                
                logger.warning(
                    f"⚠️ blockly.llm.retry",
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
                        "❌ blockly.llm.exhausted",
                        extra={
                            "total_attempts": attempt,
                            "final_error": str(last_error)[:500]
                        }
                    )
                    raise BlocklyGenerationError(f"All {attempt} retries failed: {last_error}")
        
        raise BlocklyGenerationError(f"All retries failed: {last_error}")
    
    def _has_meaningful_logic(self, blockly_data: Dict[str, Any]) -> bool:
        """Return True when blockly payload has at least one actionable block."""
        if not isinstance(blockly_data, dict):
            return False

        workspace_blocks = blockly_data.get('workspace', {}).get('blocks', [])
        nested_blocks = blockly_data.get('blocks', {}).get('blocks', [])
        custom_blocks = blockly_data.get('custom_blocks', [])

        if isinstance(workspace_blocks, list) and len(workspace_blocks) > 0:
            return True
        if isinstance(nested_blocks, list) and len(nested_blocks) > 0:
            return True
        if isinstance(custom_blocks, list) and len(custom_blocks) > 0:
            return True
        return False

    async def _robust_parse_blockly_json(self, response_text: str) -> Dict[str, Any]:
        """
        🔧 FIXED: Robust JSON parsing for Blockly with multiple fallback strategies
        """
        if not response_text or response_text.strip() == "":
            raise BlocklyGenerationError("Empty response from LLM")
        
        original_text = response_text[:500] + "..." if len(response_text) > 500 else response_text
        logger.debug(f"Raw LLM response (first 500 chars): {original_text}")
        
        # Strategy 1: Clean and parse directly
        cleaned_text = self._clean_json_response(response_text)
        
        try:
            result = json.loads(cleaned_text)
            logger.debug(f"✅ Blockly JSON parsed successfully: {len(cleaned_text)} chars")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed, trying advanced fixes: {e}")
            self.stats['json_fixes_applied'] += 1
        
        # Strategy 2: Extract JSON object from text
        extracted_json = self._extract_json_object(cleaned_text)
        if extracted_json != cleaned_text:
            try:
                result = json.loads(extracted_json)
                logger.debug(f"✅ Blockly JSON extracted and parsed: {len(extracted_json)} chars")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Extracted JSON parse failed: {e}")
        
        # Strategy 3: Try to repair common issues
        repaired_json = self._repair_json(cleaned_text)
        try:
            result = json.loads(repaired_json)
            logger.debug(f"✅ Blockly JSON repaired and parsed: {len(repaired_json)} chars")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Blockly JSON repair failed: {e}")
            
            # Strategy 4: Create minimal valid Blockly JSON as last resort
            logger.warning("Creating minimal Blockly JSON as fallback")
            return self._create_minimal_blockly_json()
    
    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response from common formatting issues"""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
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
            return f'{{"blocks": {extracted}}}'
        
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
    
    def _create_minimal_blockly_json(self) -> Dict[str, Any]:
        """Create minimal valid Blockly JSON as last resort"""
        logger.warning("Creating minimal Blockly JSON as fallback")
        
        return {
            "blocks": {
                "languageVersion": 0,
                "blocks": [
                    {
                        "type": "app_start",
                        "id": "start_1",
                        "fields": {"APP_NAME": "App"}
                    }
                ]
            },
            "variables": [
                {"name": "app_state", "id": "var_1", "type": "String"}
            ],
            "custom_blocks": [],
            "json_fallback": True
        }
    
    def _layout_to_dict(self, layout: Any) -> Dict[str, Any]:
        """Normalize layout object to dict for prompt serialization."""
        if hasattr(layout, 'model_dump'):
            return layout.model_dump()
        if hasattr(layout, 'dict'):
            return layout.dict()
        if isinstance(layout, dict):
            return layout
        if isinstance(layout, list):
            return {'components': layout}
        return {'components': []}

    def _extract_layout_components(self, layout: Any) -> List[Any]:
        """Read components from layout object/dict/list payloads."""
        if layout is None:
            return []
        if hasattr(layout, 'components'):
            components = getattr(layout, 'components', [])
            return components if isinstance(components, list) else []
        if isinstance(layout, dict):
            components = layout.get('components', [])
            return components if isinstance(components, list) else []
        if isinstance(layout, list):
            return layout
        return []

    def _extract_component_event_name(self, component: Any) -> str:
        """Resolve event name from typed or dict component payloads."""
        if hasattr(component, 'component_type'):
            return get_component_event(getattr(component, 'component_type', ''))
        if isinstance(component, dict):
            component_type = (
                component.get('component_type')
                or component.get('type')
                or component.get('component')
                or component.get('name')
                or ''
            )
            return get_component_event(component_type)
        return ''

    def _extract_component_id(self, component: Any, fallback: str) -> str:
        """Resolve component id from typed or dict component payloads."""
        if hasattr(component, 'component_id'):
            return str(getattr(component, 'component_id', fallback))
        if isinstance(component, dict):
            return str(component.get('component_id') or component.get('id') or fallback)
        return fallback

    def _extract_component_events(
        self,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> List[Dict[str, str]]:
        """Extract component events from layouts."""

        events: List[Dict[str, str]] = []

        if not layouts or not isinstance(layouts, dict):
            return events

        for screen_id, layout in layouts.items():
            components = self._extract_layout_components(layout)
            if not components:
                continue

            for idx, component in enumerate(components):
                event_name = self._extract_component_event_name(component)
                if not event_name:
                    continue

                component_id = self._extract_component_id(component, fallback=f"comp_{screen_id}_{idx}")
                events.append({
                    'screen_id': str(screen_id),
                    'component_id': component_id,
                    'component_type': (
                        getattr(component, 'component_type', None)
                        if hasattr(component, 'component_type')
                        else (component.get('component_type') if isinstance(component, dict) else 'unknown')
                    ) or 'unknown',
                    'event': event_name,
                })

        return events

    async def _generate_heuristic_blockly(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Dict[str, Any]:
        """Generate Blockly using heuristic templates"""
        
        logger.info(
            "🛡️ blockly.heuristic.generating",
            extra={"screens": len(architecture.screens) if architecture and hasattr(architecture, 'screens') else 0}
        )
        
        # Extract component events
        component_events = self._extract_component_events(layouts)
        
        # Generate blocks
        blocks = []
        variables = []
        custom_blocks = []
        
        # Create event handlers for interactive components
        for idx, event_info in enumerate(component_events):
            block_id = f"event_{idx + 1}"
            
            # Create event block
            event_block = {
                'type': 'component_event',
                'id': block_id,
                'x': 20 + (idx * 200),
                'y': 20 + (idx * 100),
                'fields': {
                    'COMPONENT': event_info['component_id'],
                    'EVENT': event_info['event'],
                    'SCREEN': event_info['screen_id']
                },
                'next': None
            }
            
            blocks.append(event_block)
        
        # Add variables from state management
        if hasattr(architecture, 'state_management') and architecture.state_management:
            for idx, state in enumerate(architecture.state_management):
                variables.append({
                    'name': state.name,
                    'id': f"var_{idx + 1}",
                    'type': 'String' if 'text' in state.name.lower() else 'Number'
                })
        else:
            # Default variables
            variables = [
                {'name': 'user_input', 'id': 'var_1', 'type': 'String'},
                {'name': 'counter', 'id': 'var_2', 'type': 'Number'}
            ]
        
        # Add navigation blocks for multi-screen apps
        if hasattr(architecture, 'screens') and len(architecture.screens) > 1:
            nav_block = {
                'type': 'navigate_screen',
                'id': 'nav_1',
                'x': 20,
                'y': 300,
                'fields': {
                    'SCREEN': (getattr(architecture.screens[0], 'id', 'home') if architecture.screens else 'home')
                }
            }
            blocks.append(nav_block)
        
        blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': blocks
            },
            'variables': variables,
            'custom_blocks': custom_blocks,
            'heuristic_generated': True
        }
        
        logger.info(
            "blockly.heuristic.generated",
            extra={
                "blocks": len(blocks),
                "variables": len(variables)
            }
        )
        
        return blockly
    
    def _create_minimal_blockly(
        self,
        architecture: ArchitectureDesign,
        layouts: Dict[str, EnhancedLayoutDefinition]
    ) -> Dict[str, Any]:
        """Create minimal blockly structure when all else fails"""
        
        # Extract component events
        component_events = self._extract_component_events(layouts)
        
        blockly = {
            'blocks': {
                'languageVersion': 0,
                'blocks': [
                    {
                        'type': 'app_start',
                        'id': 'start_1',
                        'x': 20,
                        'y': 20,
                        'fields': {'APP_NAME': architecture.app_name if hasattr(architecture, 'app_name') else 'App'}
                    }
                ]
            },
            'variables': [
                {'name': 'app_state', 'id': 'var_1', 'type': 'String'},
                {'name': 'user_data', 'id': 'var_2', 'type': 'String'}
            ],
            'custom_blocks': [],
            'minimal_fallback': True,
            'component_events_count': len(component_events)
        }
        
        return blockly
    
    def _create_empty_blockly(self, reason: str = "") -> Dict[str, Any]:
        """Create an empty but valid blockly structure"""
        
        return {
            'blocks': {
                'languageVersion': 0,
                'blocks': []
            },
            'variables': [],
            'custom_blocks': [],
            'empty_fallback': True,
            'fallback_reason': reason,
            'metadata': {
                'generation_method': 'empty_fallback',
                'provider': 'fallback',
                'generated_at': datetime.now(timezone.utc).isoformat() + "Z"
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
            'json_fixes_applied': self.stats['json_fixes_applied']
        }


# Global blockly generator instance
blockly_generator = BlocklyGenerator()