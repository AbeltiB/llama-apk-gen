"""
Fixed Context Builder - Production Ready
Works with unified schemas and proper error handling
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.services.analysis.intent_schemas import IntentAnalysisResult


class ContextRelevanceScore:
    """Calculate confidence that a project is relevant to current request"""
    
    @staticmethod
    def calculate(
        project: Dict[str, Any],
        user_id: str,
        session_id: str,
        intent_result: IntentAnalysisResult
    ) -> float:
        """
        Calculate relevance score (0.0 to 1.0)
        
        Args:
            project: Project data dict
            user_id: Current user ID
            session_id: Current session ID
            intent_result: Intent analysis result
            
        Returns:
            float: Relevance score
        """
        score = 0.0
        
        # CRITICAL: Ownership verification
        if project.get('user_id') != user_id:
            return 0.0  # Wrong user - NEVER return
        
        # Session match (highest weight)
        project_metadata = project.get('metadata', {})
        if project_metadata.get('session_id') == session_id:
            score += 0.6  # Same session = very relevant
        
        # Recency (within last hour)
        updated_at = project.get('updated_at')
        if updated_at:
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            
            age_hours = (datetime.now(timezone.utc) - updated_at).total_seconds() / 3600
            if age_hours < 1:
                score += 0.3
            elif age_hours < 24:
                score += 0.1
        
        # Intent match
        if intent_result.requires_context:
            score += 0.1
        
        return min(score, 1.0)


class EnrichedContext:
    """Container for enriched context data"""
    
    def __init__(
        self,
        original_request: Dict[str, Any],
        intent_analysis: IntentAnalysisResult,
        conversation_history: List[Dict[str, Any]] = None,
        existing_project: Optional[Dict[str, Any]] = None,
        user_preferences: Dict[str, Any] = None,
        timestamp: datetime = None
    ):
        self.original_request = original_request
        self.intent_analysis = intent_analysis
        self.conversation_history = conversation_history or []
        self.existing_project = existing_project
        self.user_preferences = user_preferences or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_request": self.original_request,
            "intent_analysis": self.intent_analysis.to_dict(),
            "conversation_history": self.conversation_history,
            "existing_project": self.existing_project,
            "user_preferences": self.user_preferences,
            "timestamp": self.timestamp.isoformat()
        }


class ProductionContextBuilder:
    """
    Production-ready context builder
    
    Features:
    - Robust error handling
    - Proper schema validation
    - Works with any database manager
    - Graceful degradation
    """
    
    # Minimum confidence threshold
    MIN_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, db_manager=None):
        """
        Initialize context builder
        
        Args:
            db_manager: Database manager instance (optional)
        """
        self.db_manager = db_manager
        self.stats = {
            'total_builds': 0,
            'with_project': 0,
            'with_history': 0,
            'errors': []
        }
        
        print("✅ Production context builder initialized")
    
    async def build_context(
        self,
        user_id: str,
        session_id: str,
        prompt: str,
        intent_result: IntentAnalysisResult,
        original_request: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> EnrichedContext:
        """
        Build comprehensive enriched context
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            prompt: User's prompt
            intent_result: Intent analysis result
            original_request: Original request data
            project_id: Optional explicit project ID
            
        Returns:
            EnrichedContext - GUARANTEED to return
        """
        self.stats['total_builds'] += 1
        
        print(f"🔨 Building context for user {user_id[:8]}...")
        
        try:
            # Create base context
            context = EnrichedContext(
                original_request=original_request,
                intent_analysis=intent_result,
                conversation_history=[],
                existing_project=None,
                user_preferences={},
                timestamp=datetime.now(timezone.utc)
            )
            
            # Load conversation history (if db available)
            if self.db_manager:
                try:
                    context.conversation_history = await self._load_conversation_history(
                        user_id=user_id,
                        session_id=session_id,
                        limit=10
                    )
                    if context.conversation_history:
                        self.stats['with_history'] += 1
                        print(f"   ✓ Loaded {len(context.conversation_history)} history items")
                except Exception as e:
                    print(f"   ⚠️  Failed to load history: {e}")
                    context.conversation_history = []
            
            # Load existing project (if needed)
            if intent_result.requires_context or project_id:
                if self.db_manager:
                    try:
                        context.existing_project = await self._load_existing_project_safe(
                            user_id=user_id,
                            session_id=session_id,
                            intent_result=intent,
                            explicit_project_id=project_id
                        )
                        
                        if context.existing_project:
                            self.stats['with_project'] += 1
                            print(f"   ✓ Loaded project: {context.existing_project.get('project_name', 'Unnamed')}")
                        elif intent_result.requires_context:
                            print(f"   ⚠️  Context required but no project found")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load project: {e}")
                        self.stats['errors'].append(str(e))
                        context.existing_project = None
            
            # Load user preferences (if db available)
            if self.db_manager:
                try:
                    context.user_preferences = await self._load_user_preferences(user_id)
                    if context.user_preferences:
                        print(f"   ✓ Loaded user preferences")
                except Exception as e:
                    print(f"   ⚠️  Failed to load preferences: {e}")
                    context.user_preferences = self._get_default_preferences()
            else:
                context.user_preferences = self._get_default_preferences()
            
            print(f"✅ Context built successfully")
            return context
            
        except Exception as e:
            print(f"❌ Critical error building context: {e}")
            self.stats['errors'].append(str(e))
            
            # Return minimal safe context
            return EnrichedContext(
                original_request=original_request,
                intent_analysis=intent_result,
                conversation_history=[],
                existing_project=None,
                user_preferences=self._get_default_preferences(),
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _load_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Load recent conversation history"""
        
        if not self.db_manager:
            return []
        
        try:
            if hasattr(self.db_manager, 'get_conversation_history'):
                conversations = await self.db_manager.get_conversation_history(
                    user_id=user_id,
                    session_id=session_id,
                    limit=limit
                )
                return conversations if conversations else []
            else:
                print("   ⓘ Database manager doesn't support conversation history")
                return []
                
        except Exception as e:
            print(f"   ⚠️  Error loading conversation history: {e}")
            return []
    
    async def _load_existing_project_safe(
        self,
        user_id: str,
        session_id: str,
        intent_result: IntentAnalysisResult,
        explicit_project_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load existing project with strict validation
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            intent_result: Intent analysis result
            explicit_project_id: Optional explicit project ID
            
        Returns:
            Project dict or None
        """
        
        if not self.db_manager:
            return None
        
        try:
            # Case 1: Explicit project ID provided
            if explicit_project_id:
                if hasattr(self.db_manager, 'get_project'):
                    project = await self.db_manager.get_project(explicit_project_id)
                    
                    if not project:
                        print(f"   ⚠️  Project {explicit_project_id} not found")
                        return None
                    
                    # CRITICAL: Verify ownership
                    if project.get('user_id') != user_id:
                        print(f"   🔴 SECURITY: Project ownership violation!")
                        return None
                    
                    # Add metadata
                    project['_confidence'] = 1.0
                    project['_match_reason'] = 'explicit_project_id'
                    
                    return project
            
            # Case 2: Match by session_id
            if hasattr(self.db_manager, 'get_user_projects'):
                projects = await self.db_manager.get_user_projects(
                    user_id=user_id,
                    limit=5
                )
                
                if not projects:
                    return None
                
                # Find best match with confidence scoring
                session_matches = []
                
                for project in projects:
                    confidence = ContextRelevanceScore.calculate(
                        project=project,
                        user_id=user_id,
                        session_id=session_id,
                        intent_result=intent
                    )
                    
                    if confidence >= self.MIN_CONFIDENCE_THRESHOLD:
                        project['_confidence'] = confidence
                        project['_match_reason'] = 'session_match'
                        session_matches.append(project)
                
                if not session_matches:
                    print(f"   ⓘ No confident project match found")
                    return None
                
                # Return highest confidence match
                best_match = max(session_matches, key=lambda p: p['_confidence'])
                return best_match
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error loading project: {e}")
            return None
    
    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences"""
        
        if not self.db_manager:
            return self._get_default_preferences()
        
        try:
            if hasattr(self.db_manager, 'get_user_preferences'):
                preferences = await self.db_manager.get_user_preferences(user_id)
                return preferences if preferences else self._get_default_preferences()
            else:
                return self._get_default_preferences()
                
        except Exception as e:
            print(f"   ⚠️  Error loading preferences: {e}")
            return self._get_default_preferences()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            "theme": "light",
            "component_style": "detailed",
            "preferred_colors": {
                "primary": "#007AFF",
                "secondary": "#5856D6",
                "background": "#FFFFFF",
                "text": "#000000"
            },
            "layout_style": "modern",
            "enable_animations": True
        }
    
    def format_context_for_prompt(self, context: EnrichedContext) -> str:
        """Format enriched context into string for LLM prompts"""
        
        parts = []
        
        # Intent information
        intent = context.intent_analysis
        parts.append(f"**Intent:** {intent.intent_type.value}")
        parts.append(f"**Complexity:** {intent.complexity.value}")
        parts.append(f"**Confidence:** {intent.confidence.overall:.2f}")
        
        # Extracted entities
        if intent.extracted_entities.components:
            parts.append(f"**Components:** {', '.join(intent.extracted_entities.components)}")
        if intent.extracted_entities.features:
            parts.append(f"**Features:** {', '.join(intent.extracted_entities.features)}")
        
        # Conversation history
        if context.conversation_history:
            recent = context.conversation_history[-3:]
            history_str = "\n".join([
                f"  - {msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}"
                for msg in recent
            ])
            parts.append(f"**Recent Conversation:**\n{history_str}")
        
        # Existing project
        if context.existing_project:
            proj = context.existing_project
            confidence = proj.get('_confidence', 0.0)
            match_reason = proj.get('_match_reason', 'unknown')
            
            parts.append(f"**Existing Project:** {proj.get('project_name', 'Unnamed')}")
            parts.append(f"  - Match confidence: {confidence:.2f} ({match_reason})")
            
            if proj.get('architecture'):
                arch = proj['architecture']
                parts.append(f"  - Type: {arch.get('app_type', 'unknown')}")
                parts.append(f"  - Screens: {len(arch.get('screens', []))}")
        
        # User preferences
        if context.user_preferences:
            prefs = context.user_preferences
            parts.append(f"**User Preferences:**")
            if 'theme' in prefs:
                parts.append(f"  - Theme: {prefs['theme']}")
            if 'component_style' in prefs:
                parts.append(f"  - Style: {prefs['component_style']}")
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total = self.stats['total_builds']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'project_load_rate': (self.stats['with_project'] / total) * 100,
            'history_load_rate': (self.stats['with_history'] / total) * 100,
        }


# Global instance (optional)
context_builder = ProductionContextBuilder()


# Factory function
def create_context_builder(db_manager=None) -> ProductionContextBuilder:
    """
    Create context builder instance
    
    Args:
        db_manager: Optional database manager
        
    Returns:
        ProductionContextBuilder instance
    """
    return ProductionContextBuilder(db_manager)


# ================== USAGE EXAMPLE ==================

if __name__ == "__main__":
    import asyncio
    from intent_schemas_fixed import (
        IntentType, ComplexityLevel, ConfidenceBreakdown,
        ExtractedEntities, SafetyStatus, ActionRecommendation,
        IntentAnalysisResult
    )
    
    async def test():
        """Test context builder"""
        
        # Create mock intent result
        intent_result = IntentAnalysisResult(
            intent_type=IntentType.CREATE_APP,
            complexity=ComplexityLevel.SIMPLE,
            confidence=ConfidenceBreakdown(),
            extracted_entities=ExtractedEntities(
                components=["Button", "Text"],
                features=["counter"]
            ),
            action_recommendation=ActionRecommendation.PROCEED,
            safety_status=SafetyStatus.SAFE,
            source="test"
        )
        
        # Create builder (no db)
        builder = create_context_builder(db_manager=None)
        
        # Build context
        context = await builder.build_context(
            user_id="test_user_123",
            session_id="test_session_456",
            prompt="Create a counter app",
            intent_result=intent,
            original_request={"prompt": "Create a counter app"},
            project_id=None
        )
        
        print("\n" + "=" * 60)
        print("CONTEXT BUILT SUCCESSFULLY")
        print("=" * 60)
        print(f"Intent: {context.intent_analysis.intent_type.value}")
        print(f"Complexity: {context.intent_analysis.complexity.value}")
        print(f"History items: {len(context.conversation_history)}")
        print(f"Has project: {context.existing_project is not None}")
        print(f"Preferences: {len(context.user_preferences)} items")
        print("=" * 60 + "\n")
        
        # Show formatted prompt
        prompt_text = builder.format_context_for_prompt(context)
        print("FORMATTED FOR PROMPT:")
        print(prompt_text)
        print("\n")
    
    asyncio.run(test())