"""
Models package - Unified schema and prompt system.

Exports:
- schemas: All data models and validation
- prompts: All prompt templates
"""

from .schemas import (
    # Core types
    PropertyValue,
    ComponentStyle,
    COMPONENT_PROPERTY_SCHEMAS,
    
    # Input/Output
    AIRequest,
    ProgressUpdate,
    ErrorResponse,
    CompleteResponse,
    
    # Architecture
    ArchitectureDesign,
    ScreenDefinition,
    NavigationStructure,
    StateDefinition,
    DataFlowDiagram,
    
    # Components
    BaseComponentProperties,
    ButtonProperties,
    InputTextProperties,
    TextProperties,
    SwitchProperties,
    CheckboxProperties,
    SliderProperties,
    EnhancedComponentDefinition,
    COMPONENT_DEFINITIONS,
    get_available_components,
    get_component_definition,
    get_component_imports,
    get_output_component_type,
    export_component_catalog,
    get_component_type_union_literal,
    
    # Layout
    Position,
    LayoutConstraints,
    EnhancedLayoutDefinition,
    
    # Blockly
    BlocklyDefinition,
    BlocklyBlock,
    BlocklyWorkspace,
    EnhancedBlocklyDefinition,
    
    # Context
    PromptContext,
    IntentAnalysis,
    EnrichedContext,
    
    # Helper functions
    create_component,
    validate_layout,
    validate_component,
    get_component_schema,
)

from .prompts import (
    PromptTemplate,
    PromptLibrary,
    PromptType,
    PromptVersion,

)

__all__ = [
    # Schemas
    'PropertyValue',
    'ComponentStyle',
    'COMPONENT_PROPERTY_SCHEMAS',
    'AIRequest',
    'ProgressUpdate',
    'ErrorResponse',
    'CompleteResponse',
    'ArchitectureDesign',
    'ScreenDefinition',
    'NavigationStructure',
    'StateDefinition',
    'DataFlowDiagram',
    'BaseComponentProperties',
    'ButtonProperties',
    'InputTextProperties',
    'TextProperties',
    'SwitchProperties',
    'CheckboxProperties',
    'SliderProperties',
    'EnhancedComponentDefinition',
    'COMPONENT_DEFINITIONS',
    'get_available_components',
    'get_component_definition',
    'get_component_imports',
    'get_output_component_type',
    'export_component_catalog',
    'get_component_type_union_literal',
    'normalize_component_type',
    'get_component_default_dimensions',
    'get_component_default_properties',
    'get_component_event',
    'get_interactive_components',
    'get_template_components',
    'is_input_component',
    'has_component_event',
    'Position',
    'LayoutConstraints',
    'EnhancedLayoutDefinition',
    'BlocklyDefinition',
    'BlocklyBlock',
    'BlocklyWorkspace',
    'EnhancedBlocklyDefinition',
    'PromptContext',
    'IntentAnalysis',
    'EnrichedContext',
    'create_component',
    'validate_layout',
    'validate_component',
    'get_component_schema',
    
    # Prompts
    'PromptTemplate',
    'PromptLibrary',
    'PromptType',
    'PromptVersion',
    'prompts',
]