"""
Prompt templates for Llama-based APIs.

ALL prompts in this file enforce STRICT VALID JSON OUTPUT.
NO markdown. NO explanations. NO code blocks.
"""

from typing import Any, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class PromptTemplate:
    """
    Reusable prompt template with system and user components.
    """
    system: str
    user_template: str

    def format(self, **kwargs: Any) -> Tuple[str, str]:
        return self.system, self.user_template.format(**kwargs)


class PromptVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class PromptType(str, Enum):
    APP_GENERATION = "app_generation"
    ARCHITECTURE_EXTEND = "architecture_extend"
    LAYOUT_GENERATION = "layout_generation"
    BLOCKLY_GENERATION = "blockly_generation"
    CODE_GENERATION = "code_generation"
    INTENT_ANALYSIS = "intent_analysis"
    OPTIMIZATION = "optimization"


class PromptLibrary:
    """
    Collection of all prompt templates used by the AI service.
    ALL outputs MUST be strict JSON.
    """

    AVAILABLE_COMPONENTS = [
        "Button", "IconButton", "FloatingActionButton",
        "InputText", "PasswordInput", "EmailInput", "PhoneInput",
        "SearchBar", "TextArea", "Text", "RichText",
        "Switch", "Checkbox", "RadioButton", "RadioGroup",
        "Slider", "ProgressBar", "Spinner",
        "Image", "VideoPlayer", "AudioPlayer", "CameraPreview",
        "List", "ListItem", "Grid", "Card",
        "ScrollView", "TabView", "BottomNavigation",
        "Modal", "Dialog", "Snackbar",
        "DatePicker", "TimePicker", "ColorPicker",
        "Map", "Chart", "LineChart", "BarChart",
        "Joystick", "WebView"
    ]

    # ======================================================================
    # SHARED STRICT JSON RULES
    # ======================================================================

    STRICT_JSON_RULES = """
CRITICAL OUTPUT RULES (MANDATORY):
1. Output MUST be a SINGLE valid JSON value
2. NO markdown, NO comments, NO explanations
3. NO text before or after JSON
4. Use DOUBLE QUOTES for all strings
5. Output MUST parse using json.loads() in Python
6. If unsure, still return valid JSON
"""

    # ======================================================================
    # ARCHITECTURE DESIGN
    # ======================================================================

    ARCHITECTURE_DESIGN = PromptTemplate(
        system=f"""
You are an expert mobile app architect.

Your task is to generate a COMPLETE mobile app architecture.

{STRICT_JSON_RULES}

Allowed UI components:
{{components}}

Output schema (MUST MATCH EXACTLY):

{{
  "app_type": "single-page" | "multi-page" | "navigation-based",
  "screens": [
    {{
      "id": "string",
      "name": "string",
      "purpose": "string",
      "components": ["ComponentName"],
      "navigation": ["screen_id"]
    }}
  ],
  "navigation": {{
    "type": "stack" | "tab" | "drawer" | "none",
    "routes": [
      {{ "from": "screen_id", "to": "screen_id", "label": "string" }}
    ]
  }},
  "state_management": [
    {{
      "name": "string",
      "type": "local-state" | "global-state" | "async-state",
      "scope": "component" | "screen" | "global",
      "initial_value": {{}}
    }}
  ],
  "data_flow": {{
    "user_interactions": ["string"],
    "api_calls": ["string"],
    "local_storage": ["string"]
  }}
}}

Rules:
- ONLY use allowed components
- Screens must be minimal and focused
""",
        user_template="""
User request:
"{prompt}"

Context:
{context_section}
"""
    )

    # ======================================================================
    # LAYOUT GENERATION
    # ======================================================================

    LAYOUT_GENERATE = PromptTemplate(
        system=f"""
You are a mobile UI layout generator.

{STRICT_JSON_RULES}

Allowed components:
{{components}}

Output schema:

{{
  "screen_id": "string",
  "layout_type": "flex" | "grid" | "absolute",
  "background_color": "#FFFFFF",
  "components": [
    {{
      "component_id": "string",
      "component_type": "ComponentName",
      "properties": {{
        "key": {{
          "type": "literal" | "binding",
          "value": "any"
        }}
      }},
      "style": {{
        "x": number,
        "y": number,
        "width": number,
        "height": number
      }},
      "z_index": number
    }}
  ]
}}

Rules:
- Touch targets >= 44px height
- No unsupported components
""",
        user_template="""
App purpose:
{prompt}

Screen architecture:
{screen_architecture}

Required components:
{required_components}

Primary action:
{primary_action}
"""
    )

    # ======================================================================
    # BLOCKLY GENERATION (STRICT JSON AST)
    # ======================================================================

    BLOCKLY_GENERATE = PromptTemplate(
        system=f"""
You are a Blockly logic generator.

{STRICT_JSON_RULES}

Output schema:

{{
  "workspace": {{
    "blocks": [
      {{
        "id": "string",
        "type": "event" | "setter" | "getter" | "math" | "logic",
        "component": "string",
        "event": "string",
        "property": "string",
        "value": "any",
        "next": "block_id | null"
      }}
    ]
  }}
}}

Rules:
- No visual descriptions
- Blocks must connect via "next"
""",
        user_template="""
Architecture:
{architecture}

Layout:
{layout}

Component events:
{component_events}
"""
    )

    # ======================================================================
    # CODE GENERATION (JSON CODE MODEL, NOT SOURCE CODE)
    # ======================================================================

    CODE_GENERATE = PromptTemplate(
        system=f"""
You are a React Native code planner.

{STRICT_JSON_RULES}

You MUST NOT output JavaScript or JSX.
Return a JSON representation of code structure.

Output schema:

{{
  "screen_name": "string",
  "imports": ["string"],
  "state": [
    {{
      "name": "string",
      "initial_value": "any"
    }}
  ],
  "handlers": [
    {{
      "name": "string",
      "logic": ["string"]
    }}
  ],
  "render_tree": {{
    "component": "string",
    "children": []
  }}
}}

Rules:
- This JSON will later be compiled into real code
""",
        user_template="""
Architecture:
{architecture}

Layout:
{layout}

Logic:
{blockly_workspace}
"""
    )
