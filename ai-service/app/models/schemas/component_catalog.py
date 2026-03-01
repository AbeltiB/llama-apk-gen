"""Centralized UI component registry.

This module is the single source of truth for component definitions used across
prompting, schema validation, and output formatting.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TypedDict


class ComponentImport(TypedDict):
    """Import definition required to render a component."""

    name: str
    source: str


class ComponentDefinition(TypedDict, total=False):
    """Full definition for a UI component."""

    id: str
    name: str
    category: str
    output_type: str
    aliases: List[str]
    required_imports: List[ComponentImport]
    schema: Dict[str, Any]


COMPONENT_DEFINITIONS: Dict[str, ComponentDefinition] = {
    "Button": {
        "id": "core.button",
        "name": "Button",
        "category": "input",
        "output_type": "Button",
        "required_imports": [{"name": "Button", "source": "tamagui"}],
        "schema": {
            "value": {"type": "string", "required": False},
            "onPress": {"type": "event_handler", "required": False},
            "variant": {"type": "string", "required": False, "enum": ["primary", "secondary", "outline", "ghost"]},
            "style": {"type": "object", "required": False},
        },
    },
    "InputText": {
        "id": "core.input_text",
        "name": "InputText",
        "category": "input",
        "output_type": "Input",
        "aliases": ["Input", "TextInput"],
        "required_imports": [{"name": "Input", "source": "tamagui"}],
        "schema": {
            "placeholder": {"type": "string", "required": False},
            "value": {"type": "string", "required": False},
            "keyboardType": {"type": "string", "required": False},
            "onChange": {"type": "event_handler", "required": False},
            "secureTextEntry": {"type": "boolean", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Text": {
        "id": "core.text",
        "name": "Text",
        "category": "display",
        "output_type": "Text_Content",
        "required_imports": [
            {"name": "Text", "source": "tamagui"},
            {"name": "XStack", "source": "tamagui"},
        ],
        "schema": {
            "value": {"type": "string", "required": True},
            "fontSize": {"type": "number", "required": False},
            "fontWeight": {"type": "string", "required": False},
            "textAlign": {"type": "string", "required": False},
            "numberOfLines": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Switch": {
        "id": "core.switch",
        "name": "Switch",
        "category": "input",
        "output_type": "Switch",
        "required_imports": [
            {"name": "Switch", "source": "tamagui"},
            {"name": "Label", "source": "tamagui"},
            {"name": "XStack", "source": "tamagui"},
        ],
        "schema": {
            "value": {"type": "boolean", "required": True},
            "onToggle": {"type": "event_handler", "required": False},
            "thumbColor": {"type": "string", "required": False},
            "trackColor": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Checkbox": {
        "id": "core.checkbox",
        "name": "Checkbox",
        "category": "input",
        "output_type": "Checkbox",
        "required_imports": [
            {"name": "Checkbox", "source": "tamagui"},
            {"name": "Label", "source": "tamagui"},
        ],
        "schema": {
            "checked": {"type": "boolean", "required": True},
            "label": {"type": "string", "required": False},
            "onChange": {"type": "event_handler", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Slider": {
        "id": "core.slider",
        "name": "Slider",
        "category": "input",
        "output_type": "Slider",
        "required_imports": [{"name": "Slider", "source": "tamagui"}],
        "schema": {
            "min": {"type": "number", "required": True},
            "max": {"type": "number", "required": True},
            "value": {"type": "number", "required": True},
            "step": {"type": "number", "required": False},
            "onChange": {"type": "event_handler", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "ProgressBar": {
        "id": "feedback.progress_bar",
        "name": "ProgressBar",
        "category": "feedback",
        "output_type": "ProgressBar",
        "required_imports": [{"name": "Progress", "source": "tamagui"}],
        "schema": {"value": {"type": "number", "required": False}},
    },
    "Spinner": {
        "id": "feedback.spinner",
        "name": "Spinner",
        "category": "feedback",
        "output_type": "Spinner",
        "required_imports": [{"name": "Spinner", "source": "tamagui"}],
        "schema": {"size": {"type": "string", "required": False}},
    },
    "DatePicker": {
        "id": "form.date_picker",
        "name": "DatePicker",
        "category": "form",
        "output_type": "DatePicker",
        "required_imports": [{"name": "DatePicker", "source": "tamagui"}],
        "schema": {},
    },
    "TimePicker": {
        "id": "form.time_picker",
        "name": "TimePicker",
        "category": "form",
        "output_type": "TimePicker",
        "required_imports": [{"name": "TimePicker", "source": "tamagui"}],
        "schema": {},
    },
    "ColorPicker": {
        "id": "form.color_picker",
        "name": "ColorPicker",
        "category": "form",
        "output_type": "ColorPicker",
        "required_imports": [{"name": "ColorPicker", "source": "tamagui"}],
        "schema": {},
    },
    "Joystick": {
        "id": "iot.joystick",
        "name": "Joystick",
        "category": "interaction",
        "output_type": "Joystick",
        "required_imports": [{"name": "Joystick", "source": "custom"}],
        "schema": {},
    },
    "Map": {
        "id": "location.map",
        "name": "Map",
        "category": "location",
        "output_type": "Map",
        "required_imports": [{"name": "Map", "source": "react-native-maps"}],
        "schema": {},
    },
    "Chart": {
        "id": "data.chart",
        "name": "Chart",
        "category": "data_viz",
        "output_type": "Chart",
        "aliases": ["Graph"],
        "required_imports": [{"name": "Chart", "source": "victory-native"}],
        "schema": {
            "data": {"type": "array", "required": False},
            "type": {"type": "string", "required": False},
        },
    },
    "TextArea": {
        "id": "core.text_area",
        "name": "TextArea",
        "category": "input",
        "output_type": "TextArea",
        "required_imports": [{"name": "TextArea", "source": "tamagui"}],
        "schema": {
            "value": {"type": "string", "required": False},
            "placeholder": {"type": "string", "required": False},
        },
    },
}


COMPONENT_DEFAULT_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "Button": (180, 44),
    "InputText": (280, 44),
    "Text": (280, 28),
    "Switch": (72, 44),
    "Checkbox": (32, 32),
    "Slider": (280, 44),
    "ProgressBar": (280, 12),
    "Spinner": (32, 32),
    "DatePicker": (280, 44),
    "TimePicker": (280, 44),
    "ColorPicker": (280, 44),
    "Joystick": (140, 140),
    "Map": (320, 220),
    "Chart": (320, 220),
    "TextArea": (280, 100),
}

COMPONENT_DEFAULT_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "Button": {"value": "Button", "variant": "primary"},
    "InputText": {"value": "", "placeholder": "Enter text"},
    "Text": {"value": "Text"},
    "Switch": {"value": False},
    "Checkbox": {"checked": False, "label": ""},
    "Slider": {"min": 0, "max": 100, "value": 50},
    "ProgressBar": {"value": 0},
    "Spinner": {"size": "small"},
    "DatePicker": {},
    "TimePicker": {},
    "ColorPicker": {},
    "Joystick": {},
    "Map": {},
    "Chart": {"type": "line", "data": []},
    "TextArea": {"value": "", "placeholder": "Enter details"},
}

COMPONENT_EVENT_BY_TYPE: Dict[str, str] = {
    "Button": "onPress",
    "InputText": "onChange",
    "Switch": "onToggle",
    "Checkbox": "onChange",
    "Slider": "onChange",
}

APP_TEMPLATE_COMPONENTS: Dict[str, List[str]] = {
    "counter": ["Text", "Button", "Button"],
    "todo": ["InputText", "Button", "Text", "Checkbox"],
    "calculator": ["Text", "Button", "Button", "Button", "Button"],
    "timer": ["Text", "Button", "Button"],
    "notes": ["InputText", "TextArea", "Button", "Text"],
    "weather": ["Text", "Text", "Button"],
    "contacts": ["InputText", "Button", "Text"],
    "quiz": ["Text", "Button", "Button", "Button", "Button"],
    "search": ["InputText", "Button", "Text"],
    "form": ["InputText", "InputText", "TextArea", "Button"],
    "generic": ["Text", "Button", "InputText"],
}


def _build_alias_index() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for canonical, definition in COMPONENT_DEFINITIONS.items():
        aliases[canonical.lower()] = canonical
        for alias in definition.get("aliases", []):
            aliases[alias.lower()] = canonical
    return aliases


_COMPONENT_ALIAS_INDEX = _build_alias_index()


def get_component_definition(component_name: str) -> Optional[ComponentDefinition]:
    canonical = normalize_component_type(component_name, fallback="")
    if not canonical:
        return None
    return COMPONENT_DEFINITIONS.get(canonical)


def get_available_components() -> List[str]:
    return sorted(COMPONENT_DEFINITIONS.keys())


def get_output_component_type(component_name: str) -> str:
    definition = get_component_definition(component_name)
    if not definition:
        return component_name
    return definition.get("output_type", component_name)


def get_component_imports(component_name: str) -> List[ComponentImport]:
    definition = get_component_definition(component_name)
    if not definition:
        return []
    return deepcopy(definition.get("required_imports", []))


def get_component_type_union_literal() -> str:
    members = " | ".join(f'"{name}"' for name in get_available_components())
    return members or '"Text"'


def normalize_component_type(component_type: str, fallback: str = "Text") -> str:
    if not component_type:
        return fallback
    normalized = component_type.strip()
    if not normalized:
        return fallback
    canonical = _COMPONENT_ALIAS_INDEX.get(normalized.lower())
    return canonical if canonical else fallback


def get_component_default_dimensions(component_type: str) -> Tuple[int, int]:
    canonical = normalize_component_type(component_type)
    return COMPONENT_DEFAULT_DIMENSIONS.get(canonical, (280, 44))


def get_component_default_properties(component_type: str) -> Dict[str, Any]:
    canonical = normalize_component_type(component_type)
    return deepcopy(COMPONENT_DEFAULT_PROPERTIES.get(canonical, {}))


def get_component_event(component_type: str) -> str:
    canonical = normalize_component_type(component_type)
    return COMPONENT_EVENT_BY_TYPE.get(canonical, "")


def get_interactive_components() -> List[str]:
    return sorted(COMPONENT_EVENT_BY_TYPE.keys())


def get_template_components(template_name: str) -> List[str]:
    if not template_name:
        return deepcopy(APP_TEMPLATE_COMPONENTS["generic"])
    return deepcopy(APP_TEMPLATE_COMPONENTS.get(template_name.lower(), APP_TEMPLATE_COMPONENTS["generic"]))


def is_input_component(component_type: str) -> bool:
    definition = get_component_definition(component_type)
    return bool(definition and definition.get("category") == "input")


def has_component_event(component_type: str, event_name: str) -> bool:
    if not event_name:
        return False
    canonical = normalize_component_type(component_type, fallback="")
    if not canonical:
        return False
    definition = COMPONENT_DEFINITIONS.get(canonical, {})
    schema = definition.get("schema", {})
    prop_schema = schema.get(event_name)
    if isinstance(prop_schema, dict) and prop_schema.get("type") == "event_handler":
        return True
    return COMPONENT_EVENT_BY_TYPE.get(canonical) == event_name


def export_component_catalog() -> Dict[str, Any]:
    return {
        "components": deepcopy(COMPONENT_DEFINITIONS),
        "aliases": deepcopy(_COMPONENT_ALIAS_INDEX),
        "interactive_components": get_interactive_components(),
        "template_components": deepcopy(APP_TEMPLATE_COMPONENTS),
        "default_dimensions": deepcopy(COMPONENT_DEFAULT_DIMENSIONS),
        "default_properties": deepcopy(COMPONENT_DEFAULT_PROPERTIES),
        "events": deepcopy(COMPONENT_EVENT_BY_TYPE),
    }
