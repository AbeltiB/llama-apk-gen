"""Centralized UI component registry.

This module is the single source of truth for component definitions used across
prompting, schema validation, and output formatting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


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


def get_component_definition(component_name: str) -> Optional[ComponentDefinition]:
    return COMPONENT_DEFINITIONS.get(component_name)


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
    return definition.get("required_imports", [])
