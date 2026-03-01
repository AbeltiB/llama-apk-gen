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


def export_component_catalog() -> Dict[str, Any]:
    """Return a serializable snapshot of the full component catalog."""
    return {
        "total": len(COMPONENT_DEFINITIONS),
        "components": COMPONENT_DEFINITIONS,
    }


def get_component_type_union_literal() -> str:
    """Return component names formatted for prompt schema literals."""
    names = get_available_components()
    return " | ".join(f"\"{name}\"" for name in names)


# Centralized aliases and runtime metadata
COMPONENT_ALIASES: Dict[str, str] = {
    'text': 'Text',
    'label': 'Text',
    'textview': 'Text',
    'textlabel': 'Text',
    'textfield': 'Text',
    'button': 'Button',
    'btn': 'Button',
    'submitbutton': 'Button',
    'actionbutton': 'Button',
    'input': 'InputText',
    'textinput': 'InputText',
    'inputtext': 'InputText',
    'edittext': 'InputText',
    'textentry': 'InputText',
    'textarea': 'TextArea',
    'textareafield': 'TextArea',
    'multilinetext': 'TextArea',
    'checkbox': 'Checkbox',
    'check': 'Checkbox',
    'checkbutton': 'Checkbox',
    'switch': 'Switch',
    'toggle': 'Switch',
    'toggleswitch': 'Switch',
    'slider': 'Slider',
    'seekbar': 'Slider',
    'range': 'Slider',
    'progressbar': 'ProgressBar',
    'progress': 'ProgressBar',
    'loadingbar': 'ProgressBar',
    'spinner': 'Spinner',
    'loading': 'Spinner',
    'loader': 'Spinner',
    'datepicker': 'DatePicker',
    'date': 'DatePicker',
    'datefield': 'DatePicker',
    'timepicker': 'TimePicker',
    'time': 'TimePicker',
    'timefield': 'TimePicker',
    'colorpicker': 'ColorPicker',
    'color': 'ColorPicker',
    'colorfield': 'ColorPicker',
    'joystick': 'Joystick',
    'gamepad': 'Joystick',
    'map': 'Map',
    'mapview': 'Map',
    'googlemap': 'Map',
    'chart': 'Chart',
    'graph': 'Chart',
    'plot': 'Chart',
}

COMPONENT_DEFAULT_DIMENSIONS: Dict[str, tuple[int, int]] = {
    'Button': (120, 44),
    'InputText': (280, 44),
    'Switch': (51, 31),
    'Checkbox': (24, 24),
    'TextArea': (280, 100),
    'Slider': (280, 44),
    'Spinner': (40, 40),
    'Text': (280, 40),
    'Joystick': (100, 100),
    'ProgressBar': (280, 8),
    'DatePicker': (280, 44),
    'TimePicker': (280, 44),
    'ColorPicker': (280, 44),
    'Map': (340, 200),
    'Chart': (340, 200),
}

COMPONENT_DEFAULT_PROPERTIES: Dict[str, Dict[str, Any]] = {
    'Checkbox': {'checked': False},
    'Switch': {'checked': False},
    'Slider': {'value': 50, 'min': 0, 'max': 100},
    'ProgressBar': {'value': 0.5, 'max': 1.0},
    'Button': {'value': 'Button'},
    'Text': {'value': 'Text'},
    'InputText': {'placeholder': 'Enter text', 'value': ''},
    'TextArea': {'placeholder': 'Enter text here...', 'value': ''},
    'Spinner': {'value': 0},
    'DatePicker': {'value': ''},
    'TimePicker': {'value': ''},
    'ColorPicker': {'value': '#000000'},
    'Map': {'latitude': 0.0, 'longitude': 0.0},
    'Chart': {'data': []},
}

COMPONENT_EVENT_MAP: Dict[str, str] = {
    'Button': 'onPress',
    'Switch': 'onToggle',
    'Checkbox': 'onToggle',
    'InputText': 'onChange',
    'TextArea': 'onChange',
}

TEMPLATE_COMPONENT_SETS: Dict[str, List[str]] = {
    'counter': ['Text', 'Button', 'Button'],
    'todo': ['InputText', 'Button', 'Text', 'Checkbox', 'Button'],
    'calculator': ['Text', 'Button'],
    'timer': ['Text', 'Button', 'Button', 'Button'],
    'quiz': ['Button', 'Text'],
    'notes': ['TextArea', 'Button'],
    'search': ['Text', 'InputText', 'Button'],
    'settings': ['Text', 'Button'],
    'form': ['InputText', 'InputText', 'InputText', 'Button'],
    'generic': ['Text', 'Button', 'InputText'],
}


def normalize_component_type(component_type: str, fallback: str = 'Text') -> str:
    if not component_type:
        return fallback

    normalized = component_type.strip()
    available = get_available_components()
    if normalized in available:
        return normalized

    lower = normalized.lower()
    aliased = COMPONENT_ALIASES.get(lower)
    if aliased and aliased in COMPONENT_DEFINITIONS:
        return aliased

    for comp in available:
        if comp.lower() == lower:
            return comp

    return fallback


def get_component_default_dimensions(component_name: str) -> tuple[int, int]:
    return COMPONENT_DEFAULT_DIMENSIONS.get(component_name, (280, 44))


def get_component_default_properties(component_name: str) -> Dict[str, Any]:
    return COMPONENT_DEFAULT_PROPERTIES.get(component_name, {})


def get_component_event(component_name: str) -> Optional[str]:
    return COMPONENT_EVENT_MAP.get(component_name)


def get_interactive_components() -> List[str]:
    return sorted(COMPONENT_EVENT_MAP.keys())


def get_template_components(template_name: str) -> List[str]:
    components = TEMPLATE_COMPONENT_SETS.get(template_name, TEMPLATE_COMPONENT_SETS['generic'])
    return [c for c in components if c in COMPONENT_DEFINITIONS]


def is_input_component(component_name: str) -> bool:
    definition = get_component_definition(component_name)
    return bool(definition and definition.get("category") == "input")


def has_component_event(component_name: str, event_name: str) -> bool:
    return get_component_event(component_name) == event_name
