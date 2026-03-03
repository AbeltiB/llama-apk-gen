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
    "Group": {
        "id": "layout.group",
        "name": "Group",
        "category": "layout",
        "output_type": "Group",
        "required_imports": [{"name": "Stack", "source": "react-native"}],
        "schema": {
            "backgroundColor": {"type": "string", "required": False},
            "borderColor": {"type": "string", "required": False},
            "borderWidth": {"type": "number", "required": False},
            "borderStyle": {"type": "string", "required": False},
            "borderRadius": {"type": "number", "required": False},
            "padding": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Layout": {
        "id": "layout.stack",
        "name": "Layout",
        "category": "layout",
        "output_type": "Layout",
        "required_imports": [
            {"name": "YStack", "source": "tamagui"},
            {"name": "XStack", "source": "tamagui"},
        ],
        "schema": {
            "direction": {"type": "string", "required": False, "enum": ["vertical", "horizontal"]},
            "spacing": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Button": {
        "id": "core.button",
        "name": "Button",
        "category": "input",
        "output_type": "Button",
        "required_imports": [{"name": "Button", "source": "tamagui"}],
        "schema": {
            "text": {"type": "string", "required": False},
            "value": {"type": "string", "required": False},
            "onPress": {"type": "event_handler", "required": False},
            "variant": {"type": "string", "required": False, "enum": ["primary", "secondary", "outline", "ghost"]},
            "size": {"type": "string", "required": False},
            "color": {"type": "string", "required": False},
            "backgroundColor": {"type": "string", "required": False},
            "borderWidth": {"type": "number", "required": False},
            "borderColor": {"type": "string", "required": False},
            "borderRadius": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "InputText": {
        "id": "core.input_text",
        "name": "InputText",
        "category": "input",
        "output_type": "Input_Text",
        "aliases": ["Input", "TextInput", "Input_Text"],
        "required_imports": [{"name": "Input", "source": "tamagui"}],
        "schema": {
            "placeholder": {"type": "string", "required": False},
            "value": {"type": "string", "required": False},
            "keyboardType": {"type": "string", "required": False},
            "onChange": {"type": "event_handler", "required": False},
            "secureTextEntry": {"type": "boolean", "required": False},
            "placeholderColor": {"type": "string", "required": False},
            "focusStyle": {"type": "object", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Text": {
        "id": "core.text",
        "name": "Text",
        "category": "display",
        "output_type": "Text_Content",
        "aliases": ["Text_Content"],
        "required_imports": [
            {"name": "Text", "source": "tamagui"},
            {"name": "XStack", "source": "tamagui"},
        ],
        "schema": {
            "text": {"type": "string", "required": False},
            "value": {"type": "string", "required": False},
            "fontSize": {"type": "number", "required": False},
            "fontWeight": {"type": "string", "required": False},
            "textAlign": {"type": "string", "required": False},
            "color": {"type": "string", "required": False},
            "backgroundColor": {"type": "string", "required": False},
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
            "defaultChecked": {"type": "boolean", "required": False},
            "value": {"type": "boolean", "required": False},
            "label": {"type": "string", "required": False},
            "onToggle": {"type": "event_handler", "required": False},
            "thumbColor": {"type": "string", "required": False},
            "trackColor": {"type": "string", "required": False},
            "checkedColor": {"type": "string", "required": False},
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
            {"name": "XStack", "source": "tamagui"},
            {"name": "Check", "source": "@tamagui/lucide-icons"},
        ],
        "schema": {
            "checked": {"type": "boolean", "required": True},
            "label": {"type": "string", "required": False},
            "onChange": {"type": "event_handler", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "TextArea": {
        "id": "core.text_area",
        "name": "TextArea",
        "category": "input",
        "output_type": "Text_Area",
        "aliases": ["Text_Area"],
        "required_imports": [{"name": "TextArea", "source": "tamagui"}],
        "schema": {
            "value": {"type": "string", "required": False},
            "placeholder": {"type": "string", "required": False},
            "multiline": {"type": "boolean", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Spinner": {
        "id": "feedback.spinner",
        "name": "Spinner",
        "category": "feedback",
        "output_type": "Spinner",
        "required_imports": [{"name": "Spinner", "source": "tamagui"}],
        "schema": {
            "size": {"type": "string", "required": False},
            "visible": {"type": "boolean", "required": False},
            "color": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Chart": {
        "id": "data.chart",
        "name": "Chart",
        "category": "data_viz",
        "output_type": "Chart",
        "aliases": ["Graph"],
        "required_imports": [{"name": "ChartWidget", "source": "../utils/ChartWidget"}],
        "schema": {
            "type": {"type": "string", "required": False},
            "values": {"type": "array", "required": False},
            "labels": {"type": "array", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Image": {
        "id": "media.image",
        "name": "Image",
        "category": "media",
        "output_type": "Image",
        "required_imports": [{"name": "Image", "source": "react-native"}],
        "schema": {
            "source": {"type": "string", "required": False},
            "alt": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Video": {
        "id": "media.video",
        "name": "Video",
        "category": "media",
        "output_type": "Video",
        "required_imports": [
            {"name": "VideoView", "source": "expo-video"},
            {"name": "useVideoPlayer", "source": "expo-video"},
        ],
        "schema": {
            "source": {"type": "string", "required": False},
            "shouldPlay": {"type": "boolean", "required": False},
            "isLooping": {"type": "boolean", "required": False},
            "isMuted": {"type": "boolean", "required": False},
            "volume": {"type": "number", "required": False},
            "resizeMode": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Joystick": {
        "id": "iot.joystick",
        "name": "Joystick",
        "category": "interaction",
        "output_type": "Joystick",
        "required_imports": [{"name": "Joystick", "source": "react-native-joystick-lite"}],
        "schema": {
            "size": {"type": "number", "required": False},
            "color": {"type": "string", "required": False},
            "haptics": {"type": "boolean", "required": False},
            "interval": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "Slider": {
        "id": "core.slider",
        "name": "Slider",
        "category": "input",
        "output_type": "Slider",
        "required_imports": [
            {"name": "Slider", "source": "tamagui"},
            {"name": "XStack", "source": "tamagui"},
        ],
        "schema": {
            "min": {"type": "number", "required": True},
            "max": {"type": "number", "required": True},
            "value": {"type": "number", "required": True},
            "step": {"type": "number", "required": False},
            "onChange": {"type": "event_handler", "required": False},
            "trackColor": {"type": "string", "required": False},
            "thumbColor": {"type": "string", "required": False},
            "activeTrackColor": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "ProgressBar": {
        "id": "feedback.progress_bar",
        "name": "ProgressBar",
        "category": "feedback",
        "output_type": "Progress_Bar",
        "aliases": ["Progress_Bar"],
        "required_imports": [
            {"name": "Progress", "source": "tamagui"},
            {"name": "XStack", "source": "tamagui"},
            {"name": "Text", "source": "tamagui"},
        ],
        "schema": {
            "value": {"type": "number", "required": False},
            "min": {"type": "number", "required": False},
            "max": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "ToastAlert": {
        "id": "feedback.toast_alert",
        "name": "ToastAlert",
        "category": "feedback",
        "output_type": "ToastAlert",
        "required_imports": [
            {"name": "Toast", "source": "@tamagui/toast"},
            {"name": "useToastState", "source": "@tamagui/toast"},
            {"name": "useToastController", "source": "@tamagui/toast"},
            {"name": "YStack", "source": "tamagui"},
        ],
        "schema": {
            "title": {"type": "string", "required": False},
            "description": {"type": "string", "required": False},
            "duration": {"type": "number", "required": False},
            "animation": {"type": "string", "required": False},
            "opacity": {"type": "number", "required": False},
            "scale": {"type": "number", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "FilePickerInput": {
        "id": "input.file_picker",
        "name": "FilePickerInput",
        "category": "input",
        "output_type": "FilePickerInput",
        "required_imports": [
            {"name": "Image", "source": "react-native"},
            {"name": "TouchableOpacity", "source": "react-native"},
            {"name": "Linking", "source": "react-native"},
            {"name": "Modal", "source": "react-native"},
            {"name": "Text", "source": "tamagui"},
            {"name": "DocumentPicker", "source": "expo-document-picker"},
            {"name": "WebBrowser", "source": "expo-web-browser"},
        ],
        "schema": {
            "placeholder": {"type": "string", "required": False},
            "pickImageFile": {"type": "event_handler", "required": False},
            "image": {"type": "string", "required": False},
            "file": {"type": "string", "required": False},
            "isPreviewing": {"type": "boolean", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "DatePicker": {
        "id": "form.date_picker",
        "name": "DatePicker",
        "category": "form",
        "output_type": "datePicker",
        "aliases": ["datePicker"],
        "required_imports": [
            {"name": "Text", "source": "tamagui"},
            {"name": "Button", "source": "tamagui"},
        ],
        "schema": {
            "UTC_date": {"type": "string", "required": False},
            "SelectedDate": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "TimePicker": {
        "id": "form.time_picker",
        "name": "TimePicker",
        "category": "form",
        "output_type": "timePicker",
        "aliases": ["timePicker"],
        "required_imports": [
            {"name": "Text", "source": "tamagui"},
            {"name": "Button", "source": "tamagui"},
        ],
        "schema": {
            "UTC_time": {"type": "string", "required": False},
            "SelectedTime": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "ColorPicker": {
        "id": "form.color_picker",
        "name": "ColorPicker",
        "category": "form",
        "output_type": "ColorPicker",
        "required_imports": [
            {"name": "Modal", "source": "react-native"},
            {"name": "TouchableWithoutFeedback", "source": "react-native"},
            {"name": "Pressable", "source": "react-native"},
            {"name": "Text", "source": "tamagui"},
        ],
        "schema": {"style": {"type": "object", "required": False}},
    },
    "Map": {
        "id": "location.map",
        "name": "Map",
        "category": "location",
        "output_type": "GoogleMap",
        "aliases": ["GoogleMap"],
        "required_imports": [
            {"name": "View", "source": "react-native"},
            {"name": "TouchableOpacity", "source": "react-native"},
            {"name": "Text", "source": "react-native"},
            {"name": "SafeAreaView", "source": "react-native-safe-area-context"},
            {"name": "useMapController", "source": "../hooks/useMapController"},
            {"name": "MapUnit", "source": "../utils/MapUnit"},
        ],
        "schema": {
            "latitude": {"type": "number", "required": False},
            "longitude": {"type": "number", "required": False},
            "wsUrl": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "LocalPushNotification": {
        "id": "device.local_push_notification",
        "name": "LocalPushNotification",
        "category": "device",
        "output_type": "LocalPushNotification",
        "required_imports": [
            {"name": "setupAndroidChannel", "source": "../utils/LocalPushNotification"},
            {"name": "triggerNotification", "source": "../utils/LocalPushNotification"},
            {"name": "ensureNotificationPermission", "source": "../utils/LocalPushNotification"},
        ],
        "schema": {
            "title": {"type": "string", "required": False},
            "body": {"type": "string", "required": False},
            "subtitle": {"type": "string", "required": False},
            "sound": {"type": "string", "required": False},
            "badge": {"type": "number", "required": False},
            "timeToLive": {"type": "number", "required": False},
            "priority": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
    "List": {
        "id": "core.list",
        "name": "List",
        "category": "display",
        "output_type": "List",
        "aliases": ["TodoList", "FlatList"],
        "required_imports": [
            {"name": "YStack", "source": "tamagui"},
            {"name": "Text", "source": "tamagui"},
        ],
        "schema": {
            "items": {"type": "array", "required": False},
            "value": {"type": "string", "required": False},
            "style": {"type": "object", "required": False},
        },
    },
}


COMPONENT_DEFAULT_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "Group": (320, 240),
    "Layout": (320, 240),
    "Button": (180, 44),
    "InputText": (280, 44),
    "Text": (280, 28),
    "Switch": (72, 44),
    "Checkbox": (40, 32),
    "TextArea": (280, 100),
    "Spinner": (32, 32),
    "Chart": (320, 220),
    "Image": (160, 120),
    "Video": (280, 180),
    "Joystick": (140, 140),
    "Slider": (280, 44),
    "ProgressBar": (280, 16),
    "ToastAlert": (280, 96),
    "FilePickerInput": (280, 44),
    "DatePicker": (280, 44),
    "TimePicker": (280, 44),
    "ColorPicker": (280, 120),
    "Map": (320, 220),
    "LocalPushNotification": (280, 44),
    "List": (280, 180),
}

COMPONENT_DEFAULT_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "Group": {"backgroundColor": "#FFFFFF"},
    "Layout": {"direction": "vertical", "spacing": 8},
    "Button": {"text": "Button", "variant": "primary"},
    "InputText": {"value": "", "placeholder": "Enter text"},
    "Text": {"text": "Text"},
    "Switch": {"value": False, "label": ""},
    "Checkbox": {"checked": False, "label": ""},
    "TextArea": {"value": "", "placeholder": "Enter details", "multiline": True},
    "Spinner": {"size": "small", "visible": True},
    "Chart": {"type": "line", "values": [], "labels": []},
    "Image": {"source": "", "alt": ""},
    "Video": {"source": "", "shouldPlay": False, "isLooping": False, "isMuted": False, "volume": 1},
    "Joystick": {"size": 120, "interval": 100},
    "Slider": {"min": 0, "max": 100, "value": 50, "step": 1},
    "ProgressBar": {"value": 0, "min": 0, "max": 100},
    "ToastAlert": {"title": "", "description": "", "duration": 3000, "animation": "bouncy"},
    "FilePickerInput": {"placeholder": "Select file", "isPreviewing": False},
    "DatePicker": {"UTC_date": "", "SelectedDate": ""},
    "TimePicker": {"UTC_time": "", "SelectedTime": ""},
    "ColorPicker": {},
    "Map": {"latitude": 0, "longitude": 0, "wsUrl": ""},
    "LocalPushNotification": {"title": "", "body": "", "priority": "default"},
    "List": {"items": ["Item 1", "Item 2"]},
}

COMPONENT_EVENT_BY_TYPE: Dict[str, str] = {
    "Button": "onPress",
    "InputText": "onChange",
    "Switch": "onToggle",
    "Checkbox": "onChange",
    "Slider": "onChange",
    "FilePickerInput": "pickImageFile",
}

APP_TEMPLATE_COMPONENTS: Dict[str, List[str]] = {
    "counter": ["Text", "Button", "Button", "Layout"],
    "todo": ["InputText", "Button", "List", "Checkbox", "Group"],
    "calculator": ["InputText", "Text", "Layout", "Button", "Button", "Button", "Button"],
    "timer": ["Text", "Button", "Button", "ProgressBar"],
    "notes": ["InputText", "TextArea", "Button", "Text", "FilePickerInput"],
    "weather": ["Text", "Text", "Button", "Image", "Map"],
    "contacts": ["InputText", "Button", "Text", "Layout"],
    "quiz": ["Text", "Button", "Button", "Button", "Button", "ProgressBar"],
    "search": ["InputText", "Button", "Text", "List"],
    "form": ["InputText", "InputText", "TextArea", "DatePicker", "TimePicker", "Button"],
    "generic": ["Layout", "Text", "Button", "InputText"],
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
