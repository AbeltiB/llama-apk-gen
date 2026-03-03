"""
output_JSON_formatter.py
Transforms AI pipeline output to the exact ideeza-project schema required by System FE.

Supports two input shapes:
  Path A – "structured":  input has result.componentManager.components  (LLM success)
  Path B – "raw":         input has raw_result.layout / raw_result.architecture  (heuristic fallback)

Output schema (top-level keys, matching ideeza-project.json exactly):
  importManager, stateManager, functionManager, componentManager,
  uiManager, code, blocklyManager, blocklyByScreen   ← blocklyByScreen at ROOT, NOT inside blocklyManager
"""

import json
import re
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

COMPONENT_TYPE_MAP: Dict[str, str] = {
    "Button": "Button",
    "Text": "Text_Content",
    "TextView": "Text_Content",
    "Text_Content": "Text_Content",
    "Switch": "Switch",
    "List": "List",
    "ListView": "List",
    "Input": "Input",
    "TextInput": "Input",
    "TextField": "Input",
}

COMPONENT_IMPORTS: Dict[str, List[Dict[str, str]]] = {
    "Button": [{"name": "Button", "source": "tamagui"}],
    "Text_Content": [
        {"name": "Text", "source": "tamagui"},
        {"name": "XStack", "source": "tamagui"},
    ],
    "Switch": [
        {"name": "Switch", "source": "tamagui"},
        {"name": "Label", "source": "tamagui"},
        {"name": "XStack", "source": "tamagui"},
    ],
    "List": [{"name": "ScrollView", "source": "react-native"}],
    "Input": [{"name": "Input", "source": "tamagui"}],
}

UI_SHORTCUTS: List[Dict] = [
    {"key": "Delete"},
    {"key": "Backspace"},
    {"key": "z", "ctrlKey": True},
    {"key": "y", "ctrlKey": True},
    {"key": "["},
    {"key": "]"},
    {"key": "ArrowLeft", "shiftKey": True, "altKey": True},
    {"key": "ArrowRight", "shiftKey": True, "altKey": True},
    {"key": "r", "shiftKey": True, "altKey": True},
]

EMPTY_IMPORT_MANAGER: Dict[str, Any] = {"globalImports": [], "componentImports": {}}
EMPTY_FUNCTION_MANAGER: Dict[str, Any] = {"functions": {}}
EMPTY_XML = '<xml xmlns="https://developers.google.com/blockly/xml"></xml>'

REQUIRED_TOP_LEVEL_KEYS = frozenset({
    "importManager", "stateManager", "functionManager",
    "componentManager", "uiManager", "code", "blocklyManager", "blocklyByScreen",
})


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(value or "id"))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "id"


def _ideeza_type(raw_type: str) -> str:
    return COMPONENT_TYPE_MAP.get(raw_type, "Text_Content")


def _reverse_ideeza_type(t: str) -> str:
    return {"Text_Content": "Text", "Button": "Button", "Switch": "Switch",
            "List": "ListView", "Input": "TextInput"}.get(t, t)


def _get_imports(ideeza_type: str) -> List[Dict[str, str]]:
    return COMPONENT_IMPORTS.get(ideeza_type, COMPONENT_IMPORTS["Text_Content"])


def _unwrap(prop: Any) -> Any:
    """Unwrap a {type, value} envelope if present."""
    if isinstance(prop, dict) and "value" in prop:
        return prop["value"]
    return prop


def _wrap(value: Any) -> Dict[str, Any]:
    """Wrap a value in the ideeza {type: 'literal'} envelope."""
    return {"type": "literal", "value": value}


def _screen_display_name(screen_id: str) -> str:
    return screen_id.replace("_", " ").replace("-", " ").title()


def _extract_style(raw_props: Dict[str, Any]) -> Dict[str, int]:
    """Extract {top, left, width, height} from component properties."""
    raw = _unwrap(raw_props.get("style", {}))
    if not isinstance(raw, dict):
        raw = {}
    return {
        "top":    raw.get("top",    raw.get("y", 64)),
        "left":   raw.get("left",   raw.get("x", 137)),
        "width":  raw.get("width",  100),
        "height": raw.get("height", 50),
    }


def _extract_label(comp_type: str, raw_props: Dict[str, Any], comp_id: str) -> str:
    """
    Best-effort extraction of display label from various prop shapes:
      - props.properties.value.text.value   (structured LLM path)
      - props.value.value                   (raw heuristic path)
      - props.component_id.value            (semantic name)
      - fallback: humanise comp_id
    """
    # Structured LLM: props.properties.value.text.value
    nested_props = _unwrap(raw_props.get("properties", {}))
    if isinstance(nested_props, dict):
        text_field = nested_props.get("text")
        if text_field is not None:
            v = _unwrap(text_field)
            if v not in (None, ""):
                return str(v)

    # Semantic component_id
    for lable_key in ("label", "name", "title"):
        candidate = _unwrap(raw_props.get(lable_key, ""))
        if candidate is not None and candidate not in (None, ""):
            return str(candidate)
    sem_id = _unwrap(raw_props.get("component_id", ""))
    if sem_id and not _looks_like_autogenerated_id(str(sem_id)):
        return _humanize_identifier(str(sem_id))

    # Raw heuristic value field
    val = _unwrap(raw_props.get("value", ""))
    if val and val not in ("Button", "Sample Text", "Click Me"):
        return str(val)

    # Humanise comp_id as last resort
    human = _humanize_identifier(comp_id)
    if _looks_like_autogenerated_id(comp_id):
        fallback_by_type = {
            "Button": "Button",
            "Text_Content": "Text Content",
            "Switch": "Switch",
            "List": "List",
            "Input": "Input",
        }
        return fallback_by_type.get(_ideeza_type(comp_type), "Component")
    return human if human else ("Text Content" if _ideeza_type(comp_type) == "Text_Content" else "Button")
def _looks_like_autogenerated_id(value: str) -> bool:
    """Heuristic check for generated IDs like comp_counter_screen_button_2."""
    normalized = _sanitize_id(value).lower()
    generated_prefixes = (
        "comp_",
        "component_",
        "screen_",
    )
    if normalized.startswith(generated_prefixes):
        return True
    return bool(re.match(r".*_(?:button|text|input|switch|list)_\d+$", normalized))


def _humanize_identifier(value: str) -> str:
    """Convert machine identifiers into simple title-cased labels."""
    normalized = _sanitize_id(value)
    if not normalized:
        return ""

    stop_words = {
        "comp", "component", "screen", "main", "page", "view", "section", "item", "id",
        "button", "text", "input", "switch", "list",
    }
    tokens = [token for token in normalized.split("_") if token and not token.isdigit()]
    meaningful_tokens = [token for token in tokens if token.lower() not in stop_words]

    selected = meaningful_tokens or tokens
    return " ".join(selected).replace("-", " ").title()

# ──────────────────────────────────────────────────────────────────────────────
# Semantic prop builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_button_props(
    style: Dict, label: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (ideeza_props_dict, state_entry_dict) for a Button."""
    props = {
        "size":            _wrap("$4"),
        "text":            _wrap(label),
        "color":           _wrap("white"),
        "style":           _wrap(style),
        "variant":         _wrap("solid"),
        "borderColor":     _wrap("$primary"),
        "borderWidth":     _wrap("$thin"),
        "borderRadius":    _wrap("$4"),
        "backgroundColor": _wrap("$primary"),
    }
    state = {
        "size":            "$4",
        "text":            label,
        "color":           "white",
        "style":           style,
        "variant":         "solid",
        "borderColor":     "$primary",
        "borderWidth":     "$thin",
        "borderRadius":    "$4",
        "backgroundColor": "$primary",
    }
    return props, state


def _build_text_props(
    style: Dict, label: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (ideeza_props_dict, state_entry_dict) for a Text_Content."""
    props = {
        "text":            _wrap(label),
        "color":           _wrap("$color"),
        "style":           _wrap(style),
        "fontSize":        _wrap("$5"),
        "backgroundColor": _wrap("transparent"),
    }
    state = {
        "text":            label,
        "color":           "$color",
        "style":           style,
        "fontSize":        "$5",
        "backgroundColor": "transparent",
    }
    return props, state


def _build_semantic_props(
    raw_type: str,
    raw_props: Dict[str, Any],
    comp_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Dispatch to the correct prop builder based on component type."""
    ideeza_type = _ideeza_type(raw_type)
    style = _extract_style(raw_props)
    label = _extract_label(raw_type, raw_props, comp_id)

    if ideeza_type == "Button":
        return _build_button_props(style, label)
    else:
        return _build_text_props(style, label)


def _build_component_entry(
    comp_id: str,
    ideeza_type: str,
    props: Dict[str, Any],
    screen_id: str,
    parent_id: Optional[str],
    children: List[str],
) -> Dict[str, Any]:
    return {
        "id":              comp_id,
        "name":            comp_id,
        "type":            ideeza_type,
        "props":           props,
        "events":          {},
        "children":        children or [],
        "parentId":        parent_id or "root",
        "screenId":        screen_id,
        "condition":       "",
        "requiredImports": _get_imports(ideeza_type),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Blockly helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ws_blockly_entry(comp_id: str, y_pos: int) -> Tuple[str, str, str]:
    """Returns (xml_block_str, code_line, function_name) for a WebSocket button."""
    fn_name = f"{comp_id}onPress"
    block_id = f"block_{comp_id}"
    xml = (
        f'  <block type="button_on_click" id="{block_id}" x="450" y="{y_pos}">\n'
        f'    <mutation xmlns="http://www.w3.org/1999/xhtml" button_id="{comp_id}"/>\n'
        f'    <field name="BUTTON_ID">{comp_id}</field>\n'
        f'    <statement name="DO">\n'
        f'      <block type="websocket_send_text" id="gen_{comp_id}">\n'
        f'        <value name="TEXT">\n'
        f'          <block type="text">\n'
        f'            <field name="TEXT">{comp_id}</field>\n'
        f'          </block>\n'
        f'        </value>\n'
        f'      </block>\n'
        f'    </statement>\n'
        f'  </block>'
    )
    code = f"const {fn_name} = () => {{\n  sendWebSocketText('{comp_id}');\n}};"
    return xml, code, fn_name


# ──────────────────────────────────────────────────────────────────────────────
# React code generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_react_code(
    components: Dict[str, Any],
    functions: Dict[str, Any],
    state_app: Dict[str, Any],
    active_screen: str,
) -> str:
    lines = [
        "import React, { useState, useEffect, useCallback, useRef } from 'react';",
        "import { Button, Text, XStack } from 'tamagui';",
        "import { SafeAreaView } from 'react-native-safe-area-context';",
        "import { ScrollView, View } from 'react-native';",
        "import { getDimension, getScreenDimensions } from '../utils/dimensions';",
        "import { useNavigation } from '@react-navigation/native';",
        "import { useWebSocket } from '../hooks/useWebSocket';",
        "",
        "function AppScreen1() {",
        "  const navigation = useNavigation();",
        "  const {",
        "    sendWebSocketText,",
        "    connectWebSocket,",
        "    isWebSocketConnected,",
        "  } = useWebSocket();",
        "",
        f"  const [appState, setAppState] = useState({json.dumps(state_app, indent=4)});",
        "",
        "  const { screenWidth, screenHeight } = getScreenDimensions();",
        "  const maxContentHeight = getDimension('88.46%', screenHeight);",
        "",
        "  const getAppState = (path, defaultValue = null) => {",
        "    if (!path.includes('.')) return appState[path] ?? defaultValue;",
        "    return path.split('.').reduce(",
        "      (obj, key) => (obj && obj[key] !== undefined ? obj[key] : defaultValue),",
        "      appState",
        "    );",
        "  };",
        "",
        "  const updateAppState = (path, value) => {",
        "    setAppState(prev => {",
        "      const resolve = (old) => typeof value === 'function' ? value(old) : value;",
        "      if (!path.includes('.')) return { ...prev, [path]: resolve(prev[path]) };",
        "      const parts = path.split('.');",
        "      const next = { ...prev };",
        "      const root = { ...(prev[parts[0]] || {}) };",
        "      let cur = root;",
        "      for (let i = 1; i < parts.length - 1; i++) {",
        "        cur[parts[i]] = { ...(cur[parts[i]] || {}) };",
        "        cur = cur[parts[i]];",
        "      }",
        "      cur[parts[parts.length - 1]] = resolve(cur[parts[parts.length - 1]]);",
        "      next[parts[0]] = root;",
        "      return next;",
        "    });",
        "  };",
        "",
    ]

    for fn_name, fn_def in functions.items():
        lines.append(f"  const {fn_name} = () => {{")
        lines.append(f"    {fn_def.get('body', '')}")
        lines.append("  };")
        lines.append("")

    lines += [
        "  return (",
        "    <SafeAreaView style={{ flex: 1, position: 'relative' }}>",
        "      <ScrollView",
        "        style={{ flex: 1 }}",
        "        contentContainerStyle={{ minHeight: maxContentHeight, position: 'relative' }}",
        "      >",
    ]

    for comp_id, comp in components.items():
        if comp.get("screenId") != active_screen:
            continue
        comp_type = comp.get("type", "Text_Content")
        style_val = _unwrap(comp["props"].get("style", _wrap({})))
        if not isinstance(style_val, dict):
            style_val = {}
        t = style_val.get("top", 0)
        left = style_val.get("left", 0)
        w = style_val.get("width", 100)
        h = style_val.get("height", 50)

        if comp_type == "Button":
            fn = f"{comp_id}onPress"
            lines.append(
                f"        <Button\n"
                f"          style={{{{ position: 'absolute', top: getDimension({t}, screenHeight), left: getDimension({left}, screenWidth), width: getDimension({w}, screenWidth), height: getDimension({h}, screenHeight) }}}}\n"
                f"          size={{getAppState('{comp_id}.size', '$4')}}\n"
                f"          color={{getAppState('{comp_id}.variant', 'solid') === 'outline' ? getAppState('{comp_id}.borderColor', '$primary') : getAppState('{comp_id}.color', 'white')}}\n"
                f"          backgroundColor={{getAppState('{comp_id}.variant', 'solid') === 'solid' ? getAppState('{comp_id}.backgroundColor', '$primary') : 'transparent'}}\n"
                f"          borderColor={{getAppState('{comp_id}.borderColor', '$primary')}}\n"
                f"          borderRadius={{getAppState('{comp_id}.borderRadius', '$4')}}\n"
                f"          onPress={{{fn}}}\n"
                f"        >{{getAppState('{comp_id}.text', 'Button')}}</Button>"
            )
        else:
            lines.append(
                f"        <XStack\n"
                f"          style={{{{ position: 'absolute', top: getDimension({t}, screenHeight), left: getDimension({left}, screenWidth), width: getDimension({w}, screenWidth), height: getDimension({h}, screenHeight) }}}}\n"
                f"          backgroundColor={{getAppState('{comp_id}.backgroundColor', 'transparent')}}\n"
                f"        >\n"
                f"          <Text\n"
                f"            fontSize={{getAppState('{comp_id}.fontSize', '$5')}}\n"
                f"            color={{getAppState('{comp_id}.color', '$color')}}\n"
                f"            style={{{{ width: '100%', height: '100%' }}}}\n"
                f"          >{{getAppState('{comp_id}.text', 'Text Content')}}</Text>\n"
                f"        </XStack>"
            )

    lines += [
        "      </ScrollView>",
        "    </SafeAreaView>",
        "  );",
        "}",
        "",
        "export default AppScreen1;",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Path A – already-structured input (result.componentManager.components exists)
# ──────────────────────────────────────────────────────────────────────────────

def _from_structured(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a result that already has componentManager/uiManager/etc.
    Fixes:
      - stateManager keys  → use comp_id directly (not {screenId}_{compId})
      - component props    → semantic flat props matching ideeza schema
      - blocklyByScreen    → extract from blocklyManager and promote to root level
      - componentManager.importManager → empty (FE derives from code string)
    """
    raw_comps: Dict[str, Any] = result.get("componentManager", {}).get("components", {})
    bm_raw: Dict[str, Any] = dict(result.get("blocklyManager", {}))  # shallow copy to mutate

    components: Dict[str, Any] = {}
    state_app: Dict[str, Any] = {}
    roots: List[str] = []
    ui_components: List[Dict] = []
    screens_map: Dict[str, str] = {}
    functions: Dict[str, Any] = {}
    xml_blocks: List[str] = []
    code_lines: List[str] = []
    comp_props_list: List[Dict] = []
    y_pos = 90

    for comp_id, raw_comp in raw_comps.items():
        raw_type = _reverse_ideeza_type(raw_comp.get("type", "Text_Content"))
        ideeza_type = _ideeza_type(raw_type)
        raw_props = raw_comp.get("props", {})
        screen_id = raw_comp.get("screenId", "screen-1")
        screens_map[screen_id] = screen_id

        props, state = _build_semantic_props(raw_type, raw_props, comp_id)
        state_app[comp_id] = state

        comp_obj = _build_component_entry(
            comp_id=comp_id,
            ideeza_type=ideeza_type,
            props=props,
            screen_id=screen_id,
            parent_id=raw_comp.get("parentId"),
            children=raw_comp.get("children", []),
        )
        components[comp_id] = comp_obj
        roots.append(comp_id)
        ui_components.append(comp_obj)

        if ideeza_type == "Button":
            fn_name = f"{comp_id}onPress"
            functions[fn_name] = {
                "name":       fn_name,
                "parameters": [],
                "returnType": "void",
                "body":       f"sendWebSocketText('{comp_id}');",
                "triggers":   [{"component": comp_id, "event": "onPress"}],
            }
            xml, code_line, _ = _make_ws_blockly_entry(comp_id, y_pos)
            xml_blocks.append(xml)
            code_lines.append(code_line)
            comp_props_list.append({
                "type":        "expression",
                "value":       fn_name,
                "propName":    "onPress",
                "blocklyId":   f"block_{comp_id}",
                "elementId":   comp_id,
                "elementType": "WebSocket",
            })
            y_pos += 100

    screens = [
        {"id": sid, "name": _screen_display_name(sid)}
        for sid in (screens_map or {"screen-1": "screen-1"})
    ]
    active_screen = screens[0]["id"]

    # Pull blocklyByScreen out of blocklyManager (it must live at root)
    bly_by_screen: Dict[str, Any] = bm_raw.pop("blocklyByScreen", {})
    if not bly_by_screen:
        bly_by_screen = result.get("blocklyByScreen", {})

    # If still empty, rebuild from what we have
    if not bly_by_screen:
        xml_str = (
            '<xml xmlns="https://developers.google.com/blockly/xml">\n'
            + "\n".join(xml_blocks)
            + "\n</xml>"
        )
        code_str = "\n\n".join(code_lines)
        for sid in screens_map:
            bly_by_screen[sid] = {
                "xml":            xml_str,
                "code":           code_str,
                "json":           {},
                "componentProps": comp_props_list,
            }

    # Ensure every screen entry has the required 'json' key
    for sid, entry in bly_by_screen.items():
        entry.setdefault("json", {})

    # Use blockly data from raw result if present (richer), else rebuild
    bm_xml  = bm_raw.get("xml",  bly_by_screen.get(active_screen, {}).get("xml",  EMPTY_XML))
    bm_code = bm_raw.get("code", bly_by_screen.get(active_screen, {}).get("code", ""))
    bm_props = bm_raw.get("componentProps", comp_props_list)
    bm_selected = bm_raw.get("selectedTypeID", roots[0] if roots else None)

    blockly_manager = {
        "xml":            bm_xml,
        "code":           bm_code,
        "componentProps": bm_props,
        "selectedTypeID": bm_selected,
    }

    state_manager    = {"appState": state_app}
    function_manager = result.get("functionManager", {"functions": functions}) if functions else EMPTY_FUNCTION_MANAGER
    # Use pre-existing code if available (it's richer); otherwise generate
    code = result.get("code") or _generate_react_code(components, functions, state_app, active_screen)

    return {
        "importManager":   EMPTY_IMPORT_MANAGER,
        "stateManager":    state_manager,
        "functionManager": function_manager,
        "componentManager": {
            "components":      components,
            "roots":           roots,
            "stateManager":    state_manager,
            "importManager":   EMPTY_IMPORT_MANAGER.copy(),
            "functionManager": EMPTY_FUNCTION_MANAGER.copy(),
        },
        "uiManager": {
            "selectedComponentId": None,
            "shortcuts":           UI_SHORTCUTS,
            "components":          ui_components,
            "screens":             screens,
            "activeScreenId":      active_screen,
        },
        "code":            code,
        "blocklyManager":  blockly_manager,
        "blocklyByScreen": bly_by_screen,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Path B – raw heuristic input (raw_result.layout / raw_result.architecture)
# ──────────────────────────────────────────────────────────────────────────────

def _from_raw(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build full ideeza output from the heuristic/minimal raw_result format.
    """
    layouts: Dict[str, Any] = raw_result.get("layout", {})

    components: Dict[str, Any] = {}
    state_app: Dict[str, Any] = {}
    roots: List[str] = []
    ui_components: List[Dict] = []
    screens_map: Dict[str, str] = {}
    functions: Dict[str, Any] = {}
    xml_blocks: List[str] = []
    code_lines: List[str] = []
    comp_props_list: List[Dict] = []
    y_pos = 90

    for screen_id, layout_data in layouts.items():
        screens_map[screen_id] = screen_id
        raw_comps_list = (
            layout_data.get("components", [])
            if isinstance(layout_data, dict)
            else []
        )

        for raw_comp in raw_comps_list:
            comp_id  = raw_comp.get("component_id", f"comp_{len(components)}")
            comp_type = raw_comp.get("component_type", "Text")
            ideeza_type = _ideeza_type(comp_type)

            props, state = _build_semantic_props(comp_type, raw_comp.get("properties", {}), comp_id)
            state_app[comp_id] = state

            comp_obj = _build_component_entry(
                comp_id=comp_id,
                ideeza_type=ideeza_type,
                props=props,
                screen_id=screen_id,
                parent_id=raw_comp.get("parent_id"),
                children=raw_comp.get("children_ids", []),
            )
            components[comp_id] = comp_obj
            roots.append(comp_id)
            ui_components.append(comp_obj)

            if ideeza_type == "Button":
                fn_name = f"{comp_id}onPress"
                functions[fn_name] = {
                    "name":       fn_name,
                    "parameters": [],
                    "returnType": "void",
                    "body":       f"sendWebSocketText('{comp_id}');",
                    "triggers":   [{"component": comp_id, "event": "onPress"}],
                }
                xml, code_line, _ = _make_ws_blockly_entry(comp_id, y_pos)
                xml_blocks.append(xml)
                code_lines.append(code_line)
                comp_props_list.append({
                    "type":        "expression",
                    "value":       fn_name,
                    "propName":    "onPress",
                    "blocklyId":   f"block_{comp_id}",
                    "elementId":   comp_id,
                    "elementType": "WebSocket",
                })
                y_pos += 100

    if not screens_map:
        screens_map = {"screen-1": "screen-1"}

    screens = [{"id": sid, "name": _screen_display_name(sid)} for sid in screens_map]
    active_screen = screens[0]["id"]

    xml_str  = '<xml xmlns="https://developers.google.com/blockly/xml">\n' + "\n".join(xml_blocks) + "\n</xml>"
    code_str = "\n\n".join(code_lines)

    bly_by_screen: Dict[str, Any] = {
        sid: {"xml": xml_str, "code": code_str, "json": {}, "componentProps": comp_props_list}
        for sid in screens_map
    }
    blockly_manager = {
        "xml":            xml_str,
        "code":           code_str,
        "componentProps": comp_props_list,
        "selectedTypeID": roots[0] if roots else None,
    }

    state_manager    = {"appState": state_app}
    function_manager = {"functions": functions}
    code = _generate_react_code(components, functions, state_app, active_screen)

    return {
        "importManager":   EMPTY_IMPORT_MANAGER.copy(),
        "stateManager":    state_manager,
        "functionManager": function_manager,
        "componentManager": {
            "components":      components,
            "roots":           roots,
            "stateManager":    state_manager,
            "importManager":   EMPTY_IMPORT_MANAGER.copy(),
            "functionManager": EMPTY_FUNCTION_MANAGER.copy(),
        },
        "uiManager": {
            "selectedComponentId": None,
            "shortcuts":           UI_SHORTCUTS,
            "components":          ui_components,
            "screens":             screens,
            "activeScreenId":      active_screen,
        },
        "code":            code,
        "blocklyManager":  blockly_manager,
        "blocklyByScreen": bly_by_screen,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_output(output: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    for k in REQUIRED_TOP_LEVEL_KEYS:
        if k not in output:
            errors.append(f"Missing top-level key: '{k}'")

    if "blocklyByScreen" in output.get("blocklyManager", {}):
        errors.append("CRITICAL: blocklyByScreen must be at root level, not inside blocklyManager")

    bm = output.get("blocklyManager", {})
    for k in ("xml", "code", "componentProps", "selectedTypeID"):
        if k not in bm:
            errors.append(f"blocklyManager missing key: '{k}'")

    bbs = output.get("blocklyByScreen", {})
    if not isinstance(bbs, dict):
        errors.append("blocklyByScreen must be an object")
    else:
        for sid, entry in bbs.items():
            for k in ("xml", "code", "json", "componentProps"):
                if k not in entry:
                    errors.append(f"blocklyByScreen.{sid} missing key: '{k}'")

    cm = output.get("componentManager", {})
    for k in ("components", "roots", "stateManager", "importManager", "functionManager"):
        if k not in cm:
            errors.append(f"componentManager missing key: '{k}'")

    um = output.get("uiManager", {})
    for k in ("selectedComponentId", "shortcuts", "components", "screens", "activeScreenId"):
        if k not in um:
            errors.append(f"uiManager missing key: '{k}'")

    # stateManager keys must match componentManager.components keys
    sm_keys  = set(output.get("stateManager", {}).get("appState", {}).keys())
    comp_keys = set(output.get("componentManager", {}).get("components", {}).keys())
    orphan = sm_keys - comp_keys
    if orphan:
        errors.append(f"stateManager has keys not in componentManager.components: {orphan}")

    return len(errors) == 0, errors


# ──────────────────────────────────────────────────────────────────────────────
# Fallback
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_output(error: str) -> Dict[str, Any]:
    empty_state = {"appState": {}}
    screen = [{"id": "screen-1", "name": "Home Page"}]
    bly_entry = {"xml": EMPTY_XML, "code": "", "json": {}, "componentProps": []}
    return {
        "importManager":   EMPTY_IMPORT_MANAGER.copy(),
        "stateManager":    empty_state,
        "functionManager": EMPTY_FUNCTION_MANAGER.copy(),
        "componentManager": {
            "components":      {},
            "roots":           [],
            "stateManager":    empty_state,
            "importManager":   EMPTY_IMPORT_MANAGER.copy(),
            "functionManager": EMPTY_FUNCTION_MANAGER.copy(),
        },
        "uiManager": {
            "selectedComponentId": None,
            "shortcuts":           UI_SHORTCUTS,
            "components":          [],
            "screens":             screen,
            "activeScreenId":      "screen-1",
        },
        "code":            f"// Formatter error: {error}",
        "blocklyManager":  {"xml": EMPTY_XML, "code": "", "componentProps": [], "selectedTypeID": None},
        "blocklyByScreen": {"screen-1": bly_entry},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def format_pipeline_output(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform AI pipeline output into the ideeza-project JSON schema.

    Detection logic:
      1. If raw_input.result.componentManager.components is non-empty  → Path A (structured)
      2. If raw_input.raw_result has layout / architecture             → Path B (raw)
      3. If raw_input itself has layout / architecture                  → Path B (raw)
      4. Otherwise attempt Path A on raw_input.result
    """
    try:
        result = raw_input.get("result", raw_input)

        # Path A: structured output
        if (
            isinstance(result.get("componentManager"), dict)
            and result["componentManager"].get("components")
        ):
            logger.info("Input type: structured (Path A)")
            output = _from_structured(result)

        # Path B: raw_result wrapper
        elif "raw_result" in raw_input and isinstance(raw_input["raw_result"], dict):
            logger.info("Input type: raw_result wrapper (Path B)")
            output = _from_raw(raw_input["raw_result"])

        # Path B: flat raw
        elif "layout" in result or "architecture" in result:
            logger.info("Input type: flat raw (Path B)")
            output = _from_raw(result)

        # Fallback to Path A
        else:
            logger.warning("Input type unknown – attempting Path A")
            output = _from_structured(result)

        is_valid, errors = validate_output(output)
        if is_valid:
            logger.info("✓ Output validated successfully. Components: %d", len(output["componentManager"]["components"]))
        else:
            logger.warning("Validation warnings:\n  %s", "\n  ".join(errors))

        return output

    except Exception as exc:
        logger.exception("Formatter failed: %s", exc)
        return _fallback_output(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Export‑ready JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_export_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarantee that the dictionary contains all required top‑level keys of the
    ideeza‑project schema. If any key is missing, it is added with a sensible
    default value.
    """
    # Deep copy to avoid mutating the original
    result = data.copy()

    # 1. importManager
    if "importManager" not in result:
        result["importManager"] = EMPTY_IMPORT_MANAGER.copy()
    else:
        # Ensure subkeys exist
        im = result["importManager"]
        if not isinstance(im, dict):
            im = {}
        im.setdefault("globalImports", [])
        im.setdefault("componentImports", {})
        result["importManager"] = im

    # 2. stateManager (must have appState)
    if "stateManager" not in result:
        result["stateManager"] = {"appState": {}}
    else:
        sm = result["stateManager"]
        if not isinstance(sm, dict):
            sm = {}
        sm.setdefault("appState", {})
        result["stateManager"] = sm

    # 3. functionManager
    if "functionManager" not in result:
        result["functionManager"] = EMPTY_FUNCTION_MANAGER.copy()
    else:
        fm = result["functionManager"]
        if not isinstance(fm, dict):
            fm = {}
        fm.setdefault("functions", {})
        result["functionManager"] = fm

    # 4. componentManager (complex object)
    if "componentManager" not in result:
        result["componentManager"] = {
            "components": {},
            "roots": [],
            "stateManager": result.get("stateManager", {"appState": {}}),
            "importManager": result.get("importManager", EMPTY_IMPORT_MANAGER.copy()),
            "functionManager": result.get("functionManager", EMPTY_FUNCTION_MANAGER.copy()),
        }
    else:
        cm = result["componentManager"]
        if not isinstance(cm, dict):
            cm = {}
        cm.setdefault("components", {})
        cm.setdefault("roots", [])
        cm.setdefault("stateManager", result.get("stateManager", {"appState": {}}))
        cm.setdefault("importManager", result.get("importManager", EMPTY_IMPORT_MANAGER.copy()))
        cm.setdefault("functionManager", result.get("functionManager", EMPTY_FUNCTION_MANAGER.copy()))
        result["componentManager"] = cm

    # 5. uiManager
    if "uiManager" not in result:
        result["uiManager"] = {
            "selectedComponentId": None,
            "shortcuts": UI_SHORTCUTS,
            "components": [],
            "screens": [{"id": "screen-1", "name": "Home Page"}],
            "activeScreenId": "screen-1",
        }
    else:
        um = result["uiManager"]
        if not isinstance(um, dict):
            um = {}
        um.setdefault("selectedComponentId", None)
        um.setdefault("shortcuts", UI_SHORTCUTS)
        um.setdefault("components", [])
        um.setdefault("screens", [{"id": "screen-1", "name": "Home Page"}])
        um.setdefault("activeScreenId", "screen-1")
        result["uiManager"] = um

    # 6. code (string)
    if "code" not in result:
        result["code"] = "// No code generated"

    # 7. blocklyManager
    if "blocklyManager" not in result:
        result["blocklyManager"] = {
            "xml": EMPTY_XML,
            "code": "",
            "componentProps": [],
            "selectedTypeID": None,
        }
    else:
        bm = result["blocklyManager"]
        if not isinstance(bm, dict):
            bm = {}
        bm.setdefault("xml", EMPTY_XML)
        bm.setdefault("code", "")
        bm.setdefault("componentProps", [])
        bm.setdefault("selectedTypeID", None)
        result["blocklyManager"] = bm

    # 8. blocklyByScreen
    if "blocklyByScreen" not in result:
        # Use the active screen from uiManager if possible
        active = result.get("uiManager", {}).get("activeScreenId", "screen-1")
        result["blocklyByScreen"] = {
            active: {
                "xml": result["blocklyManager"]["xml"],
                "code": result["blocklyManager"]["code"],
                "json": {},
                "componentProps": result["blocklyManager"]["componentProps"],
            }
        }
    else:
        bbs = result["blocklyByScreen"]
        if not isinstance(bbs, dict):
            bbs = {}
        for sid, entry in bbs.items():
            if not isinstance(entry, dict):
                entry = {}
            entry.setdefault("xml", result["blocklyManager"]["xml"])
            entry.setdefault("code", result["blocklyManager"]["code"])
            entry.setdefault("json", {})
            entry.setdefault("componentProps", result["blocklyManager"]["componentProps"])
            bbs[sid] = entry
        result["blocklyByScreen"] = bbs

    return result


def get_export_ready_json(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert any pipeline output (raw or structured) into a clean, export‑ready
    ideeza‑project JSON object. This function strips away all task‑metadata
    and ensures that every required field exists.
    """
    # Step 1: Run the core formatter to get the base ideeza structure
    base = format_pipeline_output(raw_input)

    # Step 2: Apply the safety defaults to fill any accidentally missing fields
    export_ready = ensure_export_schema(base)

    # Step 3: (Optional) final validation – if errors remain, log them
    is_valid, errors = validate_output(export_ready)
    if not is_valid:
        logger.warning("Export-ready output has validation warnings:\n  %s", "\n  ".join(errors))

    return export_ready


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Simple CLI: if argument is provided, read that file; otherwise read stdin.
    # If the flag --export is present, call get_export_ready_json instead of format_pipeline_output.
    export_mode = "--export" in sys.argv
    if export_mode:
        sys.argv.remove("--export")

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fh:
            data = json.load(fh)
    else:
        data = json.load(sys.stdin)

    if export_mode:
        result = get_export_ready_json(data)
        logger.info("Export‑ready JSON generated.")
    else:
        result = format_pipeline_output(data)

    # Always print the final JSON to stdout (so it can be piped or saved)
    print(json.dumps(result, indent=2, ensure_ascii=False))