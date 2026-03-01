"""
Output JSON Formatter - Transforms pipeline output to ideeza-project schema

This module converts the AI pipeline's internal representation to the target
JSON structure expected by the ideeza project system.

Target Structure:
{
  "importManager": {...},
  "stateManager": {...},
  "functionManager": {...},
  "componentManager": {...},
  "uiManager": {...},
  "blocklyManager": {...},
  "code": "..."
}
"""
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from app.utils.logging import get_logger
from app.models.schemas.component_catalog import (
    get_component_imports,
    get_output_component_type,
)

logger = get_logger(__name__)

# Component mapping and imports are sourced from the central component catalog.

# Default style tokens for ideeza theme
STYLE_TOKENS = {
    "primary": "$primary",
    "secondary": "$secondary",
    "background": "$background",
    "backgroundStrong": "$backgroundStrong",
    "color": "$color",
    "thin": "$thin",
    "normal": "$normal",
    "thick": "$thick",
    "hairline": "$hairline",
    "0": "$0",
    "4": "$4",
    "2": "$2",
}


def format_pipeline_output(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform pipeline output to ideeza-project JSON structure.
    
    Args:
        raw_result: The pipeline's output containing architecture, layout, blockly, metadata
        
    Returns:
        Dict matching the ideeza-project schema
    """
    try:
        logger.info("output_formatter.starting", extra={"raw_result_keys": list(raw_result.keys())})
        
        # Extract main sections
        architecture = raw_result.get('architecture', {})
        layouts = raw_result.get('layout', {})
        blockly = raw_result.get('blockly', {})
        metadata = raw_result.get('metadata', {})
        
        # Build each manager section
        import_manager = _build_import_manager(architecture, layouts)
        state_manager = _build_state_manager(architecture, layouts)
        function_manager = _build_function_manager(blockly)
        component_manager = _build_component_manager(architecture, layouts)
        ui_manager = _build_ui_manager(architecture, layouts)
        blockly_manager = _build_blockly_manager(blockly, layouts)
        code = _generate_react_code(architecture, layouts, blockly, function_manager)
        
        # Assemble final output
        formatted = {
            "importManager": import_manager,
            "stateManager": state_manager,
            "functionManager": function_manager,
            "componentManager": component_manager,
            "uiManager": ui_manager,
            "blocklyManager": blockly_manager,
            "code": code
        }
        
        logger.info("output_formatter.completed", extra={"components_count": len(component_manager.get('components', {}))})
        return formatted
        
    except Exception as e:
        logger.error("output_formatter.failed", extra={"error": str(e)}, exc_info=e)
        # Return minimal valid structure on error
        return _create_fallback_output(str(e))


def _build_import_manager(architecture: Dict, layouts: Dict) -> Dict[str, Any]:
    """Build importManager section"""
    global_imports = [
        {"name": "React", "source": "react", "named": False},
        {"name": "useState", "source": "react", "named": True},
        {"name": "useEffect", "source": "react", "named": True},
        {"name": "useCallback", "source": "react", "named": True},
        {"name": "useRef", "source": "react", "named": True},
        {"name": "SafeAreaView", "source": "react-native-safe-area-context", "named": True},
        {"name": "ScrollView", "source": "react-native", "named": True},
        {"name": "View", "source": "react-native", "named": True},
    ]
    
    component_imports = {}
    
    # Collect imports from components
    for screen_id, layout in layouts.items():
        components = layout.get('components', [])
        for comp in components:
            comp_type = comp.get('component_type', 'Text')
            ideeza_type = get_output_component_type(comp_type)
            imports = get_component_imports(comp_type)
            if imports:
                component_imports[ideeza_type] = imports
    
    return {
        "globalImports": global_imports,
        "componentImports": component_imports
    }


def _build_state_manager(architecture: Dict, layouts: Dict) -> Dict[str, Any]:
    """Build stateManager.appState section"""
    app_state = {}
    
    for screen_id, layout in layouts.items():
        components = layout.get('components', [])
        for comp in components:
            comp_id = comp.get('component_id', f"comp_{len(app_state)}")
            comp_type = comp.get('component_type', 'Text')
            properties = comp.get('properties', {})
            
            # Extract style properties
            style_prop = properties.get('style', {})
            style_value = style_prop.get('value', {}) if isinstance(style_prop, dict) else {}
            
            # Build component state entry
            state_entry = {
                "text": _extract_prop_value(properties.get('value')),
                "color": _extract_prop_value(properties.get('color', '$color')),
                "style": {
                    "top": style_value.get('top', 0),
                    "left": style_value.get('left', 0),
                    "width": style_value.get('width', 280),
                    "height": style_value.get('height', 44)
                }
            }
            
            # Add component-specific properties
            if comp_type == 'Button':
                state_entry.update({
                    "size": _extract_prop_value(properties.get('size', '$4')),
                    "variant": _extract_prop_value(properties.get('variant', 'solid')),
                    "borderColor": _extract_prop_value(properties.get('borderColor', '$primary')),
                    "borderWidth": _extract_prop_value(properties.get('borderWidth', '$thin')),
                    "borderRadius": _extract_prop_value(properties.get('borderRadius', '$4')),
                    "backgroundColor": _extract_prop_value(properties.get('backgroundColor', '$primary'))
                })
            elif comp_type == 'Switch':
                state_entry.update({
                    "size": _extract_prop_value(properties.get('size', '$2')),
                    "label": _extract_prop_value(properties.get('label', 'Switch')),
                    "value": _extract_prop_value(properties.get('checked', False)),
                    "thumbColor": _extract_prop_value(properties.get('thumbColor', '$secondary')),
                    "checkedColor": _extract_prop_value(properties.get('checkedColor', '$primary')),
                    "defaultChecked": _extract_prop_value(properties.get('defaultChecked', False)),
                    "backgroundColor": _extract_prop_value(properties.get('backgroundColor', '$backgroundStrong'))
                })
            elif comp_type == 'Text':
                state_entry.update({
                    "fontSize": _extract_prop_value(properties.get('fontSize', '$4')),
                    "backgroundColor": _extract_prop_value(properties.get('backgroundColor', 'transparent'))
                })
            
            app_state[comp_id] = state_entry
    
    return {"appState": app_state}


def _build_function_manager(blockly: Dict) -> Dict[str, Any]:
    """Build functionManager section from Blockly blocks"""
    functions = {}
    
    blocks = blockly.get('workspace', {}).get('blocks', [])
    for block in blocks:
        if block.get('type') == 'event' and block.get('event'):
            comp_id = block.get('component', '')
            event_name = block.get('event', '')
            func_name = f"{comp_id}{event_name}"
            
            functions[func_name] = {
                "name": func_name,
                "parameters": [],
                "returnType": "void",
                "body": "",  # Will be populated from blockly code
                "triggers": [{
                    "component": comp_id,
                    "event": event_name
                }]
            }
    
    return {"functions": functions}


def _build_component_manager(architecture: Dict, layouts: Dict) -> Dict[str, Any]:
    """Build componentManager section"""
    components = {}
    roots = []
    
    for screen_id, layout in layouts.items():
        layout_components = layout.get('components', [])
        for comp in layout_components:
            comp_id = comp.get('component_id')
            if not comp_id:
                continue
                
            comp_type = comp.get('component_type', 'Text')
            ideeza_type = get_output_component_type(comp_type)
            properties = comp.get('properties', {})
            
            # Build props with typed values
            props = {}
            for key, prop_value in properties.items():
                if isinstance(prop_value, dict) and 'value' in prop_value:
                    props[key] = {
                        "type": "literal",
                        "value": prop_value['value']
                    }
                else:
                    props[key] = {
                        "type": "literal",
                        "value": prop_value
                    }
            
            # Build component entry
            component_entry = {
                "id": comp_id,
                "name": comp_id,
                "type": ideeza_type,
                "props": props,
                "events": {},  # Populated from blockly
                "children": [],
                "parentId": "root",
                "screenId": screen_id,
                "condition": "",
                "requiredImports": get_component_imports(comp_type)
            }
            
            components[comp_id] = component_entry
            roots.append(comp_id)
    
    return {
        "components": components,
        "roots": roots,
        "stateManager": _build_state_manager(architecture, layouts),
        "importManager": _build_import_manager(architecture, layouts),
        "functionManager": _build_function_manager({})
    }


def _build_ui_manager(architecture: Dict, layouts: Dict) -> Dict[str, Any]:
    """Build uiManager section"""
    screens = []
    
    # Extract screens from architecture
    if isinstance(architecture, dict):
        arch_screens = architecture.get('screens', [])
        for screen in arch_screens:
            screens.append({
                "id": screen.get('id', 'screen-1'),
                "name": screen.get('name', 'Home Page')
            })
    
    # Default shortcuts matching ideeza schema
    shortcuts = [
        {"key": "Delete"},
        {"key": "Backspace"},
        {"key": "z", "ctrlKey": True},
        {"key": "y", "ctrlKey": True},
        {"key": "["},
        {"key": "]"},
        {"key": "ArrowLeft", "shiftKey": True, "altKey": True},
        {"key": "ArrowRight", "shiftKey": True, "altKey": True},
        {"key": "r", "shiftKey": True, "altKey": True}
    ]
    
    # Build components list for uiManager (flat list of all components)
    ui_components = []
    for screen_id, layout in layouts.items():
        for comp in layout.get('components', []):
            comp_id = comp.get('component_id')
            if comp_id:
                comp_type = comp.get('component_type', 'Text')
                ideeza_type = get_output_component_type(comp_type)
                properties = comp.get('properties', {})
                
                # Build props
                props = {}
                for key, prop_value in properties.items():
                    if isinstance(prop_value, dict) and 'value' in prop_value:
                        props[key] = {"type": "literal", "value": prop_value['value']}
                    else:
                        props[key] = {"type": "literal", "value": prop_value}
                
                ui_components.append({
                    "id": comp_id,
                    "name": comp_id,
                    "type": ideeza_type,
                    "props": props,
                    "events": {},
                    "children": [],
                    "parentId": "root",
                    "screenId": screen_id,
                    "condition": "",
                    "requiredImports": get_component_imports(comp_type)
                })
    
    return {
        "selectedComponentId": None,
        "shortcuts": shortcuts,
        "components": ui_components,
        "screens": screens if screens else [{"id": "screen-1", "name": "Home Page"}],
        "activeScreenId": screens[0]["id"] if screens else "screen-1"
    }


def _build_blockly_manager(blockly: Dict, layouts: Dict) -> Dict[str, Any]:
    """Build blocklyManager section"""
    # Extract Blockly XML from workspace
    xml = _generate_blockly_xml(blockly)
    code = _generate_blockly_code(blockly)
    
    # Build componentProps from blockly blocks
    component_props = []
    blocks = blockly.get('workspace', {}).get('blocks', [])
    
    for block in blocks:
        block_id = block.get('id', '')
        comp_id = block.get('component', '')
        event = block.get('event', '')
        prop_name = block.get('property', '')
        
        if comp_id and event:
            func_name = f"{comp_id}{event}"
            component_props.append({
                "type": "expression",
                "value": func_name if event else "",
                "propName": event if event else prop_name,
                "blocklyId": block_id,
                "elementId": comp_id,
                "elementType": "WebSocket" if 'websocket' in code.lower() else None
            })
    
    # Build blocklyByScreen
    blockly_by_screen = {}
    for screen_id in layouts.keys():
        blockly_by_screen[screen_id] = {
            "xml": xml,
            "code": code,
            "json": {},
            "componentProps": component_props
        }
    
    return {
        "xml": xml,
        "code": code,
        "componentProps": component_props,
        "selectedTypeID": blocks[0].get('component') if blocks else None,
        "blocklyByScreen": blockly_by_screen
    }


def _generate_blockly_xml(blockly: Dict) -> str:
    """Generate Blockly XML string from workspace"""
    blocks = blockly.get('workspace', {}).get('blocks', [])
    
    xml_parts = ['<xml xmlns="https://developers.google.com/blockly/xml">']
    
    for block in blocks:
        block_type = block.get('type', '')
        block_id = block.get('id', '')
        component = block.get('component', '')
        event = block.get('event', '')
        
        if block_type == 'event' and component:
            xml_parts.append(f'  <block type="button_on_click" id="{block_id}" x="450" y="90">')
            xml_parts.append(f'    <mutation xmlns="http://www.w3.org/1999/xhtml" button_id="{component}"/>')
            xml_parts.append(f'    <field name="BUTTON_ID">{component}</field>')
            xml_parts.append('    <statement name="DO">')
            
            # Add nested blocks based on event
            if event == 'onPress':
                xml_parts.append(f'      <block type="websocket_send_text" id="gen_{block_id}">')
                xml_parts.append(f'        <value name="TEXT">')
                xml_parts.append(f'          <block type="text" id="txt_{block_id}">')
                xml_parts.append(f'            <field name="TEXT">{component}</field>')
                xml_parts.append('          </block>')
                xml_parts.append('        </value>')
                xml_parts.append('      </block>')
            
            xml_parts.append('    </statement>')
            xml_parts.append('  </block>')
    
    xml_parts.append('</xml>')
    return '\n'.join(xml_parts)


def _generate_blockly_code(blockly: Dict) -> str:
    """Generate JavaScript code from Blockly blocks"""
    blocks = blockly.get('workspace', {}).get('blocks', [])
    code_lines = []
    
    for block in blocks:
        block_type = block.get('type', '')
        component = block.get('component', '')
        event = block.get('event', '')
        
        if block_type == 'event' and component and event:
            func_name = f"{component}{event}"
            code_lines.append(f"\nconst {func_name} = () => {{")
            
            if event == 'onPress':
                code_lines.append(f"      sendWebSocketText('{component.lower()}');")
            elif event == 'onCheckedChange':
                code_lines.append(f"      updateAppState('{component}.checked', isChecked);")
                code_lines.append(f"      if (isChecked == true) {{")
                code_lines.append(f"        sendWebSocketText('on');")
                code_lines.append(f"      }} else if (isChecked == false) {{")
                code_lines.append(f"        sendWebSocketText('off');")
                code_lines.append(f"      }}")
            
            code_lines.append("\n};")
    
    return '\n'.join(code_lines) if code_lines else ""


def _generate_react_code(architecture: Dict, layouts: Dict, blockly: Dict, function_manager: Dict) -> str:
    """Generate React/Tamagui component code"""
    # Collect imports
    imports = set()
    imports.add("import React, { useState, useEffect, useCallback, useRef } from 'react';")
    imports.add("import { SafeAreaView } from 'react-native-safe-area-context';")
    imports.add("import { ScrollView, View } from 'react-native';")
    
    # Add Tamagui imports based on components
    for screen_id, layout in layouts.items():
        for comp in layout.get('components', []):
            comp_type = comp.get('component_type', 'Text')
            for imp in get_component_imports(comp_type):
                imports.add(f"import {{ {imp['name']} }} from '{imp['source']}';")
                
    # Build component JSX
    component_jsx = []
    for screen_id, layout in layouts.items():
        for comp in layout.get('components', []):
            comp_id = comp.get('component_id')
            comp_type = comp.get('component_type', 'Text')
            ideeza_type = get_output_component_type(comp_type)
            properties = comp.get('properties', {})
            
            style_prop = properties.get('style', {})
            style_value = style_prop.get('value', {}) if isinstance(style_prop, dict) else {}
            
            # Generate JSX based on component type
            if ideeza_type == 'Button':
                jsx = f'''        <Button 
            style={{ position: "absolute", top: {style_value.get('top', 0)}, left: {style_value.get('left', 0)}, width: {style_value.get('width', 280)}, height: {style_value.get('height', 44)} }}
            size={{"$4"}}
            color={{"white"}}
            backgroundColor={{"$primary"}}
            onPress={{{comp_id}onPress}}
        >{_extract_prop_value(properties.get('value', 'Button'))}</Button>'''
                component_jsx.append(jsx)
            elif ideeza_type == 'Text_Content':
                jsx = f'''        <XStack
            style={{ position: "absolute", top: {style_value.get('top', 0)}, left: {style_value.get('left', 0)}, width: {style_value.get('width', 280)}, height: {style_value.get('height', 44)} }}
        >
            <Text fontSize={{"$4"}} color={{"$primary"}}>
                {_extract_prop_value(properties.get('value', 'Text'))}
            </Text>
        </XStack>'''
                component_jsx.append(jsx)
            elif ideeza_type == 'Switch':
                jsx = f'''        <XStack alignItems="center" style={{ position: "absolute", top: {style_value.get('top', 0)}, left: {style_value.get('left', 0)} }}>
            <Switch
                id="{comp_id}"
                checked={{false}}
                onCheckedChange={{{comp_id}onCheckedChange}}
            >
                <Switch.Thumb backgroundColor={{"$secondary"}}/>
            </Switch>
            <Label paddingLeft={{"$2"}}>{_extract_prop_value(properties.get('label', 'Switch'))}</Label>
        </XStack>'''
                component_jsx.append(jsx)
    
    # Build function declarations
    functions_code = []
    functions = function_manager.get('functions', {})
    for func_name, func_def in functions.items():
        triggers = func_def.get('triggers', [])
        if triggers:
            comp_id = triggers[0].get('component', '')
            event = triggers[0].get('event', '')
            if event == 'onPress':
                functions_code.append(f"\nconst {func_name} = () => {{\n      sendWebSocketText('{comp_id.lower()}');\n\n}};")
    
    # Assemble full code
    code = '\n'.join(sorted(imports))
    code += f'''

function AppScreen1() {{
  const [appState, setAppState] = useState({{
'''
    
    # Add appState initial values
    for screen_id, layout in layouts.items():
        for comp in layout.get('components', []):
            comp_id = comp.get('component_id')
            properties = comp.get('properties', {})
            style_prop = properties.get('style', {})
            style_value = style_prop.get('value', {}) if isinstance(style_prop, dict) else {}
            
            code += f'''    "{comp_id}": {{
      "text": "{_extract_prop_value(properties.get('value', ''))}",
      "color": "{_extract_prop_value(properties.get('color', '$primary'))}",
      "style": {{
        "top": {style_value.get('top', 0)},
        "left": {style_value.get('left', 0)},
        "width": {style_value.get('width', 280)},
        "height": {style_value.get('height', 44)}
      }},
'''
            if comp.get('component_type') == 'Button':
                code += f'''      "size": "$4",
      "variant": "solid",
      "borderColor": "$primary",
      "borderWidth": "$thin",
      "borderRadius": "$4",
      "backgroundColor": "$primary",
'''
            code += '''    },
'''
    
    code += '''  });

'''
    # Add helper functions
    code += '''  const { screenWidth, screenHeight } = getScreenDimensions();
  const maxContentHeight = getDimension('82.46%', screenHeight);

  const getAppState = (path, defaultValue = null) => {
    if (!path.includes('.')) {
      return appState[path] || defaultValue;
    }
    return path.split('.').reduce((obj, key) => {
      return (obj && obj[key] !== undefined) ? obj[key] : defaultValue;
    }, appState);
  };

  const updateAppState = (path, value) => {
    setAppState(prevState => {
      const isFunctional = typeof value === 'function';
      const resolve = (oldVal) => isFunctional ? value(oldVal) : (value === undefined ? null : value);
      if (!path.includes('.')) {
        const resolvedValue = resolve(prevState[path]);
        return { ...prevState, [path]: resolvedValue };
      }
      const pathParts = path.split('.');
      const rootKey = pathParts[0];
      const leafKey = pathParts[pathParts.length - 1];
      const nextState = { ...prevState };
      const rootObject = { ...(prevState[rootKey] || {}) };
      let current = rootObject;
      for (let i = 1; i < pathParts.length - 1; i++) {
        const key = pathParts[i];
        current[key] = { ...(current[key] || {}) };
        current = current[key];
      }
      const resolvedValue = resolve(current[leafKey]);
      current[leafKey] = resolvedValue;
      nextState[rootKey] = rootObject;
      return nextState;
    });
  };

'''
    # Add generated functions
    code += '\n'.join(functions_code)
    
    # Add return JSX
    code += '''
  return (
    <SafeAreaView style={{ flex: 1, position: 'relative' }}>
      <ScrollView style={{ flex: 1 }} contentContainerStyle={{ minHeight: maxContentHeight, position: 'relative' }}>
'''
    code += '\n'.join(component_jsx)
    code += '''
      </ScrollView>
    </SafeAreaView>
  );
}

export default AppScreen1;
'''
    
    return code


def _extract_prop_value(prop: Any) -> Any:
    """Extract raw value from PropertyValue object or dict"""
    if prop is None:
        return ""
    if isinstance(prop, dict):
        return prop.get('value', prop)
    if hasattr(prop, 'value'):
        return prop.value
    return prop


def _create_fallback_output(error: str) -> Dict[str, Any]:
    """Create minimal valid output on formatter error"""
    return {
        "importManager": {"globalImports": [], "componentImports": {}},
        "stateManager": {"appState": {}},
        "functionManager": {"functions": {}},
        "componentManager": {
            "components": {},
            "roots": [],
            "stateManager": {"appState": {}},
            "importManager": {"globalImports": [], "componentImports": {}},
            "functionManager": {"functions": {}}
        },
        "uiManager": {
            "selectedComponentId": None,
            "shortcuts": [],
            "components": [],
            "screens": [{"id": "screen-1", "name": "Home Page"}],
            "activeScreenId": "screen-1"
        },
        "blocklyManager": {
            "xml": "<xml xmlns=\"https://developers.google.com/blockly/xml\"></xml>",
            "code": "",
            "componentProps": [],
            "selectedTypeID": None,
            "blocklyByScreen": {}
        },
        "code": "// Error generating code: " + error,
        "_error": error,
        "_fallback": True
    }


def validate_output_schema(output: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate that output matches expected ideeza schema.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_keys = [
        "importManager", "stateManager", "functionManager",
        "componentManager", "uiManager", "blocklyManager", "code"
    ]
    
    for key in required_keys:
        if key not in output:
            errors.append(f"Missing required key: {key}")
    
    # Validate importManager
    if "importManager" in output:
        im = output["importManager"]
        if "globalImports" not in im or not isinstance(im["globalImports"], list):
            errors.append("importManager.globalImports must be a list")
        if "componentImports" not in im or not isinstance(im["componentImports"], dict):
            errors.append("importManager.componentImports must be a dict")
    
    # Validate stateManager
    if "stateManager" in output:
        sm = output["stateManager"]
        if "appState" not in sm or not isinstance(sm["appState"], dict):
            errors.append("stateManager.appState must be a dict")
    
    # Validate componentManager
    if "componentManager" in output:
        cm = output["componentManager"]
        if "components" not in cm or not isinstance(cm["components"], dict):
            errors.append("componentManager.components must be a dict")
        if "roots" not in cm or not isinstance(cm["roots"], list):
            errors.append("componentManager.roots must be a list")
    
    # Validate uiManager
    if "uiManager" in output:
        um = output["uiManager"]
        if "screens" not in um or not isinstance(um["screens"], list):
            errors.append("uiManager.screens must be a list")
        if "activeScreenId" not in um:
            errors.append("uiManager.activeScreenId is required")
    
    # Validate blocklyManager
    if "blocklyManager" in output:
        bm = output["blocklyManager"]
        if "xml" not in bm or not isinstance(bm["xml"], str):
            errors.append("blocklyManager.xml must be a string")
        if "code" not in bm or not isinstance(bm["code"], str):
            errors.append("blocklyManager.code must be a string")
    
    # Validate code
    if "code" not in output or not isinstance(output["code"], str):
        errors.append("code must be a string")
    
    return len(errors) == 0, errors