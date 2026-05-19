import json
import re

def repair_json(text: str) -> str:
    """
    Attempt to repair common LLM JSON errors:
    1. Missing commas between properties
    2. Trailing commas
    3. Single quotes used for keys/values
    4. Newlines inside strings
    """
    if not text:
        return text
        
    # 1. Remove trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # 2. Fix missing commas between properties/items
    # Look for "property": "value" "next_property": "value"
    # This regex looks for a closing quote/digit/boolean followed by a newline and then a start of a new key
    text = re.sub(r'("|\d|true|false|null)\s*\n\s*"', r'\1,\n"', text)
    
    # 3. Fix single quotes (only if they aren't part of the content)
    # This is risky but often needed for lazy LLMs
    # Fix key quotes: 'key': -> "key":
    text = re.sub(r"([{,]\s*)'([^']+)'\s*:", r'\1"\2":', text)
    # Fix value quotes: : 'value' -> : "value"
    text = re.sub(r":\s*'([^']*)'(\s*[,}])", r': "\1"\2', text)
    
    return text

def safe_json_loads(text: str) -> dict:
    """Tries to parse JSON, then tries to repair and parse again."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception:
            # If still failing, try the most aggressive approach: 
            # extract anything that looks like an object
            match = re.search(r'(\{[\s\S]*\})', text)
            if match:
                try:
                    return json.loads(repair_json(match.group(1)))
                except:
                    pass
            raise
