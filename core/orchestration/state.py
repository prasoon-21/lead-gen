from typing import Any, Dict


def get_path(data: Dict[str, Any], path: str, default=None):
    if not path:
        return default
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        return
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def map_inputs(state: Dict[str, Any], input_map: Dict[str, str]) -> Dict[str, Any]:
    mapped = {}
    for arg_name, state_path in (input_map or {}).items():
        mapped[arg_name] = get_path(state, state_path)
    return mapped


def map_outputs(state: Dict[str, Any], output: Dict[str, Any], output_map: Dict[str, str]) -> None:
    for state_path, output_path in (output_map or {}).items():
        set_path(state, state_path, get_path(output, output_path))
