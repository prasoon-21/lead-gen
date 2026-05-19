import json
from typing import Any, Dict

from config.schemas import ClassificationDimensionSpec, ClassificationProfileSpec


def build_classification_schema(profile: ClassificationProfileSpec) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required = []

    for dimension in profile.dimensions:
        properties[dimension.name] = _dimension_schema(dimension)
        if dimension.required:
            required.append(dimension.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_classification_system_prompt(profile: ClassificationProfileSpec) -> str:
    lines = [
        "You classify customer conversations into structured workflow fields.",
        "Return only valid JSON.",
        "Use only the allowed values for enum fields.",
        "If a nullable field is unknown, return null.",
    ]

    if profile.description:
        lines.append(f"Profile: {profile.description}")

    for index, instruction in enumerate(profile.instructions, start=1):
        lines.append(f"Instruction {index}: {instruction}")

    lines.append("Dimensions:")
    for dimension in profile.dimensions:
        lines.append(_dimension_instruction(dimension))

    return "\n".join(lines)


def build_classification_prompt(
    profile: ClassificationProfileSpec,
    conversation_text: str,
    additional_context: Dict[str, Any] | None = None,
) -> str:
    schema_text = json.dumps(build_classification_schema(profile), ensure_ascii=False, indent=2)
    context = additional_context or {}

    return (
        "Classify the following conversation according to the configured dimensions.\n"
        "Return JSON matching this schema exactly.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        f"Additional context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _dimension_instruction(dimension: ClassificationDimensionSpec) -> str:
    parts = [
        f"- {dimension.name} ({dimension.field_type})",
        dimension.description,
    ]
    if dimension.allowed_values:
        parts.append(f"Allowed values: {', '.join(dimension.allowed_values)}.")
    if dimension.allow_null:
        parts.append("Return null if the value cannot be inferred confidently.")
    if not dimension.required:
        parts.append("This field is optional.")
    return " ".join(parts)


def _dimension_schema(dimension: ClassificationDimensionSpec) -> Dict[str, Any]:
    schema: Dict[str, Any]
    field_type = dimension.field_type

    if field_type == "boolean":
        schema = {"type": "boolean"}
    elif field_type == "number":
        schema = {"type": "number"}
    elif field_type == "array_enum":
        schema = {
            "type": "array",
            "items": {"type": "string", "enum": dimension.allowed_values},
        }
    elif field_type == "enum":
        schema = {"type": "string", "enum": dimension.allowed_values}
    else:
        schema = {"type": "string"}

    schema["description"] = dimension.description

    if dimension.allow_null:
        schema = {"anyOf": [schema, {"type": "null"}]}

    return schema
