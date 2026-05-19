from typing import List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchText, MatchValue, MinShould

from config.schemas import FilterConfig, FilterCondition


def build_qdrant_filter(filter_config: Optional[FilterConfig]) -> Optional[Filter]:
    if not filter_config:
        return None

    must = _build_conditions(filter_config.must)
    should = _build_conditions(filter_config.should)
    must_not = _build_conditions(filter_config.must_not)

    min_should = None
    min_count = filter_config.minimum_should_match
    if min_count and should:
        min_should = MinShould(conditions=should, min_count=min_count)
        should = []

    if not must and not should and not must_not and not min_should:
        return None

    return Filter(
        must=must or None,
        should=should or None,
        must_not=must_not or None,
        min_should=min_should,
    )


def _build_conditions(conditions: List[FilterCondition]) -> List[FieldCondition]:
    built = []
    for condition in conditions:
        field = condition.field
        if not field:
            continue
        key = field if "." in field else f"metadata.{field}"
        match = _build_match(condition.op, condition.value)
        if match is None:
            continue
        built.append(FieldCondition(key=key, match=match))
    return built


def _build_match(op: str, value):
    if value is None:
        return None

    op = (op or "eq").lower()
    if op in ("eq", "="):
        if isinstance(value, (list, tuple, set)):
            return MatchAny(any=list(value))
        return MatchValue(value=value)

    if op == "in":
        if isinstance(value, (list, tuple, set)):
            return MatchAny(any=list(value))
        return MatchAny(any=[value])

    if op in ("contains", "match_text"):
        return MatchText(text=str(value))

    raise ValueError(f"Unsupported filter op: {op}")
