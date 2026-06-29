from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote


try:
    import phonenumbers
    from phonenumbers import NumberParseException, PhoneNumberFormat, PhoneNumberType
except ImportError:  # pragma: no cover - dependency is optional at import time
    phonenumbers = None
    NumberParseException = Exception
    PhoneNumberFormat = None
    PhoneNumberType = None


PHONE_TEXT_PATTERN = re.compile(
    r"(?:\+?1[\s().-]*)?(?:\(?\d{3}\)?[\s().-]*)\d{3}[\s().-]*\d{4}(?:\s*(?:x|ext\.?|extension)\s*\d{1,6})?",
    re.IGNORECASE,
)
TEL_LINK_PATTERN = re.compile(r"href=[\"']tel:([^\"']+)[\"']", re.IGNORECASE)
EMPTY_PHONE_VALUES = {"", "-", "--", "n/a", "na", "none", "null", "not found"}
INVALID_NANP_AREA_CODES = {
    "000",
    "111",
    "123",
    "157",
    "222",
    "333",
    "444",
    "555",
    "577",
    "588",
    "679",
}

SOURCE_WEIGHTS = {
    "contact_page": 86,
    "support_page": 82,
    "sales_page": 82,
    "team_page": 76,
    "about_page": 72,
    "homepage": 66,
    "linkedin_research": 64,
    "tavily_extract": 55,
    "tavily_contact_search": 50,
    "text": 45,
    "existing": 40,
}

POSITIVE_CONTEXT_WORDS = (
    "call",
    "phone",
    "tel",
    "contact",
    "office",
    "sales",
    "support",
    "text",
    "mobile",
)
BAD_CONTEXT_WORDS = ("fax", "ein", "tax id", "license", "tracking", "order")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _strip_leading_zip_code(raw: str) -> str:
    return re.sub(
        r"^\s*\d{5}(?:-\d{4})?\s+(?=(?:\+?1[\s().-]*)?\(?\d{3}\)?)",
        "",
        raw or "",
    ).strip()


def _clean_raw_phone(value: Any) -> str:
    text = _strip_leading_zip_code(unquote(str(value or "")).strip())
    text = re.sub(r"^(?:tel:|phone:)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"[^\d+xextEXT().\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_bad_before_parse(raw: str, context: str = "") -> bool:
    cleaned = _clean_raw_phone(raw)
    base_text = re.split(r"\b(?:x|ext\.?|extension)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    digits = _digits(base_text)
    if cleaned.lower() in EMPTY_PHONE_VALUES:
        return True
    if len(digits) < 10:
        return True
    if len(digits) > 11 and not re.match(r"^\+\s*(?!1\b)\d", cleaned):
        return True
    if re.search(r"\d{15,}", str(raw or "")):
        return True
    if len(set(digits[-10:])) <= 2:
        return True
    if digits[-10:] in {"1234567890", "0123456789", "9876543210"}:
        return True
    national_digits = digits[1:] if len(digits) == 11 and digits.startswith("1") else digits[-10:]
    if len(national_digits) == 10:
        area_code = national_digits[:3]
        exchange = national_digits[3:6]
        if area_code in INVALID_NANP_AREA_CODES or area_code[0] in {"0", "1"}:
            return True
        if exchange[0] in {"0", "1"}:
            return True
    if re.fullmatch(r"\d{4}[-/.]\d{2}[-/.]\d{2}", cleaned):
        return True
    lowered_context = (context or "").lower()
    has_positive_context = any(word in lowered_context for word in POSITIVE_CONTEXT_WORDS)
    if re.search(r"(?:https?://|src=|href=|url\(|data-id=|fbid|page_id|profile_id|facebook\.com|instagram\.com|cdn)", lowered_context):
        return True
    if re.search(r"\.(?:png|jpe?g|gif|svg|webp|css|js|ico|avif)\b|[\w-]+_\d{6,}(?:_\d{3,})+", lowered_context):
        return True
    if "fax" in lowered_context and not has_positive_context:
        return True
    return any(word in lowered_context for word in BAD_CONTEXT_WORDS) and not has_positive_context


def _phone_type_label(number: Any) -> str:
    if not phonenumbers or PhoneNumberType is None:
        return "unknown"
    phone_type = phonenumbers.number_type(number)
    labels = {
        PhoneNumberType.FIXED_LINE: "fixed_line",
        PhoneNumberType.MOBILE: "mobile",
        PhoneNumberType.FIXED_LINE_OR_MOBILE: "fixed_line_or_mobile",
        PhoneNumberType.TOLL_FREE: "toll_free",
        PhoneNumberType.VOIP: "voip",
        PhoneNumberType.PREMIUM_RATE: "premium_rate",
        PhoneNumberType.SHARED_COST: "shared_cost",
    }
    return labels.get(phone_type, "unknown")


def _fallback_phone_candidate(
    raw_text: str,
    *,
    source_type: str,
    source_url: str,
    context: str,
    source_kind: str,
    region: str,
) -> Dict[str, Any] | None:
    digits = _digits(raw_text)
    if not digits:
        return None

    national_digits = digits
    country_code = ""
    if len(digits) == 11 and digits.startswith("1"):
        country_code = "1"
        national_digits = digits[1:]
    elif len(digits) == 10 and (region or "").upper() in {"US", "CA"}:
        country_code = "1"
    elif 10 <= len(digits) <= 15:
        country_code = digits[: len(digits) - 10]
        national_digits = digits[-10:]
    else:
        return None

    if len(national_digits) != 10:
        return None
    if national_digits[0] in {"0", "1"} or national_digits[3] in {"0", "1"}:
        return None
    if national_digits[:3] in INVALID_NANP_AREA_CODES:
        return None

    display = f"({national_digits[:3]}) {national_digits[3:6]}-{national_digits[6:]}"
    e164 = f"+{country_code or '1'}{national_digits}"
    source_score = SOURCE_WEIGHTS.get(source_type, SOURCE_WEIGHTS["text"])
    if source_kind == "tel_link":
        source_score += 12
    lowered_context = (context or "").lower()
    if any(word in lowered_context for word in POSITIVE_CONTEXT_WORDS):
        source_score += 6
    confidence = max(0, min(source_score + 2, 88))
    return {
        "raw": raw_text,
        "display": display,
        "e164": e164,
        "digits": _digits(e164),
        "source": source_type,
        "source_kind": source_kind,
        "source_url": source_url,
        "validation_status": "possible",
        "phone_type": "unknown",
        "confidence": confidence,
        "context": re.sub(r"\s+", " ", context or "").strip()[:180],
    }


def _context_window(text: str, start: int, end: int, size: int = 60) -> str:
    return (text or "")[max(0, start - size) : min(len(text or ""), end + size)]


def _match_is_inside_noise(text: str, match: re.Match[str]) -> bool:
    body = text or ""
    start, end = match.start(), match.end()
    prefix = body[max(0, start - 200) : start]
    suffix = body[end : min(len(body), end + 120)]
    context = f"{prefix}{match.group(0)}{suffix}"
    if re.search(r"(?:https?://|src=|href=|url\(|data-id=|fbid|page_id|profile_id)[^\s<>\"']*$", prefix, re.IGNORECASE):
        return True
    if re.search(r"\.(?:png|jpe?g|gif|svg|webp|css|js|ico|avif)[^\s<>\"']*$", prefix, re.IGNORECASE):
        return True
    if re.search(r"[\w-]+_\d{6,}(?:_\d{3,})*", prefix):
        return True
    if re.search(r"(?:facebook|instagram|cdn|static|assets?|images?)[^\s<>\"']*\d{8,}", context, re.IGNORECASE):
        return True
    stripped_candidate_digits = _digits(_strip_leading_zip_code(match.group(0)))
    if len(stripped_candidate_digits) in {10, 11}:
        return False
    digit_context = re.sub(r"\D", "", context)
    if len(digit_context) >= 15 and not any(word in context.lower() for word in POSITIVE_CONTEXT_WORDS):
        return True
    return False


def normalize_phone_candidate(
    raw: Any,
    *,
    source_type: str = "text",
    source_url: str = "",
    context: str = "",
    source_kind: str = "text",
    region: str = "US",
) -> Dict[str, Any] | None:
    raw_text = _clean_raw_phone(raw)
    if _looks_bad_before_parse(raw_text, context):
        return None

    if not phonenumbers:
        return _fallback_phone_candidate(
            raw_text,
            source_type=source_type,
            source_url=source_url,
            context=context,
            source_kind=source_kind,
            region=region,
        )

    try:
        parsed = phonenumbers.parse(raw_text, region)
    except NumberParseException:
        return None

    possible = phonenumbers.is_possible_number(parsed)
    valid = phonenumbers.is_valid_number(parsed)
    if not possible:
        return None

    e164 = phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
    national = phonenumbers.format_number(parsed, PhoneNumberFormat.NATIONAL)
    source_score = SOURCE_WEIGHTS.get(source_type, SOURCE_WEIGHTS["text"])
    if source_kind == "tel_link":
        source_score += 12
    lowered_context = (context or "").lower()
    if any(word in lowered_context for word in POSITIVE_CONTEXT_WORDS):
        source_score += 6
    validation_status = "valid" if valid else "possible"
    confidence = source_score + (12 if valid else 2)
    if "fax" in lowered_context:
        confidence -= 20
    confidence = max(0, min(confidence, 99))

    return {
        "raw": raw_text,
        "display": national,
        "e164": e164,
        "digits": _digits(e164),
        "source": source_type,
        "source_kind": source_kind,
        "source_url": source_url,
        "validation_status": validation_status,
        "phone_type": _phone_type_label(parsed),
        "confidence": confidence,
        "context": re.sub(r"\s+", " ", context or "").strip()[:180],
    }


def extract_phone_candidates(
    text: str,
    *,
    source_type: str = "text",
    source_url: str = "",
    region: str = "US",
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_raw = set()
    body = text or ""

    for match in TEL_LINK_PATTERN.finditer(body):
        raw = match.group(1)
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        candidate = normalize_phone_candidate(
            raw,
            source_type=source_type,
            source_url=source_url,
            context="tel link",
            source_kind="tel_link",
            region=region,
        )
        if candidate:
            candidates.append(candidate)

    for match in PHONE_TEXT_PATTERN.finditer(body):
        if _match_is_inside_noise(body, match):
            continue
        raw = match.group(0)
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        candidate = normalize_phone_candidate(
            raw,
            source_type=source_type,
            source_url=source_url,
            context=_context_window(body, match.start(), match.end()),
            source_kind="text",
            region=region,
        )
        if candidate:
            candidates.append(candidate)

    return rank_phone_candidates(candidates)


def phone_candidates_from_values(
    values: Iterable[Any],
    *,
    source_type: str = "existing",
    source_url: str = "",
    region: str = "US",
) -> List[Dict[str, Any]]:
    candidates = [
        normalize_phone_candidate(value, source_type=source_type, source_url=source_url, region=region)
        for value in values
    ]
    return rank_phone_candidates([candidate for candidate in candidates if candidate])


def rank_phone_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        key = str(candidate.get("e164") or candidate.get("digits") or candidate.get("display") or "").strip()
        if not key:
            continue
        existing = best_by_key.get(key)
        if not existing or int(candidate.get("confidence") or 0) > int(existing.get("confidence") or 0):
            best_by_key[key] = candidate
    return sorted(
        best_by_key.values(),
        key=lambda item: (
            int(item.get("confidence") or 0),
            1 if item.get("validation_status") == "valid" else 0,
            1 if item.get("source_kind") == "tel_link" else 0,
        ),
        reverse=True,
    )


def build_phone_summary(candidates: Iterable[Dict[str, Any]], *, max_alternates: int = 4) -> Dict[str, Any]:
    ranked = rank_phone_candidates(candidates)
    if not ranked:
        return {
            "contact_phone": "",
            "alternate_phones": [],
            "phone_confidence": 0,
            "phone_source": "",
            "phone_validation_status": "no_phone_found",
            "phone_type": "",
            "phone_candidates": [],
        }

    primary = ranked[0]
    alternates = [item["display"] for item in ranked[1 : max_alternates + 1] if item.get("display")]
    return {
        "contact_phone": primary.get("display", ""),
        "alternate_phones": alternates,
        "phone_confidence": int(primary.get("confidence") or 0),
        "phone_source": primary.get("source", ""),
        "phone_validation_status": primary.get("validation_status", ""),
        "phone_type": primary.get("phone_type", ""),
        "phone_candidates": ranked[: max_alternates + 1],
    }


def phone_summary_from_text(
    text: str,
    *,
    source_type: str = "text",
    source_url: str = "",
    region: str = "US",
) -> Dict[str, Any]:
    return build_phone_summary(
        extract_phone_candidates(text, source_type=source_type, source_url=source_url, region=region)
    )
