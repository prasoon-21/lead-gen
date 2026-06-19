import json
import logging
import re
import smtplib
import socket
import time


logger = logging.getLogger("email.verification")


ROLE_EMAIL_PREFIXES = {
    "admin",
    "billing",
    "careers",
    "contact",
    "hello",
    "help",
    "hr",
    "info",
    "office",
    "sales",
    "service",
    "support",
    "team",
}


PLACEHOLDER_EMAILS = {
    "email@domain.com",
    "name@example.com",
    "test@example.com",
    "user@domain.com",
    "user@example.com",
}

PLACEHOLDER_DOMAINS = {
    "domain.com",
    "example.com",
    "example.net",
    "example.org",
}


PROVIDER_MARKERS = {
    "google": ("google", "googlemail", "aspmx", "gmail"),
    "microsoft": ("outlook", "office365", "protection.outlook", "hotmail"),
    "zoho": ("zoho",),
    "proofpoint": ("proofpoint",),
    "mimecast": ("mimecast",),
    "godaddy": ("secureserver", "godaddy"),
}


FALLBACK_DNS_NAMESERVERS = ("1.1.1.1", "8.8.8.8", "9.9.9.9")


def _decode_smtp_message(message) -> str:
    if isinstance(message, bytes):
        return message.decode("utf-8", errors="ignore")
    return str(message or "")


def _provider_from_mx(mx_record: str) -> str:
    lowered = (mx_record or "").lower()
    for provider, markers in PROVIDER_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return provider
    return "unknown"


def _is_role_email(email: str) -> tuple[bool, str]:
    local = (email or "").split("@", 1)[0].lower().strip()
    local = re.sub(r"[^a-z0-9._+-]", "", local)
    prefix = local.split(".", 1)[0].split("-", 1)[0].split("_", 1)[0]
    if prefix in ROLE_EMAIL_PREFIXES:
        return True, prefix
    return False, ""


def _check(name: str, status: str, detail: str = "", **extra) -> dict:
    item = {"name": name, "status": status, "detail": detail}
    item.update(extra)
    return item


def _resolve_mx_records(dns_resolver, domain: str) -> tuple[list, str]:
    try:
        return list(dns_resolver.resolve(domain, "MX")), "default"
    except Exception as default_exc:
        resolver = dns_resolver.Resolver(configure=False)
        resolver.nameservers = list(FALLBACK_DNS_NAMESERVERS)
        resolver.timeout = 3
        resolver.lifetime = 6
        try:
            return list(resolver.resolve(domain, "MX")), "fallback_public_dns"
        except Exception:
            raise default_exc


def _log_stage(stage: str, result: dict, status: str, detail: str = "", **extra) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "email": result.get("normalized_email") or result.get("email") or "",
        "domain": result.get("domain") or "",
        "provider_type": result.get("provider_type") or "unknown",
        "mx_found": result.get("mx_found", False),
        "smtp_connected": result.get("smtp_connected", False),
        "smtp_verified": result.get("smtp_verified", False),
        "catch_all": result.get("catch_all", False),
        "is_role_email": result.get("is_role_email", False),
        "confidence": result.get("confidence", 0),
        "detail": detail,
    }
    payload.update(extra)
    logger.info("email_verification_stage", extra={"payload": json.dumps(payload, default=str)})

def verify_email_legitimacy(email: str, sender_email: str = "verify@agentic-core.ai") -> dict:
    """
    Verifies if an email address is likely legitimate using:
    1. Syntax validation
    2. DNS MX record lookup
    3. SMTP handshake
    4. Catch-all probe
    5. Recipient acceptance probe
    6. Role/generic mailbox detection

    Third-party imports (dnspython, email-validator) are intentionally deferred
    to inside this function so that a missing package in the Vercel runtime cache
    never prevents the app from starting up.
    """
    started_at = time.perf_counter()
    raw_email = str(email or "").strip()

    result = {
        "email": raw_email,
        "normalized_email": raw_email,
        "local_part": "",
        "domain": "",
        "valid_syntax": False,
        "domain_found": False,
        "mx_found": False,
        "mx_records": [],
        "selected_mx": "",
        "provider_type": "unknown",
        "smtp_connected": False,
        "smtp_verified": False,
        "catch_all": False,
        "catch_all_probe_code": None,
        "recipient_probe_code": None,
        "recipient_probe_message": "",
        "is_role_email": False,
        "is_placeholder_email": False,
        "role_email_type": "",
        "status": "invalid",
        "confidence": 0,
        "message": "",
        "checks": [],
        "duration_ms": 0,
    }

    # --- Lazy imports (kept inside function to avoid import-time crashes) ---
    try:
        import dns.resolver as _dns_resolver
    except ImportError:
        result["status"] = "error"
        result["message"] = "dnspython is not installed in this environment."
        result["checks"].append(_check("dependency_check", "failed", result["message"]))
        result["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
        _log_stage("dependency_check", result, "failed", result["message"])
        return result

    try:
        from email_validator import validate_email, EmailNotValidError
    except ImportError:
        result["status"] = "error"
        result["message"] = "email-validator is not installed in this environment."
        result["checks"].append(_check("dependency_check", "failed", result["message"]))
        result["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
        _log_stage("dependency_check", result, "failed", result["message"])
        return result
    # -----------------------------------------------------------------------
    _log_stage("dependency_check", result, "passed", "Required verification packages are available.")

    # 1. Syntax Validation
    try:
        valid = validate_email(raw_email, check_deliverability=False)
        email = valid.email
        local_part, domain = email.split("@", 1)
        is_role, role_type = _is_role_email(email)
        result.update(
            {
                "normalized_email": email,
                "local_part": local_part,
                "domain": domain,
                "valid_syntax": True,
                "is_role_email": is_role,
                "role_email_type": role_type,
            }
        )
        result["valid_syntax"] = True
        detail = "Email format is valid."
        if is_role:
            detail += f" Role mailbox detected: {role_type}@."
        result["checks"].append(_check("syntax_validation", "passed", detail))
        _log_stage(
            "syntax_validation",
            result,
            "passed",
            detail,
            local_part=local_part,
            role_email_type=role_type,
        )
        if email.lower() in PLACEHOLDER_EMAILS or domain.lower() in PLACEHOLDER_DOMAINS:
            result["is_placeholder_email"] = True
            result["status"] = "invalid"
            result["message"] = "Placeholder email detected; this is not a real fetched contact email."
            result["checks"].append(_check("placeholder_detection", "failed", result["message"]))
            result["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
            _log_stage("placeholder_detection", result, "failed", result["message"])
            return result
    except EmailNotValidError as e:
        result["message"] = str(e)
        result["checks"].append(_check("syntax_validation", "failed", result["message"]))
        result["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
        _log_stage("syntax_validation", result, "failed", result["message"])
        return result

    # 2. DNS/MX Record Lookup
    domain = result["domain"]
    try:
        raw_records, resolver_source = _resolve_mx_records(_dns_resolver, domain)
        records = sorted(raw_records, key=lambda record: int(record.preference))
        mx_records = [str(record.exchange).rstrip(".") for record in records]
        mx_record = mx_records[0]
        result["domain_found"] = True
        result["mx_found"] = True
        result["mx_records"] = mx_records
        result["selected_mx"] = mx_record
        result["provider_type"] = _provider_from_mx(mx_record)
        result["checks"].append(
            _check("dns_mx_lookup", "passed", f"Found {len(mx_records)} MX record(s).", mx_records=mx_records)
        )
        _log_stage(
            "dns_mx_lookup",
            result,
            "passed",
            f"Found {len(mx_records)} MX record(s).",
            selected_mx=mx_record,
            mx_count=len(mx_records),
            resolver_source=resolver_source,
        )
    except Exception as exc:
        result["message"] = f"No MX records found for {domain}: {type(exc).__name__}"
        result["checks"].append(_check("dns_mx_lookup", "failed", result["message"]))
        result["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
        _log_stage("dns_mx_lookup", result, "failed", result["message"])
        return result

    # 3. SMTP Handshake
    server = None
    try:
        host = socket.gethostname()

        server = smtplib.SMTP(timeout=10)
        server.set_debuglevel(0)

        connect_code, connect_message = server.connect(mx_record)
        result["smtp_connected"] = True
        result["checks"].append(
            _check(
                "smtp_connection",
                "passed",
                f"Connected to {mx_record}.",
                response_code=connect_code,
                response_message=_decode_smtp_message(connect_message),
            )
        )
        _log_stage(
            "smtp_connection",
            result,
            "passed",
            f"Connected to {mx_record}.",
            response_code=connect_code,
        )

        helo_code, helo_message = server.helo(host)
        helo_status = "passed" if 200 <= helo_code < 400 else "warning"
        result["checks"].append(
            _check(
                "smtp_handshake",
                helo_status,
                f"SMTP HELO returned {helo_code}.",
                response_code=helo_code,
                response_message=_decode_smtp_message(helo_message),
            )
        )
        _log_stage("smtp_handshake", result, helo_status, f"SMTP HELO returned {helo_code}.", response_code=helo_code)

        mail_code, mail_message = server.mail(sender_email)
        mail_status = "passed" if 200 <= mail_code < 400 else "warning"
        result["checks"].append(
            _check(
                "smtp_sender_probe",
                mail_status,
                f"MAIL FROM returned {mail_code}.",
                response_code=mail_code,
                response_message=_decode_smtp_message(mail_message),
            )
        )
        _log_stage("smtp_sender_probe", result, mail_status, f"MAIL FROM returned {mail_code}.", response_code=mail_code)
        
        # 3a. Check for Catch-all
        random_email = f"verify_test_{socket.gethostname()[:5]}_{int(time.time())}@{domain}"
        code_rand, rand_message = server.rcpt(random_email)
        is_catch_all = (code_rand == 250)
        result["catch_all"] = is_catch_all
        result["catch_all_probe_code"] = code_rand
        result["checks"].append(
            _check(
                "catch_all_detection",
                "warning" if is_catch_all else "passed",
                "Domain accepts random mailbox probes." if is_catch_all else "Random mailbox was not accepted.",
                response_code=code_rand,
                response_message=_decode_smtp_message(rand_message),
            )
        )
        _log_stage(
            "catch_all_detection",
            result,
            "warning" if is_catch_all else "passed",
            "Domain accepts random mailbox probes." if is_catch_all else "Random mailbox was not accepted.",
            response_code=code_rand,
        )

        # 3b. Check the actual email
        code, message = server.rcpt(email)
        msg_text = _decode_smtp_message(message)
        result["recipient_probe_code"] = code
        result["recipient_probe_message"] = msg_text
        result["checks"].append(
            _check(
                "recipient_acceptance",
                "passed" if code == 250 else ("failed" if code == 550 else "warning"),
                f"Recipient probe returned {code}.",
                response_code=code,
                response_message=msg_text,
            )
        )
        recipient_status = "passed" if code == 250 else ("failed" if code == 550 else "warning")
        _log_stage(
            "recipient_acceptance",
            result,
            recipient_status,
            f"Recipient probe returned {code}.",
            response_code=code,
        )

        if code == 250:
            result["smtp_verified"] = not is_catch_all
            if is_catch_all:
                result["status"] = "risky"
                result["confidence"] = 55 if not result["is_role_email"] else 45
                result["message"] = "Server accepted the mailbox, but the domain is catch-all, so this is risky."
            else:
                result["status"] = "valid"
                result["confidence"] = 92 if not result["is_role_email"] else 78
                result["message"] = "Mailbox was accepted by the mail server."
        elif code == 550:
            if result["provider_type"] in {"google", "microsoft"}:
                result["status"] = "risky"
                result["confidence"] = 45
                result["message"] = "Protected Google/Microsoft server rejected the probe; mailbox may still exist."
            else:
                result["status"] = "invalid"
                result["confidence"] = 12
                result["message"] = "Mailbox does not exist (550)."
        else:
            result["status"] = "risky"
            result["confidence"] = 40
            result["message"] = f"Server responded with code {code}: {msg_text}"

    except Exception as e:
        result["status"] = "risky"
        result["confidence"] = 35 if result["mx_found"] else 15
        result["message"] = f"SMTP Verification failed: {str(e)}"
        result["checks"].append(_check("smtp_verification", "warning", result["message"]))
        _log_stage("smtp_verification", result, "warning", result["message"])
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass
        if result["is_role_email"] and result["status"] == "valid":
            result["message"] += " This is a generic role mailbox, not a person-specific email."
        result["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
        _log_stage(
            "final_decision",
            result,
            result.get("status") or "unknown",
            result.get("message") or "",
            duration_ms=result["duration_ms"],
            recipient_probe_code=result.get("recipient_probe_code"),
            catch_all_probe_code=result.get("catch_all_probe_code"),
        )

    return result
