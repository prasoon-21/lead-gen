import asyncio
import json
import logging
import re
import time
import uuid
from urllib.parse import urljoin, urlparse
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.schemas import AgentSpec
from core.adapters.base import BaseLLMAdapter
from core.agents.memory import MemoryStore
from core.services.lead_discovery_policy import (
    DEFAULT_TAVILY_SEARCH_DEPTH,
    build_targeted_directory_queries,
    clamp_web_search_max_results,
    is_whitelisted_directory_url,
    looks_like_bad_directory_path,
)
from core.services.production_lead_pipeline import ProductionLeadPipeline
from core.tools import ToolContext, ToolExecutor, ToolPolicy, ToolRegistry
from core.tools.external.tavily_client import TavilyClient
from core.services.lead_quality_service import score_lead


class AgentKernel:
    DECISION_SCHEMA = {
        "type": "tool_call or final_answer",
        "thought": "short reasoning for logs",
        "tool_name": "required when type=tool_call",
        "arguments": {},
        "final_answer": "required when type=final_answer",
    }
    CONFIRMATION_PATTERN = re.compile(
        r"\b(yes|confirm|confirmed|go ahead|proceed|approved|approve|do it)\b",
        re.IGNORECASE,
    )
    _email_pattern = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
    _phone_pattern = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
    _contact_path_pattern = re.compile(
        r'href=["\']([^"\']*(?:contact|about|team|leadership)[^"\']*)["\']',
        re.IGNORECASE,
    )
    _anchor_pattern = re.compile(r'(?is)<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>')
    _title_pattern = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
    _meta_description_pattern = re.compile(
        r'(?is)<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']'
    )
    _noisy_domains = {
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "crunchbase.com",
        "zoominfo.com",
        "clutch.co",
        "goodfirms.co",
        "yelp.com",
        "builtin.com",
        "wellfound.com",
        "growthlist.co",
        "builtinmumbai.in",
        "angel.co",
        "tracxn.com",
        "yourstory.com",
        "startupblink.com",
        "topdevelopers.co",
        "scribd.com",
    }
    _noisy_title_markers = (
        "top ",
        "best ",
        "companies in ",
        "startups in ",
        "funded ",
        "list of ",
        "directory",
        "agencies in ",
        "firms in ",
        "providers in ",
    )
    _bad_company_name_markers = (
        "companies",
        "startups",
        "directory",
        "list",
        "funded",
        "agencies",
        "firms",
        "providers",
    )
    _directory_title_markers = (
        "directory",
        "member",
        "members",
        "listing",
        "list of",
        "companies",
        "businesses",
        "suppliers",
        "providers",
        "firms",
        "association",
        "chamber",
        "contractors",
        "vendors",
    )
    _directory_path_markers = (
        "directory",
        "member",
        "members",
        "listing",
        "list",
        "companies",
        "businesses",
        "suppliers",
        "providers",
        "firms",
        "association",
        "chamber",
        "contractors",
        "vendors",
    )
    _internal_listing_markers = (
        "member",
        "listing",
        "profile",
        "company",
        "business",
        "supplier",
        "provider",
        "vendor",
        "firm",
    )
    _ignore_anchor_prefixes = ("mailto:", "tel:", "javascript:", "#")
    _social_domains = {
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "youtube.com",
        "tiktok.com",
    }
    _blocked_page_markers = (
        "checking your browser",
        "security checkpoint",
        "access denied",
        "captcha",
        "cf-chl",
        "cloudflare",
        "aws waf",
        "vercel security checkpoint",
        "x-vercel-challenge-token",
        "enable javascript and cookies",
        "bot verification",
        "human verification",
    )
    _parked_page_markers = (
        "domain for sale",
        "buy this domain",
        "this domain may be for sale",
        "parked free",
        "sedo domain parking",
        "godaddy parked",
        "coming soon",
        "under construction",
        "website coming soon",
        "default web site page",
        "placeholder",
    )
    _directory_block_pattern = re.compile(
        r'(?is)<(tr|li|article|section|div)[^>]*(?:class|id)=["\'][^"\']*'
        r'(?:member|listing|company|business|supplier|provider|vendor|profile|card|result)'
        r'[^"\']*["\'][^>]*>(.*?)</\1>'
    )
    _heading_pattern = re.compile(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>")
    _website_label_pattern = re.compile(
        r'(?is)(?:website|visit|homepage|official site)\s*</[^>]+>\s*<a[^>]+href=["\']([^"\']+)["\']'
    )
    _directory_type_priorities = {
        "association_member_directory": 1,
        "chamber_member_directory": 1,
        "industry_member_directory": 1,
        "vendor_partner_directory": 2,
        "business_directory": 2,
        "local_business_directory": 2,
        "company_listing_directory": 3,
        "general_listing_page": 4,
        "listicle_roundup": 8,
    }
    _manual_directory_discovery_ids = {"lead_directory_discovery_agent"}
    _manual_company_extraction_ids = {"lead_directory_company_extraction_agent"}
    _manual_company_validation_ids = {"lead_company_validation_agent"}
    _manual_contact_enrichment_ids = {
        "lead_contact_enrichment_agent",
        "lead_company_contact_enrichment_agent",
        "lead_people_contact_enrichment_agent",
    }
    _manual_linkedin_enrichment_ids = {"lead_linkedin_enrichment_agent"}
    _manual_quality_scoring_ids = {"lead_quality_scoring_agent"}
    _browser_like_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(
        self,
        llm: BaseLLMAdapter,
        tool_registry: ToolRegistry,
        token_tracker=None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.executor = ToolExecutor(tool_registry)
        self.token_tracker = token_tracker
        self.memory_store = memory_store or MemoryStore(max_turns=6)
        self._runtime_logger = logging.getLogger("agent.runtime")

    def _system_prompt(self, spec: AgentSpec, system_prompt_text: str = "", developer_prompt_text: str = "") -> str:
        parts: List[str] = []
        if system_prompt_text:
            parts.append(system_prompt_text)
        elif spec.system_prompt:
            parts.append(spec.system_prompt)

        if developer_prompt_text:
            parts.append(developer_prompt_text)
        elif spec.developer_prompt:
            parts.append(spec.developer_prompt)

        parts.append(
            "You are a tool-calling orchestrator. Decide one action per step.\n"
            "If a tool is needed, return type=tool_call.\n"
            "If enough information is available, return type=final_answer."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _preview_text(value: Any, limit: int = 400) -> str:
        cleaned = " ".join(str(value or "").split()).strip()
        if len(cleaned) > limit:
            return cleaned[:limit] + "...[truncated]"
        return cleaned

    @classmethod
    def _preview_payload(cls, value: Any, limit: int = 1200) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = cls._preview_text(value, limit=limit)
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    def _llm_usage(self) -> Dict[str, Any]:
        usage = getattr(self.llm, "last_usage", {}) or {}
        return {
            "model": usage.get("model", getattr(self.llm, "model_name", "")),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "latency_ms": float(usage.get("latency_ms", 0) or 0),
            "mode": usage.get("mode", ""),
            "parse_status": usage.get("parse_status", ""),
        }

    @staticmethod
    def _normalize_source_details(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
        return []

    @classmethod
    def _append_source_detail(
        cls,
        target: Dict[str, Any],
        *,
        stage: str,
        source_type: str,
        value: str = "",
        url: str = "",
        provider: str = "",
        query: str = "",
        title: str = "",
    ) -> None:
        details = cls._normalize_source_details(target.get("source_details"))
        entry = {
            "stage": cls._clean_text(stage),
            "type": cls._clean_text(source_type),
            "value": cls._clean_text(value),
            "url": cls._clean_text(url),
            "provider": cls._clean_text(provider),
            "query": cls._clean_text(query),
            "title": cls._clean_text(title),
        }
        entry = {key: item for key, item in entry.items() if item}
        if not entry:
            return
        if entry not in details:
            details.append(entry)
        target["source_details"] = details

    @classmethod
    def _merge_source_details(cls, target: Dict[str, Any], *values: Any) -> None:
        details = cls._normalize_source_details(target.get("source_details"))
        for value in values:
            for item in cls._normalize_source_details(value):
                if item not in details:
                    details.append(item)
        if details:
            target["source_details"] = details

    @staticmethod
    def _increment_counter(counter: Dict[str, int], key: str, amount: int = 1) -> None:
        if not key:
            return
        counter[key] = int(counter.get(key, 0) or 0) + amount

    @classmethod
    def _has_contact_path(cls, lead: Dict[str, Any]) -> bool:
        return any(
            cls._clean_text(lead.get(field))
            for field in ("contact_email", "contact_phone", "linkedin_url", "contact_page")
        )

    @classmethod
    def _has_linkedin_person(cls, lead: Dict[str, Any]) -> bool:
        linkedin_url = cls._clean_text(lead.get("linkedin_url"))
        person_name = cls._first_non_empty(lead.get("contact_person_name"), lead.get("founder_name"))
        return bool(linkedin_url and person_name)

    @classmethod
    def _normalize_directory_record(cls, item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        directory_name = cls._clean_text(item.get("directory_name") or item.get("name"))
        directory_url = cls._clean_text(item.get("directory_url") or item.get("url"))
        if not directory_name or not directory_url.startswith("http"):
            return None
        directory_type = cls._clean_text(item.get("directory_type")) or cls._classify_directory_type(
            title=directory_name,
            snippet=cls._clean_text(item.get("snippet") or item.get("content")),
            url=directory_url,
        )
        return {
            "directory_name": directory_name[:180],
            "directory_url": directory_url.rstrip("/"),
            "directory_type": directory_type or "business_directory",
            "industry": cls._clean_text(item.get("industry")) or cls._clean_text(context.get("industry")),
            "location": cls._clean_text(item.get("location")) or cls._clean_text(context.get("location")),
            "source": cls._clean_text(item.get("source")) or "web_search",
            "source_details": cls._normalize_source_details(item.get("source_details")),
        }

    @classmethod
    def _normalize_company_record(cls, item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        company_website = cls._normalize_website(item.get("company_website") or item.get("website") or "")
        company_name = cls._clean_text(item.get("company_name") or item.get("name"))
        if not company_website or not company_name:
            return None
        contact_page = cls._clean_text(item.get("contact_page"))
        normalized = {
            "company_name": company_name[:180],
            "company_website": company_website,
            "industry": cls._clean_text(item.get("industry")) or cls._clean_text(context.get("industry")),
            "location": cls._clean_text(item.get("location")) or cls._clean_text(context.get("location")),
            "value_proposition": cls._clean_text(item.get("value_proposition"))[:280],
            "company_size": cls._clean_text(item.get("company_size")),
            "contact_page": contact_page,
            "source": item.get("source") if isinstance(item.get("source"), list) else cls._clean_text(item.get("source")),
            "source_details": cls._normalize_source_details(item.get("source_details")),
        }
        return normalized

    @classmethod
    def _normalize_lead_record(cls, item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        company = cls._normalize_company_record(item, context)
        if not company:
            return None
        lead = {
            **company,
            "contact_person_name": cls._clean_text(item.get("contact_person_name")),
            "founder_name": cls._clean_text(item.get("founder_name")),
            "contact_person_title": cls._clean_text(item.get("contact_person_title")),
            "contact_email": cls._clean_text(item.get("contact_email")).lower(),
            "contact_phone": cls._clean_text(item.get("contact_phone")),
            "linkedin_url": cls._clean_text(item.get("linkedin_url")),
            "confidence": cls._clean_text(item.get("confidence")).lower() or "low",
            "quality_score": int(item.get("quality_score", 0) or 0),
            "quality_status": cls._clean_text(item.get("quality_status")),
            "usable": bool(item.get("usable", False)),
            "rejected": bool(item.get("rejected", False)),
            "quality_notes": item.get("quality_notes") if isinstance(item.get("quality_notes"), list) else [],
            "quality_warnings": item.get("quality_warnings") if isinstance(item.get("quality_warnings"), list) else [],
            "verification_status": cls._clean_text(item.get("verification_status")),
            "verification_message": cls._clean_text(item.get("verification_message")),
            "field_sources": item.get("field_sources") if isinstance(item.get("field_sources"), dict) else {},
        }
        return lead

    @classmethod
    def _finalize_directory_records(cls, items: List[Dict[str, Any]], context: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            record = cls._normalize_directory_record(item, context)
            if not record:
                continue
            key = record["directory_url"]
            if key in seen:
                continue
            seen.add(key)
            normalized.append(record)
        normalized.sort(
            key=lambda item: (
                cls._directory_type_priorities.get(item.get("directory_type", ""), 5),
                len(item.get("directory_name", "")),
            )
        )
        return normalized[:limit]

    @classmethod
    def _finalize_company_records(cls, items: List[Dict[str, Any]], context: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            record = cls._normalize_company_record(item, context)
            if not record:
                continue
            key = record["company_website"]
            if key in seen:
                continue
            seen.add(key)
            normalized.append(record)
            if len(normalized) >= limit:
                break
        return normalized

    @classmethod
    def _finalize_lead_records(cls, items: List[Dict[str, Any]], context: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            record = cls._normalize_lead_record(item, context)
            if not record:
                continue
            dedupe_key = (
                record["company_website"],
                record.get("contact_email", ""),
                record.get("linkedin_url", ""),
                record.get("contact_person_name", "") or record.get("founder_name", ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(record)
            if len(normalized) >= limit:
                break
        return normalized

    def _decision_prompt(
        self,
        user_message: str,
        context: Dict[str, Any],
        memory_context: str,
        tool_specs: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
        step: int,
        max_steps: int,
    ) -> str:
        return (
            f"Step: {step}/{max_steps}\n"
            f"User message: {user_message}\n"
            f"Context JSON: {json.dumps(context, ensure_ascii=False)}\n"
            f"Memory:\n{memory_context or '(empty)'}\n"
            f"Available tools JSON: {json.dumps(tool_specs, ensure_ascii=False)}\n"
            f"Tool history JSON: {json.dumps(tool_history, ensure_ascii=False)}\n\n"
            "Return JSON with this schema:\n"
            f"{json.dumps(self.DECISION_SCHEMA, ensure_ascii=False)}\n"
            "Rules:\n"
            "- Use exactly one action per step.\n"
            "- type must be 'tool_call' or 'final_answer'.\n"
            "- If tool_call, include tool_name and arguments.\n"
            "- If final_answer, include final_answer string.\n"
            "- Keep thought concise."
        )

    def _log_step(self, trace_id: str, event: str, payload: Dict[str, Any]) -> None:
        step = int(payload.get("step", 0))
        session_id = str(payload.get("session_id", ""))
        agent_id = str(payload.get("agent_id", ""))
        tool_name = str(payload.get("tool_name", "-"))
        status = str(payload.get("decision_type") or payload.get("ok") or payload.get("error") or "-")
        self._runtime_logger.info(
            event,
            extra={
                "event": event,
                "trace_id": trace_id,
                "session_id": session_id,
                "agent_id": agent_id,
                "workflow_id": "-",
                "step": step,
                "tool_name": tool_name,
                "status": status,
                "payload": self._preview_payload(payload),
            },
        )
        if not self.token_tracker:
            return
        if hasattr(self.token_tracker, "log_agent_step"):
            self.token_tracker.log_agent_step(
                trace_id=trace_id,
                session_id=session_id,
                agent_id=agent_id,
                step=step,
                event=event,
                payload=payload,
            )
        else:
            self.token_tracker.log_event(
                event=event,
                data={"trace_id": trace_id, **payload},
            )

    @staticmethod
    def _is_lead_generation_agent(spec: AgentSpec) -> bool:
        return spec.agent_id == "lead_gen_agent"

    async def _run_production_lead_generation(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        industry = self._clean_text(context.get("industry"))
        location = self._clean_text(context.get("location"))
        try:
            target_lead_count = int(context.get("target_lead_count") or 15)
        except (TypeError, ValueError):
            target_lead_count = 15
        target_lead_count = max(1, min(target_lead_count, 20))

        pipeline = ProductionLeadPipeline()
        result = await pipeline.run(
            industry=industry,
            location=location,
            seed_query=message,
            target_count=target_lead_count,
        )

        leads = self._finalize_lead_records(result.get("leads") or [], context, target_lead_count)
        json_output = {"leads": leads}
        self.memory_store.add_message(session_id, "user", message)
        self.memory_store.add_message(session_id, "assistant", json.dumps(json_output, ensure_ascii=False))

        metadata = {
            "agent_id": spec.agent_id,
            "tool_calls": 0,
            "completed": True,
            "manual_flow": True,
            "production_pipeline": True,
            "contact_info_count": sum(1 for lead in leads if self._has_contact_path(lead)),
            "linkedin_people_count": sum(1 for lead in leads if self._has_linkedin_person(lead)),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": (time.perf_counter() - run_started) * 1000,
        }
        metadata.update(result.get("metadata") or {})

        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": json_output,
            "steps": result.get("steps") or [],
            "metadata": metadata,
        }

    @staticmethod
    def _is_manual_discovery_agent(spec: AgentSpec) -> bool:
        return spec.agent_id == "lead_company_discovery_agent"

    @classmethod
    def _is_manual_enrichment_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in {"lead_contact_enrichment_agent", "lead_company_contact_enrichment_agent"}

    @classmethod
    def _is_manual_directory_discovery_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in cls._manual_directory_discovery_ids

    @classmethod
    def _is_manual_company_extraction_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in cls._manual_company_extraction_ids

    @classmethod
    def _is_manual_company_validation_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in cls._manual_company_validation_ids

    @classmethod
    def _is_manual_contact_stage_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in cls._manual_contact_enrichment_ids - {"lead_company_contact_enrichment_agent"}

    @classmethod
    def _is_manual_linkedin_stage_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in cls._manual_linkedin_enrichment_ids

    @classmethod
    def _is_manual_quality_stage_agent(cls, spec: AgentSpec) -> bool:
        return spec.agent_id in cls._manual_quality_scoring_ids

    @classmethod
    def _is_manual_lead_pipeline_agent(cls, spec: AgentSpec) -> bool:
        return (
            cls._is_manual_discovery_agent(spec)
            or cls._is_manual_enrichment_agent(spec)
            or cls._is_manual_directory_discovery_agent(spec)
            or cls._is_manual_company_extraction_agent(spec)
            or cls._is_manual_company_validation_agent(spec)
            or cls._is_manual_contact_stage_agent(spec)
            or cls._is_manual_linkedin_stage_agent(spec)
            or cls._is_manual_quality_stage_agent(spec)
        )

    @staticmethod
    def _normalize_company_name(title: str, url: str) -> str:
        cleaned_title = (title or "").strip()
        if cleaned_title:
            for separator in (" | ", " - ", " — ", " :: ", ": "):
                if separator in cleaned_title:
                    cleaned_title = cleaned_title.split(separator)[0].strip()
                    break
        if cleaned_title:
            return cleaned_title

        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return "Unknown Company"
        base = host.split(".")[0].replace("-", " ").replace("_", " ").strip()
        return base.title() if base else host

    def _build_forced_search_arguments(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        industry = str(context.get("industry") or "").strip()
        location = str(context.get("location") or "").strip()
        target_role = str(context.get("target_role") or "").strip()
        target_count = int(context.get("target_company_pool") or max(12, int(context.get("target_lead_count") or 15) * 2))

        if industry and location:
            query = f"{industry} companies in {location}"
        elif message.strip():
            query = message.strip()
        else:
            query = "B2B companies"

        if target_role:
            query = f"{query} {target_role}"

        return {
            "query": query,
            "industry": industry,
            "location": location,
            "expand_queries": False,
            "max_results": clamp_web_search_max_results(target_count),
            "max_results_per_query": 4,
            "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
            "include_raw_content": False,
        }

    @staticmethod
    def _safe_json_list(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
                if isinstance(parsed, dict):
                    for key in ("companies", "directories", "leads"):
                        items = parsed.get(key)
                        if isinstance(items, list):
                            return [item for item in items if isinstance(item, dict)]
            except Exception:
                return []
        return []

    @staticmethod
    def _clean_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    @classmethod
    def _extract_domain(cls, url: str) -> str:
        host = (urlparse(url or "").hostname or "").lower().strip()
        return host[4:] if host.startswith("www.") else host

    @classmethod
    def _normalize_website(cls, url: str) -> str:
        parsed = urlparse((url or "").strip())
        if not parsed.netloc:
            return ""
        scheme = parsed.scheme or "https"
        return f"{scheme}://{parsed.netloc}".rstrip("/")

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            cleaned = " ".join(str(value or "").split()).strip()
            if cleaned:
                return cleaned
        return ""

    @staticmethod
    def _keyword_tokens(value: str, *, min_length: int = 3) -> List[str]:
        stopwords = {
            "and", "the", "for", "with", "from", "that", "this", "into", "your",
            "their", "company", "companies", "services", "solutions", "group",
            "inc", "llc", "ltd", "co", "of", "in", "on", "at", "to",
        }
        tokens: List[str] = []
        for token in re.findall(r"[a-z0-9]+", (value or "").lower()):
            if len(token) < min_length or token in stopwords:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens

    @classmethod
    def _lead_relevance_signals(cls, lead: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        industry = cls._clean_text(context.get("industry")).lower()
        location = cls._clean_text(context.get("location")).lower()
        source_details = cls._normalize_source_details(lead.get("source_details"))
        evidence_parts = [
            cls._clean_text(lead.get("company_name")),
            cls._clean_text(lead.get("company_website")),
            cls._clean_text(lead.get("value_proposition")),
            cls._clean_text(lead.get("contact_page")),
            cls._clean_text(lead.get("linkedin_url")),
        ]
        for item in source_details:
            evidence_parts.extend(
                [
                    cls._clean_text(item.get("url")),
                    cls._clean_text(item.get("title")),
                    cls._clean_text(item.get("value")),
                ]
            )
        evidence_text = " ".join(part for part in evidence_parts if part).lower()

        industry_tokens = cls._keyword_tokens(industry)
        location_tokens = cls._keyword_tokens(location, min_length=2)
        industry_matches = [token for token in industry_tokens if token in evidence_text]
        location_matches = [token for token in location_tokens if token in evidence_text]

        return {
            "industry_required": bool(industry_tokens),
            "location_required": bool(location_tokens),
            "industry_matches": industry_matches,
            "location_matches": location_matches,
            "industry_match": not industry_tokens or bool(industry_matches),
            "location_match": not location_tokens or bool(location_matches),
            "evidence_preview": cls._preview_text(evidence_text, limit=220),
        }

    @classmethod
    def _extract_emails(cls, text: str) -> List[str]:
        emails: List[str] = []
        for match in cls._email_pattern.findall(text or ""):
            cleaned = match.strip().lower()
            if cleaned not in emails:
                emails.append(cleaned)
        return emails[:5]

    @classmethod
    def _extract_phones(cls, text: str) -> List[str]:
        phones: List[str] = []
        for match in cls._phone_pattern.findall(text or ""):
            cleaned = cls._clean_text(match)
            if len(re.sub(r"\D", "", cleaned)) < 8:
                continue
            if cleaned not in phones:
                phones.append(cleaned)
        return phones[:5]

    @classmethod
    def _extract_contact_links(cls, base_url: str, html: str) -> List[str]:
        links: List[str] = []
        base = cls._normalize_website(base_url)
        for match in cls._contact_path_pattern.findall(html or ""):
            href = cls._clean_text(match)
            if not href:
                continue
            if href.startswith("http"):
                normalized = href.rstrip("/")
            elif href.startswith("/"):
                normalized = f"{base}{href}".rstrip("/")
            else:
                normalized = f"{base}/{href.lstrip('/')}".rstrip("/")
            if normalized not in links:
                links.append(normalized)
        return links[:6]

    @staticmethod
    def _result_text(result_dict: Dict[str, Any]) -> str:
        return str(((result_dict.get("data") or {}).get("text")) or "")

    @staticmethod
    def _result_json(result_dict: Dict[str, Any]) -> Dict[str, Any]:
        return ((result_dict.get("data") or {}).get("json")) or {}

    @classmethod
    def _looks_parked_or_placeholder(cls, title: str, description: str, body: str) -> bool:
        haystack = " ".join(
            part.lower()
            for part in (title or "", description or "", body or "")
            if part
        )
        return any(marker in haystack for marker in cls._parked_page_markers)

    async def _validate_website_candidate(
        self,
        client: httpx.AsyncClient,
        candidate: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        company_name = self._clean_text(candidate.get("company_name") or candidate.get("name"))
        website_url = self._normalize_website(candidate.get("company_website") or candidate.get("website") or "")
        diagnostic = {
            "company_name": company_name,
            "requested_url": website_url,
            "final_url": "",
            "status_code": 0,
            "ok": False,
            "reason": "",
            "title": "",
            "meta_description": "",
            "contact_page": "",
            "value_proposition": "",
            "response_bytes": 0,
            "source_details": self._normalize_source_details(candidate.get("source_details")),
        }
        if not company_name:
            diagnostic["reason"] = "missing_company_name"
            return diagnostic
        if not website_url:
            diagnostic["reason"] = "missing_company_website"
            return diagnostic
        if self._is_noisy_domain(website_url):
            diagnostic["reason"] = "noisy_company_domain"
            return diagnostic

        try:
            response = await client.get(
                website_url,
                headers=dict(self._browser_like_headers),
            )
            body = response.text or ""
            final_url = str(response.url)
            status_code = int(response.status_code or 0)
            title = self._parse_title_from_html(body)
            meta_description = self._parse_meta_description(body)
            refined_name = self._best_company_name_from_page(title, final_url, company_name)
            contact_links = self._extract_contact_links(final_url, body)
            diagnostic.update(
                {
                    "final_url": self._normalize_website(final_url) or website_url,
                    "status_code": status_code,
                    "title": title[:180],
                    "meta_description": meta_description[:280],
                    "contact_page": contact_links[0] if contact_links else "",
                    "value_proposition": meta_description[:280],
                    "response_bytes": len(body.encode("utf-8", errors="ignore")),
                }
            )
            if status_code >= 400:
                diagnostic["reason"] = f"http_{status_code}"
                return diagnostic
            if self._is_blocked_page(body, status_code):
                diagnostic["reason"] = "blocked_page"
                return diagnostic
            if self._looks_parked_or_placeholder(title, meta_description, body):
                diagnostic["reason"] = "parked_or_placeholder_page"
                return diagnostic
            if not refined_name or self._looks_like_listing_result(refined_name, meta_description, final_url):
                diagnostic["reason"] = "homepage_failed_validation"
                return diagnostic

            validated_candidate = dict(candidate)
            validated_candidate["company_name"] = refined_name
            validated_candidate["company_website"] = diagnostic["final_url"] or website_url
            if meta_description:
                validated_candidate["value_proposition"] = meta_description[:280]
            if diagnostic["contact_page"]:
                validated_candidate["contact_page"] = diagnostic["contact_page"]
            self._append_source_detail(
                validated_candidate,
                stage="company_validation",
                source_type="homepage_validation",
                provider="direct_http",
                url=validated_candidate["company_website"],
                title=title[:180],
                value=refined_name,
            )
            if diagnostic["contact_page"]:
                self._append_source_detail(
                    validated_candidate,
                    stage="company_validation",
                    source_type="contact_page",
                    provider="direct_http",
                    url=diagnostic["contact_page"],
                    value=refined_name,
                )
            diagnostic["company"] = validated_candidate
            diagnostic["ok"] = True
            diagnostic["reason"] = "validated"
            diagnostic["source_details"] = self._normalize_source_details(validated_candidate.get("source_details"))
            return diagnostic
        except httpx.TimeoutException:
            diagnostic["reason"] = "request_timeout"
            return diagnostic
        except httpx.HTTPError as exc:
            diagnostic["reason"] = f"http_error:{type(exc).__name__}"
            return diagnostic
        except Exception as exc:
            diagnostic["reason"] = f"validation_error:{type(exc).__name__}"
            return diagnostic

    async def _validate_company_candidates_batch(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        timeout = httpx.Timeout(15.0, connect=8.0)
        limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, limits=limits) as client:
            tasks = [self._validate_website_candidate(client, candidate, context) for candidate in candidates]
            return await asyncio.gather(*tasks)

    @staticmethod
    def _extract_text_from_tavily_extract_item(item: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ("raw_content", "content", "markdown", "text", "excerpt"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            for key in ("description", "title"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
        return "\n".join(parts)

    @classmethod
    def _extract_linkedin_urls(cls, text: str) -> List[str]:
        matches = re.findall(r"https?://(?:www\.)?linkedin\.com/(?:in|company)/[^\s)\]>\"']+", text or "", re.IGNORECASE)
        urls: List[str] = []
        for match in matches:
            cleaned = match.rstrip(".,);]")
            if cleaned not in urls:
                urls.append(cleaned)
        return urls[:5]

    @classmethod
    def _extract_contact_page_candidates(cls, base_url: str, text: str) -> List[str]:
        matches = re.findall(r"https?://[^\s)\]>\"']+", text or "", re.IGNORECASE)
        contact_pages: List[str] = []
        base_domain = cls._extract_domain(base_url)
        for match in matches:
            cleaned = match.rstrip(".,);]")
            if cls._extract_domain(cleaned) != base_domain:
                continue
            if any(tag in cleaned.lower() for tag in ("contact", "about", "team", "leadership")) and cleaned not in contact_pages:
                contact_pages.append(cleaned)
        return contact_pages[:4]

    async def _run_tavily_extract_for_companies(
        self,
        *,
        session_id: str,
        trace_id: str,
        agent_id: str,
        companies: List[Dict[str, Any]],
        step_logs: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        urls = []
        for company in companies:
            website_url = self._first_non_empty(company.get("company_website"))
            if website_url and website_url not in urls:
                urls.append(website_url)
        if not urls:
            return {}

        try:
            tavily = TavilyClient()
        except Exception as exc:
            self._log_step(
                trace_id,
                "tavily_extract_skipped",
                {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "step": len(step_logs) + 1,
                    "tool_name": "web_extract",
                    "error": str(exc),
                    "url_count": len(urls),
                },
            )
            return {}

        extracts_by_url: Dict[str, Dict[str, Any]] = {}
        for index in range(0, len(urls), 5):
            batch_urls = urls[index:index + 5]
            step = len(step_logs) + 1
            started = time.perf_counter()
            try:
                response = await tavily.extract({"urls": batch_urls, "extract_depth": "basic"})
                result_items = response.get("results")
                if not isinstance(result_items, list):
                    result_items = response.get("data") if isinstance(response.get("data"), list) else []
                for item in result_items or []:
                    if not isinstance(item, dict):
                        continue
                    item_url = self._normalize_website(item.get("url") or item.get("source_url") or "")
                    if not item_url:
                        continue
                    extracted_text = self._extract_text_from_tavily_extract_item(item)
                    extracts_by_url[item_url] = {
                        "url": item_url,
                        "text": extracted_text,
                        "raw": item,
                    }
                result_dict = {"data": response, "ok": True}
            except Exception as exc:
                result_dict = {"data": {}, "ok": False, "error": str(exc)}
            latency_ms = (time.perf_counter() - started) * 1000
            step_log = {
                "step": step,
                "stage": "tavily_extract",
                "type": "tool_call",
                "tool_name": "web_extract",
                "arguments": {"urls": batch_urls, "extract_depth": "basic"},
                "result": result_dict,
                "latency_ms": latency_ms,
            }
            step_logs.append(step_log)
            tool_history.append(step_log)
            self._log_step(
                trace_id,
                "agent_tool_call",
                {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "step": step,
                    "tool_name": "web_extract",
                    "ok": result_dict.get("ok", False),
                    "latency_ms": latency_ms,
                    "error": result_dict.get("error", ""),
                    "stage": "tavily_extract",
                    "url_count": len(batch_urls),
                },
            )
        return extracts_by_url

    async def _execute_manual_tool(
        self,
        *,
        session_id: str,
        trace_id: str,
        agent_id: str,
        step: int,
        stage: str,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
        resources: Dict[str, Any],
        policy: ToolPolicy,
        step_logs: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tool_context = ToolContext(
            session_id=session_id,
            trace_id=trace_id,
            resources=resources,
            state=context,
        )
        result = await self.executor.execute(
            tool_name=tool_name,
            arguments=arguments,
            context=tool_context,
            policy=policy,
        )
        result_dict = result.to_dict()
        step_log = {
            "step": step,
            "stage": stage,
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result_dict,
            "latency_ms": result.latency_ms,
        }
        step_logs.append(step_log)
        tool_history.append(step_log)
        context[f"tool_{step}_{tool_name}"] = result_dict
        self._log_step(
            trace_id,
            "agent_tool_call",
            {
                "session_id": session_id,
                "agent_id": agent_id,
                "step": step,
                "tool_name": tool_name,
                "ok": result.ok,
                "latency_ms": result.latency_ms,
                "error": result.error,
                "stage": stage,
            },
        )
        return result_dict

    def _build_manual_discovery_queries(self, message: str, context: Dict[str, Any]) -> List[str]:
        industry = self._clean_text(context.get("industry"))
        location = self._clean_text(context.get("location"))
        target_role = self._clean_text(context.get("target_role"))
        base_queries: List[str] = []

        def add(query: str) -> None:
            cleaned = self._clean_text(query)
            if cleaned and cleaned not in base_queries:
                base_queries.append(cleaned)

        if industry and location:
            add(f"{industry} companies in {location}")
            add(f"{industry} startups in {location}")
            add(f"{location} {industry} companies official website")
            add(f"{location} {industry} founder CEO contact")
            add(f"site:.com {industry} {location}")
            add(f"site:.in {industry} {location}")
        if target_role and industry:
            add(f"{industry} {target_role} {location}")
        if message.strip():
            add(message.strip())
        return base_queries[:6] or [message.strip() or "B2B companies"]

    def _build_directory_queries(self, message: str, context: Dict[str, Any]) -> List[str]:
        return build_targeted_directory_queries(
            industry=self._clean_text(context.get("industry")),
            location=self._clean_text(context.get("location")),
            seed_query=message,
            max_queries=6,
        ) or [self._clean_text(message) or "business directory"]

    @classmethod
    def _build_domain_retry_query(cls, directory_url: str, context: Dict[str, Any]) -> str:
        parsed = urlparse(directory_url or "")
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        industry = cls._clean_text(context.get("industry"))
        location = cls._clean_text(context.get("location"))
        terms = [f"site:{host}"] if host else []
        if industry:
            terms.append(f'"{industry}"')
        if location:
            terms.append(f'"{location}"')
        terms.extend(['"member directory"', '"directory"', '"members"'])
        return " ".join(terms).strip()

    def _parse_directory_list(self, message: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        directories = self._safe_json_list(message)
        if directories:
            return directories
        nested = context.get("directories")
        return self._safe_json_list(nested)

    def _parse_lead_list(self, message: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        leads = self._safe_json_list(message)
        if leads:
            return leads
        nested = context.get("leads")
        return self._safe_json_list(nested)

    async def _retry_directory_listing_search(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        step: int,
        directory: Dict[str, Any],
        context: Dict[str, Any],
        resources: Dict[str, Any],
        policy: ToolPolicy,
        step_logs: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        directory_url = self._clean_text(directory.get("directory_url") or directory.get("url"))
        retry_query = self._build_domain_retry_query(directory_url, context)
        if not retry_query:
            return None, step

        step += 1
        result_dict = await self._execute_manual_tool(
            session_id=session_id,
            trace_id=trace_id,
            agent_id=spec.agent_id,
            step=step,
            stage="directory_retry_search",
            tool_name="web_search",
            arguments={
                "query": retry_query,
                "industry": context.get("industry", ""),
                "location": context.get("location", ""),
                "expand_queries": False,
                "max_results": 4,
                "max_results_per_query": 4,
                "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                "include_raw_content": False,
            },
            context=context,
            resources=resources,
            policy=policy,
            step_logs=step_logs,
            tool_history=tool_history,
        )

        for item in ((result_dict.get("data") or {}).get("results") or []):
            result_url = self._clean_text(item.get("url"))
            if not is_whitelisted_directory_url(result_url):
                continue
            retried_directory = self._directory_from_search_result(item, context)
            if retried_directory:
                self._append_source_detail(
                    retried_directory,
                    stage="directory_retry_search",
                    source_type="retry_search_result",
                    provider="web_search",
                    url=result_url,
                    title=self._clean_text(item.get("title"))[:180],
                    query=retry_query,
                )
                return retried_directory, step
        return None, step

    @classmethod
    def _is_noisy_domain(cls, url: str) -> bool:
        domain = cls._extract_domain(url)
        return any(domain == item or domain.endswith(f".{item}") for item in cls._noisy_domains)

    @classmethod
    def _looks_like_listing_result(cls, title: str, snippet: str, url: str) -> bool:
        title_l = cls._clean_text(title).lower()
        snippet_l = cls._clean_text(snippet).lower()
        if cls._is_noisy_domain(url):
            return True
        if any(marker in title_l for marker in cls._noisy_title_markers):
            return True
        if title_l[:4].isdigit() and any(marker in title_l for marker in ("companies", "startups", "funded")):
            return True
        if sum(1 for marker in cls._bad_company_name_markers if marker in title_l) >= 2:
            return True
        if "including" in snippet_l and "," in snippet_l and any(marker in snippet_l for marker in ("companies", "startups")):
            return True
        return False

    @classmethod
    def _clean_company_name_from_title(cls, title: str, url: str) -> str:
        company_name = cls._normalize_company_name(title, url)
        if cls._looks_like_listing_result(company_name, title, url):
            return ""
        return company_name

    @classmethod
    def _parse_title_from_html(cls, html: str) -> str:
        match = cls._title_pattern.search(html or "")
        return cls._clean_text(match.group(1)) if match else ""

    @classmethod
    def _parse_meta_description(cls, html: str) -> str:
        match = cls._meta_description_pattern.search(html or "")
        return cls._clean_text(match.group(1)) if match else ""

    @staticmethod
    def _strip_html(value: str) -> str:
        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", value or "")
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _extract_anchor_pairs(cls, html: str) -> List[Tuple[str, str]]:
        anchors: List[Tuple[str, str]] = []
        for href, label in cls._anchor_pattern.findall(html or ""):
            cleaned_href = cls._clean_text(href)
            cleaned_label = cls._strip_html(label)[:160]
            if cleaned_href:
                anchors.append((cleaned_href, cleaned_label))
        return anchors

    @classmethod
    def _is_blocked_page(cls, html: str, status_code: int = 0) -> bool:
        if status_code in {401, 403, 405, 429}:
            return True
        text = cls._clean_text(html).lower()
        if not text:
            return False
        return any(marker in text for marker in cls._blocked_page_markers)

    @classmethod
    def _classify_directory_type(cls, title: str, snippet: str, url: str) -> str:
        haystack = " ".join([cls._clean_text(title).lower(), cls._clean_text(snippet).lower(), url.lower()])
        if any(marker in haystack for marker in ("chamber", "chamber of commerce")):
            return "chamber_member_directory"
        if any(marker in haystack for marker in ("association", "member directory", "members directory", "membership")):
            return "association_member_directory"
        if any(marker in haystack for marker in ("partner directory", "vendors", "suppliers", "contractors", "providers")):
            return "vendor_partner_directory"
        if any(marker in haystack for marker in ("business directory", "local businesses", "nearby businesses")):
            return "local_business_directory"
        if any(marker in haystack for marker in ("top ", "best ", "list of ", "companies in ")) and "directory" not in haystack:
            return "listicle_roundup"
        if any(marker in haystack for marker in ("listing", "companies", "businesses", "directory")):
            return "company_listing_directory"
        return "general_listing_page"

    @classmethod
    def _extract_heading_text(cls, html: str) -> str:
        match = cls._heading_pattern.search(html or "")
        if match:
            return cls._strip_html(match.group(1))
        return ""

    @classmethod
    def _collect_company_candidates_from_block(
        cls,
        *,
        page_url: str,
        block_html: str,
        directory_url: str,
        directory_domain: str,
        page_domain: str,
        context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        companies: List[Dict[str, Any]] = []
        detail_pages: List[str] = []
        seen_websites = set()
        seen_details = set()
        block_heading = cls._extract_heading_text(block_html)
        anchors = cls._extract_anchor_pairs(block_html)

        website_label_match = cls._website_label_pattern.search(block_html or "")
        if website_label_match:
          raw_href = cls._clean_text(website_label_match.group(1))
          absolute = urljoin(page_url, raw_href)
          if cls._is_external_company_candidate_url(absolute, directory_domain):
              website = cls._normalize_website(absolute)
              company_name = block_heading or cls._normalize_company_name("", website)
              if website and cls._looks_like_company_name_text(company_name):
                  companies.append(
                      {
                          "company_name": company_name,
                          "company_website": website,
                          "industry": cls._clean_text(context.get("industry")),
                          "location": cls._clean_text(context.get("location")),
                          "value_proposition": f"Deterministically extracted from directory block on {directory_url}",
                          "company_size": "",
                          "source": directory_url,
                          "source_details": [
                              {
                                  "stage": "directory_company_extraction",
                                  "type": "deterministic_directory_block",
                                  "url": directory_url,
                                  "value": company_name,
                              },
                              {
                                  "stage": "directory_company_extraction",
                                  "type": "company_website",
                                  "url": website,
                                  "value": company_name,
                              },
                          ],
                      }
                  )
                  seen_websites.add(website)

        for raw_href, label in anchors:
            if raw_href.startswith(cls._ignore_anchor_prefixes):
                continue
            absolute = urljoin(page_url, raw_href)
            if cls._is_external_company_candidate_url(absolute, directory_domain):
                website = cls._normalize_website(absolute)
                if not website or website in seen_websites:
                    continue
                company_name = cls._clean_text(label) or block_heading or cls._normalize_company_name("", website)
                if not cls._looks_like_company_name_text(company_name):
                    continue
                companies.append(
                    {
                        "company_name": company_name,
                        "company_website": website,
                        "industry": cls._clean_text(context.get("industry")),
                        "location": cls._clean_text(context.get("location")),
                        "value_proposition": f"Deterministically extracted from directory block on {directory_url}",
                        "company_size": "",
                        "source": directory_url,
                        "source_details": [
                            {
                                "stage": "directory_company_extraction",
                                "type": "deterministic_directory_block",
                                "url": directory_url,
                                "value": company_name,
                            },
                            {
                                "stage": "directory_company_extraction",
                                "type": "company_website",
                                "url": website,
                                "value": company_name,
                            },
                        ],
                    }
                )
                seen_websites.add(website)
                continue

            absolute_domain = cls._extract_domain(absolute)
            path_l = urlparse(absolute).path.lower()
            if absolute_domain != page_domain:
                continue
            if not any(marker in path_l for marker in cls._internal_listing_markers):
                continue
            normalized_detail = absolute.rstrip("/")
            if normalized_detail in seen_details:
                continue
            seen_details.add(normalized_detail)
            detail_pages.append(normalized_detail)

        return companies, detail_pages

    @classmethod
    def _structured_company_candidates_from_html(
        cls,
        *,
        page_url: str,
        html: str,
        directory_url: str,
        context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        companies: List[Dict[str, Any]] = []
        detail_pages: List[str] = []
        seen_websites = set()
        seen_details = set()
        directory_domain = cls._extract_domain(directory_url)
        page_domain = cls._extract_domain(page_url)

        def add_company(name: str, website_url: str) -> None:
            website = cls._normalize_website(website_url)
            company_name = cls._clean_text(name) or cls._normalize_company_name("", website)
            if not website or website in seen_websites or not cls._looks_like_company_name_text(company_name):
                return
            companies.append(
                {
                    "company_name": company_name,
                    "company_website": website,
                    "industry": cls._clean_text(context.get("industry")),
                    "location": cls._clean_text(context.get("location")),
                    "value_proposition": f"Extracted from structured data on {directory_url}",
                    "company_size": "",
                    "source": directory_url,
                    "source_details": [
                        {
                            "stage": "directory_company_extraction",
                            "type": "structured_data",
                            "url": directory_url,
                            "value": company_name,
                        },
                        {
                            "stage": "directory_company_extraction",
                            "type": "company_website",
                            "url": website,
                            "value": company_name,
                        },
                    ],
                }
            )
            seen_websites.add(website)

        def add_detail(url: str) -> None:
            normalized = cls._clean_text(url).rstrip("/")
            if not normalized or normalized in seen_details:
                return
            seen_details.add(normalized)
            detail_pages.append(normalized)

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                name = cls._clean_text(
                    value.get("name")
                    or value.get("legalName")
                    or value.get("alternateName")
                    or value.get("title")
                )
                url_value = value.get("url")
                if isinstance(url_value, str):
                    cleaned_url = cls._clean_text(url_value)
                    absolute = urljoin(page_url, cleaned_url)
                    if cls._is_external_company_candidate_url(absolute, directory_domain):
                        add_company(name, absolute)
                    elif cls._extract_domain(absolute) == page_domain:
                        path_l = urlparse(absolute).path.lower()
                        if any(marker in path_l for marker in cls._internal_listing_markers):
                            add_detail(absolute)
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for match in re.finditer(
            r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html or "",
        ):
            raw_json = cls._clean_text(match.group(1))
            if not raw_json:
                continue
            try:
                parsed = json.loads(raw_json)
            except Exception:
                continue
            walk(parsed)

        return companies[:20], detail_pages[:8]

    @classmethod
    def _looks_like_directory_result(cls, title: str, snippet: str, url: str) -> bool:
        if not url:
            return False
        title_l = cls._clean_text(title).lower()
        snippet_l = cls._clean_text(snippet).lower()
        url_l = url.lower()
        url_domain = cls._extract_domain(url)
        if any(url_domain == domain or url_domain.endswith(f".{domain}") for domain in cls._social_domains):
            return False
        markers_found = 0
        for marker in cls._directory_title_markers:
            if marker in title_l or marker in snippet_l or marker in url_l:
                markers_found += 1
        return markers_found >= 1

    @classmethod
    def _directory_result_matches_context(cls, item: Dict[str, Any], context: Dict[str, Any]) -> bool:
        industry = cls._clean_text(context.get("industry")).lower()
        location = cls._clean_text(context.get("location")).lower()
        title = cls._clean_text(item.get("title")).lower()
        snippet = cls._clean_text(item.get("content")).lower()
        url = cls._clean_text(item.get("url")).lower()
        evidence = " ".join(part for part in (title, snippet, url) if part)

        industry_tokens = cls._keyword_tokens(industry)
        location_tokens = cls._keyword_tokens(location, min_length=2)

        industry_match_count = sum(1 for token in industry_tokens if token in evidence)
        location_match_count = sum(1 for token in location_tokens if token in evidence)

        if industry_tokens:
            exact_industry_phrase = industry and industry in evidence
            required_industry_matches = min(2, len(industry_tokens))
            if not exact_industry_phrase and industry_match_count < required_industry_matches:
                return False

        if location_tokens and location_match_count == 0:
            return False

        return True

    @classmethod
    def _looks_like_company_name_text(cls, value: str) -> bool:
        cleaned = cls._clean_text(value)
        if len(cleaned) < 3 or len(cleaned) > 120:
            return False
        lowered = cleaned.lower()
        if any(marker == lowered or f" {marker}" in lowered for marker in ("contact", "about", "home", "read more", "view more")):
            return False
        if sum(1 for marker in cls._bad_company_name_markers if marker in lowered) >= 2:
            return False
        return any(char.isalpha() for char in cleaned)

    @classmethod
    def _is_external_company_candidate_url(cls, url: str, directory_domain: str) -> bool:
        domain = cls._extract_domain(url)
        if not domain or domain == directory_domain:
            return False
        if any(domain == item or domain.endswith(f".{item}") for item in cls._noisy_domains | cls._social_domains):
            return False
        return url.startswith("http")

    @classmethod
    def _collect_company_candidates_from_directory_page(
        cls,
        *,
        page_url: str,
        html: str,
        directory_url: str,
        context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        companies: List[Dict[str, Any]] = []
        detail_pages: List[str] = []
        seen_websites = set()
        seen_details = set()
        directory_domain = cls._extract_domain(directory_url)
        page_domain = cls._extract_domain(page_url)

        for _, block_html in cls._directory_block_pattern.findall(html or "")[:60]:
            block_companies, block_detail_pages = cls._collect_company_candidates_from_block(
                page_url=page_url,
                block_html=block_html,
                directory_url=directory_url,
                directory_domain=directory_domain,
                page_domain=page_domain,
                context=context,
            )
            for company in block_companies:
                website = cls._normalize_website(company.get("company_website", ""))
                if not website or website in seen_websites:
                    continue
                seen_websites.add(website)
                companies.append(company)
            for detail_url in block_detail_pages:
                normalized_detail = detail_url.rstrip("/")
                if normalized_detail in seen_details:
                    continue
                seen_details.add(normalized_detail)
                detail_pages.append(normalized_detail)

        for raw_href, label in cls._extract_anchor_pairs(html):
            if raw_href.startswith(cls._ignore_anchor_prefixes):
                continue
            absolute = urljoin(page_url, raw_href)
            if cls._is_external_company_candidate_url(absolute, directory_domain):
                website = cls._normalize_website(absolute)
                if not website or website in seen_websites:
                    continue
                company_name = cls._clean_text(label) or cls._normalize_company_name("", website)
                if not cls._looks_like_company_name_text(company_name):
                    continue
                companies.append(
                    {
                        "company_name": company_name,
                        "company_website": website,
                        "industry": cls._clean_text(context.get("industry")),
                        "location": cls._clean_text(context.get("location")),
                        "value_proposition": f"Extracted from directory page {directory_url}",
                        "company_size": "",
                        "source": directory_url,
                        "source_details": [
                            {
                                "stage": "directory_company_extraction",
                                "type": "directory_link",
                                "url": directory_url,
                                "value": company_name,
                            },
                            {
                                "stage": "directory_company_extraction",
                                "type": "company_website",
                                "url": website,
                                "value": company_name,
                            },
                        ],
                    }
                )
                seen_websites.add(website)
                continue

            absolute_domain = cls._extract_domain(absolute)
            path_l = urlparse(absolute).path.lower()
            if absolute_domain != page_domain:
                continue
            if not any(marker in path_l for marker in cls._internal_listing_markers):
                continue
            normalized_detail = absolute.rstrip("/")
            if normalized_detail in seen_details:
                continue
            seen_details.add(normalized_detail)
            detail_pages.append(normalized_detail)

        structured_companies, structured_detail_pages = cls._structured_company_candidates_from_html(
            page_url=page_url,
            html=html,
            directory_url=directory_url,
            context=context,
        )
        for company in structured_companies:
            website = cls._normalize_website(company.get("company_website", ""))
            if not website or website in seen_websites:
                continue
            seen_websites.add(website)
            companies.append(company)
        for detail_url in structured_detail_pages:
            normalized_detail = detail_url.rstrip("/")
            if normalized_detail in seen_details:
                continue
            seen_details.add(normalized_detail)
            detail_pages.append(normalized_detail)

        return companies, detail_pages[:8]

    @classmethod
    def _directory_from_search_result(cls, item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = cls._clean_text(item.get("title"))
        snippet = cls._clean_text(item.get("content"))
        url = cls._clean_text(item.get("url"))
        if not cls._looks_like_directory_result(title, snippet, url):
            return None
        directory_type = cls._classify_directory_type(title, snippet, url)
        return {
            "directory_name": title[:180] or cls._normalize_company_name(title, url),
            "directory_url": url,
            "directory_type": directory_type,
            "industry": cls._clean_text(context.get("industry")),
            "location": cls._clean_text(context.get("location")),
            "source": "web_search",
            "source_details": [
                {
                    "stage": "directory_discovery",
                    "type": "search_result",
                    "provider": "web_search",
                    "url": url,
                    "title": title[:180],
                }
            ],
        }

    @classmethod
    def _best_company_name_from_page(cls, title: str, website_url: str, fallback_name: str) -> str:
        cleaned_title = cls._clean_text(title)
        if cleaned_title:
            for separator in (" | ", " - ", " — ", " :: ", ": "):
                if separator in cleaned_title:
                    cleaned_title = cleaned_title.split(separator)[0].strip()
                    break
        if cleaned_title and not cls._looks_like_listing_result(cleaned_title, "", website_url):
            return cleaned_title
        return fallback_name or cls._normalize_company_name("", website_url)

    @classmethod
    def _best_official_website_from_results(
        cls,
        *,
        company_name: str,
        current_website: str,
        results: List[Dict[str, Any]],
    ) -> str:
        company_name_l = cls._clean_text(company_name).lower()
        current_domain = cls._extract_domain(current_website)
        candidates: List[Tuple[int, str]] = []
        for item in results:
            url = cls._clean_text(item.get("url"))
            title = cls._clean_text(item.get("title"))
            snippet = cls._clean_text(item.get("content"))
            if not url or cls._looks_like_listing_result(title, snippet, url):
                continue
            domain = cls._extract_domain(url)
            score = 0
            if current_domain and domain == current_domain:
                score += 5
            if company_name_l and company_name_l in title.lower():
                score += 4
            if "official" in title.lower() or "official" in snippet.lower():
                score += 3
            if domain and company_name_l and company_name_l.replace(" ", "") in domain.replace("-", "").replace("_", ""):
                score += 2
            candidates.append((score, cls._normalize_website(url)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] if candidates else current_website

    def _company_from_search_result(self, item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = self._clean_text(item.get("url"))
        if not url:
            return None
        domain = self._extract_domain(url)
        if not domain:
            return None
        title = self._clean_text(item.get("title"))
        snippet = self._clean_text(item.get("content"))
        if self._looks_like_listing_result(title, snippet, url):
            return None
        website = self._normalize_website(url)
        company_name = self._clean_company_name_from_title(title, url)
        if not company_name or company_name.lower() in {"linkedin", "contact", "about"}:
            return None
        return {
            "company_name": company_name,
            "company_website": website,
            "industry": self._clean_text(context.get("industry")),
            "location": self._clean_text(context.get("location")),
            "value_proposition": snippet[:280] or f"{company_name} appears in public web results.",
            "company_size": "",
            "source": "web_search",
            "contact_page": "",
            "source_details": [
                {
                    "stage": "company_discovery",
                    "type": "search_result",
                    "provider": "web_search",
                    "url": url,
                    "title": title[:180],
                }
            ],
            "_domain": domain,
        }

    async def _run_manual_company_discovery(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(
            allowed_tools=set(spec.allowed_tools),
            max_tool_calls=spec.max_tool_calls,
        )
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        companies: List[Dict[str, Any]] = []
        seen_domains = set()
        target_company_pool = max(10, int(context.get("target_company_pool") or 40))
        step = 0

        for query in self._build_manual_discovery_queries(message, context):
            if len(companies) >= target_company_pool or step >= spec.max_tool_calls:
                break
            if time.perf_counter() - run_started > spec.max_runtime_seconds:
                step_logs.append({"step": step + 1, "stage": "discovery", "type": "timeout"})
                break

            step += 1
            result_dict = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=spec.agent_id,
                step=step,
                stage="discovery_search",
                tool_name="web_search",
                arguments={
                    "query": query,
                    "industry": context.get("industry", ""),
                    "location": context.get("location", ""),
                    "expand_queries": False,
                    "max_results": clamp_web_search_max_results(max(6, target_company_pool // 4)),
                    "max_results_per_query": 4,
                    "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                    "include_raw_content": False,
                },
                context=context,
                resources=resources,
                policy=policy,
                step_logs=step_logs,
                tool_history=tool_history,
            )
            results = ((result_dict.get("data") or {}).get("results")) or []
            for item in results:
                company = self._company_from_search_result(item, context)
                if not company:
                    continue
                domain = company.pop("_domain", "")
                if domain in seen_domains:
                    continue
                seen_domains.add(domain)
                companies.append(company)
                if len(companies) >= target_company_pool:
                    break

        companies.sort(
            key=lambda item: (
                1 if item.get("company_website") else 0,
                len(item.get("value_proposition") or ""),
            ),
            reverse=True,
        )
        raw_candidates = companies[: max(target_company_pool * 2, 20)]
        companies = []

        for candidate in raw_candidates:
            if len(companies) >= target_company_pool or step >= spec.max_tool_calls:
                break
            if time.perf_counter() - run_started > spec.max_runtime_seconds:
                step_logs.append({"step": step + 1, "stage": "discovery_validation", "type": "timeout"})
                break

            company_name = candidate.get("company_name", "")
            website_url = candidate.get("company_website", "")

            step += 1
            website_result = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=spec.agent_id,
                step=step,
                stage="official_website_search",
                tool_name="web_search",
                arguments={
                    "query": f"\"{company_name}\" official website",
                    "company_name": company_name,
                    "industry": context.get("industry", ""),
                    "location": context.get("location", ""),
                    "expand_queries": False,
                    "max_results": 4,
                    "max_results_per_query": 4,
                    "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                    "include_raw_content": False,
                },
                context=context,
                resources=resources,
                policy=policy,
                step_logs=step_logs,
                tool_history=tool_history,
            )
            website_url = self._best_official_website_from_results(
                company_name=company_name,
                current_website=website_url,
                results=((website_result.get("data") or {}).get("results") or []),
            )
            if not website_url or self._is_noisy_domain(website_url):
                continue
            candidate["company_website"] = website_url

            if step >= spec.max_tool_calls:
                companies.append(candidate)
                break

            step += 1
            page_result = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=spec.agent_id,
                step=step,
                stage="company_homepage_fetch",
                tool_name="http_request",
                arguments={"method": "GET", "url": website_url, "timeout_seconds": 12},
                context=context,
                resources=resources,
                policy=policy,
                step_logs=step_logs,
                tool_history=tool_history,
            )
            page_html = self._result_text(page_result)
            title = self._parse_title_from_html(page_html)
            meta_description = self._parse_meta_description(page_html)
            refined_name = self._best_company_name_from_page(title, website_url, company_name)
            if not refined_name or self._looks_like_listing_result(refined_name, meta_description, website_url):
                continue
            candidate["company_name"] = refined_name
            if meta_description:
                candidate["value_proposition"] = meta_description[:280]
            contact_links = self._extract_contact_links(website_url, page_html)
            if contact_links:
                candidate["contact_page"] = contact_links[0]
            companies.append(candidate)

        final_answer = json.dumps({"companies": companies}, ensure_ascii=False)
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"companies": companies},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    async def _run_manual_directory_discovery(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(allowed_tools=set(spec.allowed_tools), max_tool_calls=spec.max_tool_calls)
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        directories: List[Dict[str, Any]] = []
        seen_urls = set()
        drop_reasons: Dict[str, int] = {}
        step = 0
        target_count = max(5, int(context.get("target_directory_count") or 12))
        generated_queries = self._build_directory_queries(message, context)

        for query in generated_queries:
            if len(directories) >= target_count or step >= spec.max_tool_calls:
                break
            step += 1
            result_dict = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=spec.agent_id,
                step=step,
                stage="directory_search",
                tool_name="web_search",
                arguments={
                    "query": query,
                    "industry": context.get("industry", ""),
                    "location": context.get("location", ""),
                    "expand_queries": False,
                    "max_results": clamp_web_search_max_results(10),
                    "max_results_per_query": 4,
                    "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                    "include_raw_content": False,
                },
                context=context,
                resources=resources,
                policy=policy,
                step_logs=step_logs,
                tool_history=tool_history,
            )
            for item in ((result_dict.get("data") or {}).get("results") or []):
                result_url = self._clean_text(item.get("url"))
                if not is_whitelisted_directory_url(result_url):
                    self._increment_counter(
                        drop_reasons,
                        "bad_directory_path" if looks_like_bad_directory_path(result_url) else "not_whitelisted_directory_domain",
                    )
                    continue
                if not self._directory_result_matches_context(item, context):
                    self._increment_counter(drop_reasons, "directory_context_mismatch")
                    continue
                directory = self._directory_from_search_result(item, context)
                if not directory:
                    self._increment_counter(drop_reasons, "not_directory_like")
                    continue
                directory_url = directory.get("directory_url", "").rstrip("/")
                if not directory_url or directory_url in seen_urls:
                    self._increment_counter(drop_reasons, "duplicate_directory_url" if directory_url else "missing_directory_url")
                    continue
                seen_urls.add(directory_url)
                directories.append(directory)
                if len(directories) >= target_count:
                    break

        directories = self._finalize_directory_records(directories, context, target_count)
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"directories": directories},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "directories_found": len(directories),
                "generated_queries": generated_queries,
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    async def _run_manual_directory_company_extraction(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(allowed_tools=set(spec.allowed_tools), max_tool_calls=spec.max_tool_calls)
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        companies: List[Dict[str, Any]] = []
        seen_domains = set()
        drop_reasons: Dict[str, int] = {}
        directory_diagnostics: List[Dict[str, Any]] = []
        step = 0
        target_company_pool = max(12, int(context.get("target_company_pool") or 40))

        parsed_directories = self._finalize_directory_records(
            self._parse_directory_list(message, context),
            context,
            10,
        )

        for directory in parsed_directories:
            if len(companies) >= target_company_pool or step >= spec.max_tool_calls:
                break
            directory_url = self._clean_text(directory.get("directory_url") or directory.get("url"))
            if not directory_url:
                self._increment_counter(drop_reasons, "missing_directory_url")
                continue
            diagnostic = {
                "directory_url": directory_url,
                "directory_type": self._clean_text(directory.get("directory_type")) or self._classify_directory_type(
                    self._clean_text(directory.get("directory_name") or directory.get("name")),
                    "",
                    directory_url,
                ),
                "fetch_status": 0,
                "blocked_page": False,
                "is_blocked_page": False,
                "page_companies": 0,
                "detail_pages_found": 0,
                "detail_pages_fetched": 0,
                "detail_page_companies": 0,
                "extracted_companies": 0,
                "drop_reason": "",
            }

            step += 1
            page_result = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=spec.agent_id,
                step=step,
                stage="directory_page_fetch",
                tool_name="http_request",
                arguments={"method": "GET", "url": directory_url, "timeout_seconds": 15},
                context=context,
                resources=resources,
                policy=policy,
                step_logs=step_logs,
                tool_history=tool_history,
            )
            page_html = self._result_text(page_result)
            page_data = page_result.get("data") or {}
            status_code = int(page_data.get("status_code", 0) or 0)
            diagnostic["fetch_status"] = status_code
            page_url = self._clean_text(((page_result.get("data") or {}).get("url")) or directory_url)
            if looks_like_bad_directory_path(page_url):
                retried_directory, step = await self._retry_directory_listing_search(
                    spec=spec,
                    session_id=session_id,
                    trace_id=trace_id,
                    step=step,
                    directory=directory,
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                if retried_directory:
                    directory_url = self._clean_text(retried_directory.get("directory_url") or directory_url)
                    diagnostic["directory_url"] = directory_url
                    diagnostic["directory_type"] = self._clean_text(retried_directory.get("directory_type")) or diagnostic["directory_type"]
                    step += 1
                    page_result = await self._execute_manual_tool(
                        session_id=session_id,
                        trace_id=trace_id,
                        agent_id=spec.agent_id,
                        step=step,
                        stage="directory_retry_fetch",
                        tool_name="http_request",
                        arguments={"method": "GET", "url": directory_url, "timeout_seconds": 15},
                        context=context,
                        resources=resources,
                        policy=policy,
                        step_logs=step_logs,
                        tool_history=tool_history,
                    )
                    page_html = self._result_text(page_result)
                    page_data = page_result.get("data") or {}
                    status_code = int(page_data.get("status_code", 0) or 0)
                    diagnostic["fetch_status"] = status_code
                    page_url = self._clean_text(((page_result.get("data") or {}).get("url")) or directory_url)
                else:
                    diagnostic["drop_reason"] = "not_directory_listing_page"
                    self._increment_counter(drop_reasons, "not_directory_listing_page")
                    directory_diagnostics.append(diagnostic)
                    continue
            if not page_html.strip():
                diagnostic["drop_reason"] = "empty_directory_page"
                self._increment_counter(drop_reasons, "empty_directory_page")
                directory_diagnostics.append(diagnostic)
                continue
            if self._is_blocked_page(page_html, status_code):
                diagnostic["blocked_page"] = True
                diagnostic["is_blocked_page"] = True
                diagnostic["drop_reason"] = "blocked_directory_page"
                self._increment_counter(drop_reasons, "blocked_directory_page")
                directory_diagnostics.append(diagnostic)
                continue
            page_companies, detail_pages = self._collect_company_candidates_from_directory_page(
                page_url=page_url,
                html=page_html,
                directory_url=directory_url,
                context=context,
            )
            diagnostic["page_companies"] = len(page_companies)
            diagnostic["detail_pages_found"] = len(detail_pages)
            extracted_before_directory = len(companies)
            for candidate in page_companies:
                domain = self._extract_domain(candidate.get("company_website", ""))
                if not domain or domain in seen_domains:
                    self._increment_counter(drop_reasons, "duplicate_or_missing_company_domain")
                    continue
                seen_domains.add(domain)
                companies.append(candidate)
                if len(companies) >= target_company_pool:
                    break

            for detail_url in detail_pages[:4]:
                if len(companies) >= target_company_pool or step >= spec.max_tool_calls:
                    break
                step += 1
                diagnostic["detail_pages_fetched"] += 1
                detail_result = await self._execute_manual_tool(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step=step,
                    stage="directory_detail_fetch",
                    tool_name="http_request",
                    arguments={"method": "GET", "url": detail_url, "timeout_seconds": 15},
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                detail_html = self._result_text(detail_result)
                detail_data = detail_result.get("data") or {}
                detail_status = int(detail_data.get("status_code", 0) or 0)
                if not detail_html.strip():
                    self._increment_counter(drop_reasons, "empty_directory_detail_page")
                    continue
                if self._is_blocked_page(detail_html, detail_status):
                    diagnostic["is_blocked_page"] = True
                    self._increment_counter(drop_reasons, "blocked_directory_detail_page")
                    continue
                detail_page_url = self._clean_text(((detail_result.get("data") or {}).get("url")) or detail_url)
                detail_companies, _ = self._collect_company_candidates_from_directory_page(
                    page_url=detail_page_url,
                    html=detail_html,
                    directory_url=directory_url,
                    context=context,
                )
                diagnostic["detail_page_companies"] += len(detail_companies)
                for candidate in detail_companies:
                    domain = self._extract_domain(candidate.get("company_website", ""))
                    if not domain or domain in seen_domains:
                        self._increment_counter(drop_reasons, "duplicate_or_missing_company_domain")
                        continue
                    seen_domains.add(domain)
                    title = self._parse_title_from_html(detail_html)
                    if title and candidate.get("company_name") and len(candidate["company_name"]) < 4:
                        candidate["company_name"] = self._normalize_company_name(title, candidate.get("company_website", ""))
                    candidate["source"] = directory_url
                    companies.append(candidate)
                    if len(companies) >= target_company_pool:
                        break

            diagnostic["extracted_companies"] = len(companies) - extracted_before_directory
            if diagnostic["extracted_companies"] == 0 and not diagnostic["drop_reason"]:
                if diagnostic["detail_pages_found"] == 0 and diagnostic["page_companies"] == 0:
                    diagnostic["drop_reason"] = "no_company_candidates_found"
                elif diagnostic["detail_pages_fetched"] > 0 and diagnostic["detail_page_companies"] == 0:
                    diagnostic["drop_reason"] = "detail_pages_no_companies"
                else:
                    diagnostic["drop_reason"] = "directory_candidates_filtered_out"
                self._increment_counter(drop_reasons, diagnostic["drop_reason"])
            directory_diagnostics.append(diagnostic)
            self._log_step(
                trace_id,
                "directory_extraction_diagnostic",
                {
                    "session_id": session_id,
                    "agent_id": spec.agent_id,
                    "step": step,
                    "tool_name": "http_request",
                    "directory_url": directory_url,
                    "directory_type": diagnostic["directory_type"],
                    "fetch_status": diagnostic["fetch_status"],
                    "blocked_page": diagnostic["blocked_page"],
                    "is_blocked_page": diagnostic["is_blocked_page"],
                    "page_companies": diagnostic["page_companies"],
                    "detail_pages_found": diagnostic["detail_pages_found"],
                    "detail_pages_fetched": diagnostic["detail_pages_fetched"],
                    "detail_page_companies": diagnostic["detail_page_companies"],
                    "extracted_companies": diagnostic["extracted_companies"],
                    "drop_reason": diagnostic["drop_reason"],
                },
            )

        companies = self._finalize_company_records(companies, context, target_company_pool)
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"companies": companies},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "companies_extracted": len(companies),
                "directory_diagnostics": directory_diagnostics,
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    async def _run_manual_company_validation(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(allowed_tools=set(spec.allowed_tools), max_tool_calls=spec.max_tool_calls)
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        companies: List[Dict[str, Any]] = []
        drop_reasons: Dict[str, int] = {}
        validation_diagnostics: List[Dict[str, Any]] = []
        target_company_pool = max(10, int(context.get("target_company_pool") or 40))
        candidates = self._parse_company_list_for_enrichment(message, context)[: max(target_company_pool * 2, 20)]
        validation_results = await self._validate_company_candidates_batch(candidates, context)

        for index, result in enumerate(validation_results, start=1):
            step_logs.append(
                {
                    "step": index,
                    "stage": "company_validation",
                    "type": "direct_http_validation",
                    "tool_name": "direct_http_validation",
                    "result": {
                        "company_name": result.get("company_name", ""),
                        "requested_url": result.get("requested_url", ""),
                        "final_url": result.get("final_url", ""),
                        "status_code": int(result.get("status_code", 0) or 0),
                        "ok": bool(result.get("ok", False)),
                        "reason": result.get("reason", ""),
                    },
                }
            )
            validation_diagnostics.append(
                {
                    "company_name": result.get("company_name", ""),
                    "requested_url": result.get("requested_url", ""),
                    "final_url": result.get("final_url", ""),
                    "status_code": int(result.get("status_code", 0) or 0),
                    "ok": bool(result.get("ok", False)),
                    "reason": result.get("reason", ""),
                    "contact_page": result.get("contact_page", ""),
                    "response_bytes": int(result.get("response_bytes", 0) or 0),
                }
            )
            self._log_step(
                trace_id,
                "company_validation_result",
                {
                    "session_id": session_id,
                    "agent_id": spec.agent_id,
                    "step": index,
                    "tool_name": "direct_http_validation",
                    "company_name": result.get("company_name", ""),
                    "requested_url": result.get("requested_url", ""),
                    "final_url": result.get("final_url", ""),
                    "status_code": int(result.get("status_code", 0) or 0),
                    "ok": bool(result.get("ok", False)),
                    "reason": result.get("reason", ""),
                },
            )
            if not result.get("ok"):
                self._increment_counter(drop_reasons, result.get("reason", "validation_failed"))
                continue
            company = result.get("company")
            if isinstance(company, dict):
                companies.append(company)
            if len(companies) >= target_company_pool:
                break

        companies = self._finalize_company_records(companies, context, target_company_pool)
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"companies": companies},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "valid_websites": len(companies),
                "validated_companies": len(companies),
                "validation_diagnostics": validation_diagnostics,
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    def _parse_company_list_for_enrichment(self, message: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        companies = self._safe_json_list(message)
        if companies:
            return companies
        nested = context.get("companies")
        return self._safe_json_list(nested)

    @classmethod
    def _is_usable_company_candidate(cls, company: Dict[str, Any]) -> bool:
        company_name = cls._clean_text(company.get("company_name"))
        website_url = cls._clean_text(company.get("company_website"))
        if not company_name or not website_url:
            return False
        if cls._looks_like_listing_result(company_name, company.get("value_proposition", ""), website_url):
            return False
        if cls._is_noisy_domain(website_url):
            return False
        return True

    async def _fetch_website_contact_fallback(
        self,
        *,
        session_id: str,
        trace_id: str,
        agent_id: str,
        step_start: int,
        company_name: str,
        website_url: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        policy: ToolPolicy,
        step_logs: List[Dict[str, Any]],
        tool_history: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        if not website_url:
            return {}, step_start
        collected_text = ""
        pages = [website_url]
        next_step = step_start

        for url in pages[:3]:
            next_step += 1
            result_dict = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=agent_id,
                step=next_step,
                stage="website_contact_fetch",
                tool_name="http_request",
                arguments={"method": "GET", "url": url, "timeout_seconds": 12},
                context=context,
                resources=resources,
                policy=policy,
                step_logs=step_logs,
                tool_history=tool_history,
            )
            page_text = self._result_text(result_dict)
            collected_text += "\n" + page_text
            for link in self._extract_contact_links(url, page_text):
                if link not in pages:
                    pages.append(link)
            if len(pages) >= 3:
                break

        emails = self._extract_emails(collected_text)
        phones = self._extract_phones(collected_text)
        contact_page = next((page for page in pages if any(tag in page.lower() for tag in ("contact", "about", "team", "leadership"))), "")
        return {
            "company_name": company_name,
            "contact_email": emails[0] if emails else "",
            "contact_phone": phones[0] if phones else "",
            "contact_page": contact_page,
            "company_website": website_url,
            "source": "website",
            "source_details": [
                {"stage": "website_contact_enrichment", "type": "website", "url": website_url, "value": company_name},
                *[
                    {"stage": "website_contact_enrichment", "type": "page_scanned", "url": page}
                    for page in pages[:3]
                ],
                *(
                    [{"stage": "website_contact_enrichment", "type": "email", "value": emails[0], "url": contact_page or website_url}]
                    if emails
                    else []
                ),
                *(
                    [{"stage": "website_contact_enrichment", "type": "phone", "value": phones[0], "url": contact_page or website_url}]
                    if phones
                    else []
                ),
                *(
                    [{"stage": "website_contact_enrichment", "type": "contact_page", "url": contact_page}]
                    if contact_page
                    else []
                ),
            ],
        }, next_step

    async def _run_manual_contact_enrichment(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(
            allowed_tools=set(spec.allowed_tools),
            max_tool_calls=spec.max_tool_calls,
        )
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        companies = self._parse_company_list_for_enrichment(message, context)
        target_lead_count = max(1, int(context.get("target_lead_count") or 15))
        target_roles = [self._clean_text(context.get("target_role"))] if self._clean_text(context.get("target_role")) else ["Founder", "CEO", "Owner"]
        leads: List[Dict[str, Any]] = []
        drop_reasons: Dict[str, int] = {}
        step = 0
        enrichment_cap = min(len(companies), min(20, max(10, target_lead_count)))
        capped_companies = [
            company
            for company in companies[: max(target_lead_count * 3, target_lead_count + 8)]
            if self._is_usable_company_candidate(company)
        ][:enrichment_cap]
        if not capped_companies and companies:
            self._increment_counter(drop_reasons, "no_usable_validated_companies")

        tavily_extract_map = await self._run_tavily_extract_for_companies(
            session_id=session_id,
            trace_id=trace_id,
            agent_id=spec.agent_id,
            companies=capped_companies,
            step_logs=step_logs,
            tool_history=tool_history,
        )

        for company in capped_companies:
            if len(leads) >= target_lead_count or step >= spec.max_tool_calls:
                break
            if time.perf_counter() - run_started > spec.max_runtime_seconds:
                step_logs.append({"step": step + 1, "stage": "enrichment", "type": "timeout"})
                break

            company_name = self._first_non_empty(company.get("company_name"), company.get("name"))
            website_url = self._first_non_empty(company.get("company_website"), company.get("website"))
            location = self._first_non_empty(company.get("location"), context.get("location"))
            search_query = f"\"{company_name}\" {location} founder CEO owner LinkedIn"

            normalized_website = self._normalize_website(website_url)
            extract_record = tavily_extract_map.get(normalized_website) or {}
            extract_text = self._clean_text(extract_record.get("text"))
            extracted_emails = self._extract_emails(extract_text)
            extracted_phones = self._extract_phones(extract_text)
            extracted_linkedin_urls = self._extract_linkedin_urls(extract_text)
            extracted_contact_pages = self._extract_contact_page_candidates(website_url, extract_text)

            linkedin_url = extracted_linkedin_urls[0] if extracted_linkedin_urls else ""
            contact_page = self._first_non_empty(company.get("contact_page"), extracted_contact_pages[0] if extracted_contact_pages else "")

            lead = {
                "company_name": company_name,
                "company_website": website_url,
                "industry": self._first_non_empty(company.get("industry"), context.get("industry")),
                "location": location,
                "value_proposition": self._first_non_empty(company.get("value_proposition"), ""),
                "company_size": self._first_non_empty(company.get("company_size"), ""),
                "contact_person_name": "",
                "founder_name": "",
                "contact_person_title": "",
                "contact_email": extracted_emails[0] if extracted_emails else "",
                "contact_phone": extracted_phones[0] if extracted_phones else "",
                "linkedin_url": linkedin_url,
                "contact_page": contact_page,
                "source": self._first_non_empty(company.get("source"), "website"),
                "source_details": self._normalize_source_details(company.get("source_details")),
            }
            self._append_source_detail(
                lead,
                stage="people_contact_enrichment",
                source_type="company_seed",
                url=website_url,
                value=company_name,
            )
            if extract_text:
                self._append_source_detail(
                    lead,
                    stage="tavily_extract",
                    source_type="website_extract",
                    provider="tavily_extract",
                    url=website_url,
                    value=company_name,
                )
            if lead.get("contact_email"):
                self._append_source_detail(
                    lead,
                    stage="tavily_extract",
                    source_type="email",
                    provider="tavily_extract",
                    url=website_url,
                    value=lead["contact_email"],
                )
            if lead.get("contact_phone"):
                self._append_source_detail(
                    lead,
                    stage="tavily_extract",
                    source_type="phone",
                    provider="tavily_extract",
                    url=website_url,
                    value=lead["contact_phone"],
                )
            if lead.get("linkedin_url"):
                self._append_source_detail(
                    lead,
                    stage="tavily_extract",
                    source_type="linkedin_url",
                    provider="tavily_extract",
                    url=lead["linkedin_url"],
                    value=company_name,
                )
            if lead.get("contact_page"):
                self._append_source_detail(
                    lead,
                    stage="tavily_extract",
                    source_type="contact_page",
                    provider="tavily_extract",
                    url=lead["contact_page"],
                    value=company_name,
                )

            if (not self._has_contact_path(lead) or not lead.get("contact_email")) and website_url and step < spec.max_tool_calls:
                website_fallback, step = await self._fetch_website_contact_fallback(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step_start=step,
                    company_name=company_name,
                    website_url=website_url,
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                lead.update({
                    "contact_email": self._first_non_empty(lead.get("contact_email"), website_fallback.get("contact_email")),
                    "contact_phone": self._first_non_empty(lead.get("contact_phone"), website_fallback.get("contact_phone")),
                    "contact_page": self._first_non_empty(lead.get("contact_page"), website_fallback.get("contact_page")),
                    "source": self._first_non_empty(lead.get("source"), website_fallback.get("source"), lead.get("source")),
                })
                self._merge_source_details(lead, website_fallback.get("source_details"))

            if not lead.get("linkedin_url") and step < spec.max_tool_calls:
                step += 1
                search_result = await self._execute_manual_tool(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step=step,
                    stage="linkedin_search",
                    tool_name="web_search",
                    arguments={
                        "query": search_query,
                        "company_name": company_name,
                        "industry": context.get("industry", ""),
                        "location": location,
                        "expand_queries": False,
                        "max_results": 4,
                        "max_results_per_query": 4,
                        "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                        "include_raw_content": False,
                    },
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                for item in ((search_result.get("data") or {}).get("results") or []):
                    candidate_url = self._clean_text(item.get("url"))
                    candidate_title = self._clean_text(item.get("title"))
                    candidate_snippet = self._clean_text(item.get("content"))
                    if self._looks_like_listing_result(candidate_title, candidate_snippet, candidate_url):
                        continue
                    if "linkedin.com/in/" in candidate_url or "linkedin.com/company/" in candidate_url:
                        lead["linkedin_url"] = candidate_url
                        self._append_source_detail(
                            lead,
                            stage="linkedin_search",
                            source_type="linkedin_search_result",
                            provider="web_search",
                            url=candidate_url,
                            title=candidate_title[:180],
                            query=search_query,
                        )
                        break

            if lead.get("linkedin_url") and step < spec.max_tool_calls:
                step += 1
                linkedin_result = await self._execute_manual_tool(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step=step,
                    stage="linkedin_enrichment",
                    tool_name="linkedin_research",
                    arguments={
                        "url": lead.get("linkedin_url"),
                        "company_name": company_name,
                        "website_url": website_url,
                        "target_roles": target_roles,
                        "allow_fallback_contact_paths": True,
                    },
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                linkedin_data = linkedin_result.get("data") or {}
                lead.update({
                    "company_name": self._first_non_empty(linkedin_data.get("company_name"), lead.get("company_name")),
                    "company_website": self._first_non_empty(linkedin_data.get("company_website"), lead.get("company_website")),
                    "contact_person_name": self._first_non_empty(linkedin_data.get("contact_person_name"), linkedin_data.get("full_name")),
                    "founder_name": self._first_non_empty(linkedin_data.get("founder_name"), linkedin_data.get("full_name")),
                    "contact_person_title": self._first_non_empty(linkedin_data.get("contact_person_title"), lead.get("contact_person_title")),
                    "contact_email": self._first_non_empty(linkedin_data.get("contact_email"), lead.get("contact_email")),
                    "contact_phone": self._first_non_empty(linkedin_data.get("contact_phone"), lead.get("contact_phone")),
                    "linkedin_url": self._first_non_empty(linkedin_data.get("linkedin_url"), lead.get("linkedin_url")),
                    "contact_page": self._first_non_empty(linkedin_data.get("contact_page"), lead.get("contact_page")),
                    "source": self._first_non_empty(linkedin_data.get("source"), lead.get("source"), "linkedin"),
                    "value_proposition": self._first_non_empty(linkedin_data.get("value_proposition"), lead.get("value_proposition")),
                    "company_size": self._first_non_empty(linkedin_data.get("company_size"), lead.get("company_size")),
                })
                self._append_source_detail(
                    lead,
                    stage="linkedin_enrichment",
                    source_type="linkedin_research",
                    provider="linkedin_research",
                    url=self._first_non_empty(linkedin_data.get("linkedin_url"), lead.get("linkedin_url")),
                    value=self._first_non_empty(linkedin_data.get("contact_person_name"), linkedin_data.get("full_name")),
                )

            quality_input = score_lead(lead)
            lead.update(quality_input)
            relevance = self._lead_relevance_signals(lead, context)
            lead["relevance_signals"] = relevance
            lead["verification_status"] = "unreviewed"
            lead["verification_message"] = ""
            if not relevance["industry_match"]:
                self._increment_counter(drop_reasons, "insufficient_industry_match")
                continue
            if not relevance["location_match"]:
                self._increment_counter(drop_reasons, "insufficient_location_match")
                continue
            if lead.get("quality_score", 0) >= 45 and self._has_contact_path(lead):
                leads.append(lead)
            else:
                if not self._has_contact_path(lead):
                    self._increment_counter(drop_reasons, "no_contact_path")
                elif lead.get("quality_score", 0) < 45:
                    self._increment_counter(drop_reasons, "quality_below_threshold")
                else:
                    self._increment_counter(drop_reasons, "filtered_out")

        leads.sort(
            key=lambda item: (
                item.get("quality_score", 0),
                1 if item.get("contact_email") else 0,
                1 if item.get("contact_person_name") or item.get("founder_name") else 0,
                1 if item.get("linkedin_url") else 0,
            ),
            reverse=True,
        )
        leads = self._finalize_lead_records(leads, context, target_lead_count)

        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"leads": leads},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "validated_input_companies": len(capped_companies),
                "extract_enriched_companies": len(tavily_extract_map),
                "contact_info_count": sum(1 for lead in leads if self._has_contact_path(lead)),
                "linkedin_people_count": sum(1 for lead in leads if self._has_linkedin_person(lead)),
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    async def _run_manual_company_contact_enrichment(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(allowed_tools=set(spec.allowed_tools), max_tool_calls=spec.max_tool_calls)
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        leads: List[Dict[str, Any]] = []
        drop_reasons: Dict[str, int] = {}
        step = 0
        target_lead_count = max(1, int(context.get("target_lead_count") or 15))

        for company in self._parse_company_list_for_enrichment(message, context)[: max(target_lead_count * 3, 20)]:
            if len(leads) >= max(target_lead_count * 2, target_lead_count + 10) or step >= spec.max_tool_calls:
                break
            if not self._is_usable_company_candidate(company):
                self._increment_counter(drop_reasons, "unusable_company_candidate")
                continue

            company_name = self._first_non_empty(company.get("company_name"), company.get("name"))
            website_url = self._first_non_empty(company.get("company_website"), company.get("website"))
            lead = {
                "company_name": company_name,
                "company_website": website_url,
                "industry": self._first_non_empty(company.get("industry"), context.get("industry")),
                "location": self._first_non_empty(company.get("location"), context.get("location")),
                "value_proposition": self._first_non_empty(company.get("value_proposition")),
                "company_size": self._first_non_empty(company.get("company_size")),
                "contact_person_name": "",
                "founder_name": "",
                "contact_person_title": "",
                "contact_email": "",
                "contact_phone": "",
                "linkedin_url": self._first_non_empty(company.get("linkedin_url")),
                "contact_page": self._first_non_empty(company.get("contact_page")),
                "source": self._first_non_empty(company.get("source"), "website"),
                "source_details": self._normalize_source_details(company.get("source_details")),
            }
            self._append_source_detail(
                lead,
                stage="people_contact_enrichment",
                source_type="company_seed",
                url=website_url,
                value=company_name,
            )
            if website_url and step < spec.max_tool_calls:
                website_fallback, step = await self._fetch_website_contact_fallback(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step_start=step,
                    company_name=company_name,
                    website_url=website_url,
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                lead.update(
                    {
                        "contact_email": website_fallback.get("contact_email", ""),
                        "contact_phone": website_fallback.get("contact_phone", ""),
                        "contact_page": self._first_non_empty(lead.get("contact_page"), website_fallback.get("contact_page")),
                        "source": self._first_non_empty(lead.get("source"), website_fallback.get("source")),
                    }
                )
                self._merge_source_details(lead, website_fallback.get("source_details"))
            leads.append(lead)

        leads = self._finalize_lead_records(leads, context, max(target_lead_count * 2, target_lead_count + 10))
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"leads": leads},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "contact_info_count": sum(1 for lead in leads if self._has_contact_path(lead)),
                "linkedin_people_count": sum(1 for lead in leads if self._has_linkedin_person(lead)),
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    async def _run_manual_linkedin_enrichment(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        policy = ToolPolicy(allowed_tools=set(spec.allowed_tools), max_tool_calls=spec.max_tool_calls)
        step_logs: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        target_roles = [self._clean_text(context.get("target_role"))] if self._clean_text(context.get("target_role")) else ["Founder", "CEO", "Owner"]
        leads: List[Dict[str, Any]] = []
        drop_reasons: Dict[str, int] = {}
        step = 0

        for lead in self._parse_lead_list(message, context):
            if step >= spec.max_tool_calls:
                break
            item = dict(lead)
            item["source_details"] = self._normalize_source_details(item.get("source_details"))
            company_name = self._first_non_empty(item.get("company_name"))
            location = self._first_non_empty(item.get("location"), context.get("location"))
            website_url = self._first_non_empty(item.get("company_website"))
            linkedin_url = self._first_non_empty(item.get("linkedin_url"))

            if company_name and not linkedin_url and step < spec.max_tool_calls:
                step += 1
                search_result = await self._execute_manual_tool(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step=step,
                    stage="linkedin_search",
                    tool_name="web_search",
                    arguments={
                        "query": f"\"{company_name}\" {location} founder CEO owner LinkedIn",
                        "company_name": company_name,
                        "industry": context.get("industry", ""),
                        "location": location,
                        "expand_queries": False,
                        "max_results": 8,
                        "max_results_per_query": 4,
                        "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                        "include_raw_content": False,
                    },
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                for result in ((search_result.get("data") or {}).get("results") or []):
                    candidate_url = self._clean_text(result.get("url"))
                    if "linkedin.com/in/" in candidate_url or "linkedin.com/company/" in candidate_url:
                        linkedin_url = candidate_url
                        self._append_source_detail(
                            item,
                            stage="linkedin_search",
                            source_type="linkedin_search_result",
                            provider="web_search",
                            url=candidate_url,
                            value=company_name,
                        )
                        break
                if linkedin_url:
                    item["linkedin_url"] = linkedin_url
                else:
                    self._increment_counter(drop_reasons, "linkedin_not_found")

            if linkedin_url and step < spec.max_tool_calls:
                step += 1
                linkedin_result = await self._execute_manual_tool(
                    session_id=session_id,
                    trace_id=trace_id,
                    agent_id=spec.agent_id,
                    step=step,
                    stage="linkedin_enrichment",
                    tool_name="linkedin_research",
                    arguments={
                        "url": linkedin_url,
                        "company_name": company_name,
                        "website_url": website_url,
                        "target_roles": target_roles,
                        "allow_fallback_contact_paths": True,
                    },
                    context=context,
                    resources=resources,
                    policy=policy,
                    step_logs=step_logs,
                    tool_history=tool_history,
                )
                linkedin_data = linkedin_result.get("data") or {}
                item.update(
                    {
                        "company_name": self._first_non_empty(linkedin_data.get("company_name"), item.get("company_name")),
                        "company_website": self._first_non_empty(linkedin_data.get("company_website"), item.get("company_website")),
                        "contact_person_name": self._first_non_empty(linkedin_data.get("contact_person_name"), linkedin_data.get("full_name"), item.get("contact_person_name")),
                        "founder_name": self._first_non_empty(linkedin_data.get("founder_name"), linkedin_data.get("full_name"), item.get("founder_name")),
                        "contact_person_title": self._first_non_empty(linkedin_data.get("contact_person_title"), item.get("contact_person_title")),
                        "contact_email": self._first_non_empty(item.get("contact_email"), linkedin_data.get("contact_email")),
                        "contact_phone": self._first_non_empty(item.get("contact_phone"), linkedin_data.get("contact_phone")),
                        "linkedin_url": self._first_non_empty(linkedin_data.get("linkedin_url"), item.get("linkedin_url")),
                        "contact_page": self._first_non_empty(item.get("contact_page"), linkedin_data.get("contact_page")),
                        "source": self._first_non_empty(item.get("source"), linkedin_data.get("source"), "linkedin"),
                        "value_proposition": self._first_non_empty(item.get("value_proposition"), linkedin_data.get("value_proposition")),
                        "company_size": self._first_non_empty(item.get("company_size"), linkedin_data.get("company_size")),
                    }
                )
                self._append_source_detail(
                    item,
                    stage="linkedin_enrichment",
                    source_type="linkedin_research",
                    provider="linkedin_research",
                    url=self._first_non_empty(linkedin_data.get("linkedin_url"), item.get("linkedin_url")),
                    value=self._first_non_empty(linkedin_data.get("contact_person_name"), linkedin_data.get("full_name")),
                )
            elif not linkedin_url:
                self._increment_counter(drop_reasons, "missing_linkedin_url")
            leads.append(item)

        leads = self._finalize_lead_records(leads, context, len(leads) or max(1, int(context.get("target_lead_count") or 15)))
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"leads": leads},
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": len(tool_history),
                "completed": True,
                "manual_flow": True,
                "contact_info_count": sum(1 for lead in leads if self._has_contact_path(lead)),
                "linkedin_people_count": sum(1 for lead in leads if self._has_linkedin_person(lead)),
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    async def _run_manual_quality_scoring(
        self,
        *,
        spec: AgentSpec,
        session_id: str,
        trace_id: str,
        message: str,
        context: Dict[str, Any],
        resources: Dict[str, Any],
        run_started: float,
    ) -> Dict[str, Any]:
        leads: List[Dict[str, Any]] = []
        target_lead_count = max(1, int(context.get("target_lead_count") or 15))
        drop_reasons: Dict[str, int] = {}

        for lead in self._parse_lead_list(message, context):
            item = dict(lead)
            quality = score_lead(item)
            item.update(quality)
            relevance = self._lead_relevance_signals(item, context)
            item["relevance_signals"] = relevance
            item["verification_status"] = item.get("verification_status", "unreviewed")
            item["verification_message"] = item.get("verification_message", "")
            if not relevance["industry_match"]:
                self._increment_counter(drop_reasons, "insufficient_industry_match")
                continue
            if not relevance["location_match"]:
                self._increment_counter(drop_reasons, "insufficient_location_match")
                continue
            if item.get("quality_score", 0) >= 40 and (
                item.get("contact_email") or item.get("contact_phone") or item.get("linkedin_url") or item.get("contact_page")
            ):
                leads.append(item)
            else:
                if not self._has_contact_path(item):
                    self._increment_counter(drop_reasons, "no_contact_path")
                elif item.get("quality_score", 0) < 40:
                    self._increment_counter(drop_reasons, "quality_below_threshold")
                else:
                    self._increment_counter(drop_reasons, "filtered_out")

        leads.sort(
            key=lambda item: (
                item.get("quality_score", 0),
                1 if item.get("contact_email") else 0,
                1 if item.get("contact_person_name") or item.get("founder_name") else 0,
                1 if item.get("linkedin_url") else 0,
            ),
            reverse=True,
        )
        leads = self._finalize_lead_records(leads, context, target_lead_count)

        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "json",
            "text": None,
            "json": {"leads": leads},
            "steps": [],
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": 0,
                "completed": True,
                "manual_flow": True,
                "usable_leads": len(leads),
                "contact_info_count": sum(1 for lead in leads if self._has_contact_path(lead)),
                "linkedin_people_count": sum(1 for lead in leads if self._has_linkedin_person(lead)),
                "rows_dropped": sum(drop_reasons.values()),
                "dropped_reasons": drop_reasons,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }

    def _synthesize_leads_from_tool_history(self, tool_history: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        target_count = max(1, int(context.get("target_lead_count") or 15))
        industry = str(context.get("industry") or "").strip() or "General"
        location = str(context.get("location") or "").strip() or "Unknown"

        candidate_results: List[Dict[str, Any]] = []
        candidate_sites: List[str] = []
        for step in tool_history:
            if step.get("tool_name") == "web_search":
                result = ((step.get("result") or {}).get("output") or {})
                candidate_results.extend(result.get("results") or [])

        sites_set = {site for site in candidate_sites if site}
        leads: List[Dict[str, Any]] = []
        seen_domains = set()

        for item in candidate_results:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            parsed = urlparse(url)
            domain = (parsed.hostname or "").lower()
            if not domain:
                continue
            if domain.startswith("www."):
                domain = domain[4:]
            if domain in seen_domains:
                continue
            seen_domains.add(domain)

            website = f"{parsed.scheme or 'https'}://{parsed.netloc}" if parsed.netloc else url
            company_name = self._normalize_company_name(item.get("title", ""), url)
            snippet = (item.get("content") or "").strip()

            leads.append(
                {
                    "company_name": company_name,
                    "company_website": website,
                    "industry": industry,
                    "location": location,
                    "value_proposition": snippet[:280] or f"{company_name} appears in search results for {industry} in {location}.",
                    "company_size": "",
                    "contact_person_name": "",
                    "founder_name": "",
                    "contact_person_title": "",
                    "contact_email": "",
                    "contact_phone": "",
                    "linkedin_url": "",
                    "contact_page": website if website in sites_set else "",
                    "source": "web_search",
                    "confidence": "low",
                }
            )

            if len(leads) >= target_count:
                break

        return leads

    async def run(
        self,
        spec: AgentSpec,
        session_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
        system_prompt_text: str = "",
        developer_prompt_text: str = "",
    ) -> Dict[str, Any]:
        run_started = time.perf_counter()
        context = {**(spec.default_context or {}), **(context or {})}
        context.setdefault("agent_id", spec.agent_id)
        context["_latest_user_message"] = message
        context["_user_confirmation_signal"] = bool(self.CONFIRMATION_PATTERN.search(message or ""))
        resources = resources or {}
        trace_id = f"TRACE_{uuid.uuid4()}"
        memory_context = self.memory_store.get_context(session_id, limit=spec.memory_window)
        tool_specs = self.tool_registry.list_specs(spec.allowed_tools)
        policy = ToolPolicy(
            allowed_tools=set(spec.allowed_tools),
            max_tool_calls=spec.max_tool_calls,
        )

        if self._is_lead_generation_agent(spec):
            return await self._run_production_lead_generation(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                run_started=run_started,
            )
        if self._is_manual_discovery_agent(spec):
            return await self._run_manual_company_discovery(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_directory_discovery_agent(spec):
            return await self._run_manual_directory_discovery(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_company_extraction_agent(spec):
            return await self._run_manual_directory_company_extraction(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_company_validation_agent(spec):
            return await self._run_manual_company_validation(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_enrichment_agent(spec):
            return await self._run_manual_contact_enrichment(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_contact_stage_agent(spec):
            return await self._run_manual_company_contact_enrichment(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_linkedin_stage_agent(spec):
            return await self._run_manual_linkedin_enrichment(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )
        if self._is_manual_quality_stage_agent(spec):
            return await self._run_manual_quality_scoring(
                spec=spec,
                session_id=session_id,
                trace_id=trace_id,
                message=message,
                context=context,
                resources=resources,
                run_started=run_started,
            )

        tool_calls = 0
        tool_history: List[Dict[str, Any]] = []
        step_logs: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt(spec, system_prompt_text, developer_prompt_text)
        invalid_decision_count = 0
        llm_input_tokens = 0
        llm_output_tokens = 0

        for step in range(1, spec.max_steps + 1):
            elapsed = time.perf_counter() - run_started
            if elapsed > spec.max_runtime_seconds:
                step_logs.append({"step": step, "type": "timeout", "message": "agent runtime exceeded"})
                break

            decision_prompt = self._decision_prompt(
                user_message=message,
                context=context,
                memory_context=memory_context,
                tool_specs=tool_specs,
                tool_history=tool_history,
                step=step,
                max_steps=spec.max_steps,
            )
            self._log_step(
                trace_id,
                "llm_decision_start",
                {
                    "session_id": session_id,
                    "agent_id": spec.agent_id,
                    "step": step,
                    "prompt_preview": self._preview_text(decision_prompt, limit=500),
                    "system_prompt_preview": self._preview_text(system_prompt, limit=300),
                },
            )
            try:
                decision = await self.llm.generate_json(
                    prompt=decision_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                )
            except Exception as exc:
                self._log_step(
                    trace_id,
                    "llm_decision_error",
                    {
                        "session_id": session_id,
                        "agent_id": spec.agent_id,
                        "step": step,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            llm_usage = self._llm_usage()
            llm_input_tokens += llm_usage["input_tokens"]
            llm_output_tokens += llm_usage["output_tokens"]
            self._log_step(
                trace_id,
                "llm_decision_end",
                {
                    "session_id": session_id,
                    "agent_id": spec.agent_id,
                    "step": step,
                    "input_tokens": llm_usage["input_tokens"],
                    "output_tokens": llm_usage["output_tokens"],
                    "total_tokens": llm_usage["total_tokens"],
                    "latency_ms": llm_usage["latency_ms"],
                    "model": llm_usage["model"],
                    "mode": llm_usage["mode"],
                    "parse_status": llm_usage["parse_status"],
                    "decision_preview": self._preview_payload(decision, limit=600),
                },
            )

            decision_type = str(decision.get("type", "")).strip().lower()
            thought = decision.get("thought", "")
            self._log_step(
                trace_id,
                "agent_decision",
                {
                    "session_id": session_id,
                    "agent_id": spec.agent_id,
                    "step": step,
                    "decision_type": decision_type,
                    "thought": thought,
                },
            )

            if decision_type == "final_answer":
                invalid_decision_count = 0
                final_answer = str(decision.get("final_answer", "")).strip()
                if not final_answer:
                    final_answer = "I could not complete the task."

                json_output = None
                if spec.response_mode == "json":
                    try:
                        if spec.output_schema:
                            format_prompt = (
                                "Convert the text below into JSON matching this schema shape.\n"
                                f"Schema:\n{json.dumps(spec.output_schema, ensure_ascii=False)}\n\n"
                                f"Text:\n{final_answer}"
                            )
                            self._log_step(
                                trace_id,
                                "llm_format_start",
                                {
                                    "session_id": session_id,
                                    "agent_id": spec.agent_id,
                                    "step": step,
                                    "prompt_preview": self._preview_text(format_prompt, limit=500),
                                },
                            )
                            json_output = await self.llm.generate_json(
                                prompt=format_prompt,
                                system_prompt=system_prompt,
                                temperature=0.1,
                            )
                            llm_usage = self._llm_usage()
                            llm_input_tokens += llm_usage["input_tokens"]
                            llm_output_tokens += llm_usage["output_tokens"]
                            self._log_step(
                                trace_id,
                                "llm_format_end",
                                {
                                    "session_id": session_id,
                                    "agent_id": spec.agent_id,
                                    "step": step,
                                    "input_tokens": llm_usage["input_tokens"],
                                    "output_tokens": llm_usage["output_tokens"],
                                    "total_tokens": llm_usage["total_tokens"],
                                    "latency_ms": llm_usage["latency_ms"],
                                    "model": llm_usage["model"],
                                    "mode": llm_usage["mode"],
                                    "parse_status": llm_usage["parse_status"],
                                    "json_preview": self._preview_payload(json_output, limit=600),
                                },
                            )
                        else:
                            json_output = {"result": final_answer}
                    except Exception as e:
                        self._runtime_logger.warning(f"Final JSON formatting failed: {e}")
                        # Fallback: create a dummy object so the frontend can still use the text
                        json_output = {"error": "formatting_failed", "text_fallback": final_answer}

                self.memory_store.add_message(session_id, "user", message)
                self.memory_store.add_message(
                    session_id,
                    "assistant",
                    json.dumps(json_output, ensure_ascii=False) if json_output is not None else final_answer,
                )
                self._log_step(
                    trace_id,
                    "agent_final_answer",
                    {
                        "session_id": session_id,
                        "agent_id": spec.agent_id,
                        "step": step,
                        "tool_name": "-",
                        "decision_type": "final_answer",
                    },
                )
                return {
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "response_mode": spec.response_mode,
                    "text": final_answer if spec.response_mode == "text" else None,
                    "json": json_output,
                    "steps": step_logs,
                    "metadata": {
                        "agent_id": spec.agent_id,
                        "tool_calls": tool_calls,
                        "completed": True,
                        "input_tokens": llm_input_tokens,
                        "output_tokens": llm_output_tokens,
                        "total_tokens": llm_input_tokens + llm_output_tokens,
                        "latency_ms": (time.perf_counter() - run_started) * 1000,
                    },
                }

            if decision_type != "tool_call":
                step_logs.append({"step": step, "type": "invalid_decision", "decision": decision})
                invalid_decision_count += 1
                if (
                    self._is_lead_generation_agent(spec)
                    and tool_calls == 0
                    and invalid_decision_count >= 3
                    and "web_search" in spec.allowed_tools
                    and tool_calls < policy.max_tool_calls
                ):
                    forced_arguments = self._build_forced_search_arguments(message, context)
                    forced_tool_context = ToolContext(
                        session_id=session_id,
                        trace_id=trace_id,
                        resources=resources,
                        state=context,
                    )
                    forced_result = await self.executor.execute(
                        tool_name="web_search",
                        arguments=forced_arguments,
                        context=forced_tool_context,
                        policy=policy,
                    )
                    tool_calls += 1
                    forced_result_dict = forced_result.to_dict()
                    forced_step_log = {
                        "step": step,
                        "type": "forced_tool_call",
                        "tool_name": "web_search",
                        "arguments": forced_arguments,
                        "result": forced_result_dict,
                    }
                    step_logs.append(forced_step_log)
                    tool_history.append(forced_step_log)
                    context[f"tool_{step}_web_search"] = forced_result_dict
                    self._log_step(
                        trace_id,
                        "agent_tool_call",
                        {
                            "session_id": session_id,
                            "agent_id": spec.agent_id,
                            "step": step,
                            "tool_name": "web_search",
                            "ok": forced_result.ok,
                            "latency_ms": forced_result.latency_ms,
                            "error": forced_result.error,
                        },
                    )
                    invalid_decision_count = 0
                continue

            if tool_calls >= policy.max_tool_calls:
                step_logs.append({"step": step, "type": "max_tool_calls_reached"})
                break

            tool_name = str(decision.get("tool_name", "")).strip()
            arguments = decision.get("arguments", {}) or {}
            invalid_decision_count = 0
            tool_calls += 1
            tool_context = ToolContext(
                session_id=session_id,
                trace_id=trace_id,
                resources=resources,
                state=context,
            )
            result = await self.executor.execute(
                tool_name=tool_name,
                arguments=arguments,
                context=tool_context,
                policy=policy,
            )

            result_dict = result.to_dict()
            step_log = {
                "step": step,
                "type": "tool_call",
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result_dict,
            }
            step_logs.append(step_log)
            tool_history.append(step_log)

            self._log_step(
                trace_id,
                "agent_tool_call",
                {
                    "session_id": session_id,
                    "agent_id": spec.agent_id,
                    "step": step,
                    "tool_name": tool_name,
                    "ok": result.ok,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                },
            )

            context[f"tool_{step}_{tool_name}"] = result_dict

        if self._is_lead_generation_agent(spec):
            synthesized_leads = self._synthesize_leads_from_tool_history(tool_history, context)
            if synthesized_leads:
                json_output = {"leads": synthesized_leads}
                self.memory_store.add_message(session_id, "user", message)
                self.memory_store.add_message(session_id, "assistant", json.dumps(json_output, ensure_ascii=False))
                self._log_step(
                    trace_id,
                    "agent_final_answer",
                    {
                        "session_id": session_id,
                        "agent_id": spec.agent_id,
                        "step": spec.max_steps,
                        "tool_name": "-",
                        "decision_type": "forced_search_final",
                    },
                )
                return {
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "response_mode": "json",
                    "text": None,
                    "json": json_output,
                    "steps": step_logs,
                    "metadata": {
                        "agent_id": spec.agent_id,
                        "tool_calls": tool_calls,
                        "completed": True,
                        "forced_fallback": True,
                        "input_tokens": llm_input_tokens,
                        "output_tokens": llm_output_tokens,
                        "total_tokens": llm_input_tokens + llm_output_tokens,
                        "latency_ms": (time.perf_counter() - run_started) * 1000,
                    },
                }

        fallback = "I could not complete the task within the allowed steps."
        self.memory_store.add_message(session_id, "user", message)
        self.memory_store.add_message(session_id, "assistant", fallback)
        self._log_step(
            trace_id,
            "agent_fallback",
            {
                "session_id": session_id,
                "agent_id": spec.agent_id,
                "step": spec.max_steps,
                "tool_name": "-",
                "decision_type": "fallback",
            },
        )
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "response_mode": "text",
            "text": fallback,
            "json": None,
            "steps": step_logs,
            "metadata": {
                "agent_id": spec.agent_id,
                "tool_calls": tool_calls,
                "completed": False,
                "input_tokens": llm_input_tokens,
                "output_tokens": llm_output_tokens,
                "total_tokens": llm_input_tokens + llm_output_tokens,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }
