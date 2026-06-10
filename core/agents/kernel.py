import json
import logging
import re
import time
import uuid
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

from config.schemas import AgentSpec
from core.adapters.base import BaseLLMAdapter
from core.agents.memory import MemoryStore
from core.tools import ToolContext, ToolExecutor, ToolPolicy, ToolRegistry
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

    @staticmethod
    def _is_manual_discovery_agent(spec: AgentSpec) -> bool:
        return spec.agent_id == "lead_company_discovery_agent"

    @staticmethod
    def _is_manual_enrichment_agent(spec: AgentSpec) -> bool:
        return spec.agent_id == "lead_contact_enrichment_agent"

    @classmethod
    def _is_manual_lead_pipeline_agent(cls, spec: AgentSpec) -> bool:
        return cls._is_manual_discovery_agent(spec) or cls._is_manual_enrichment_agent(spec)

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
            "expand_queries": bool(context.get("search_expansion_enabled", True)),
            "max_results": min(max(target_count, 12), 30),
            "max_results_per_query": 6,
            "search_depth": "advanced",
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
                    companies = parsed.get("companies")
                    if isinstance(companies, list):
                        return [item for item in companies if isinstance(item, dict)]
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
                    "expand_queries": True,
                    "max_results": min(18, max(8, target_company_pool // 2)),
                    "max_results_per_query": 6,
                    "search_depth": "advanced",
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
                    "max_results": 5,
                    "max_results_per_query": 5,
                    "search_depth": "advanced",
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
        step = 0

        for company in companies[: max(target_lead_count * 3, target_lead_count + 8)]:
            if len(leads) >= target_lead_count or step >= spec.max_tool_calls:
                break
            if time.perf_counter() - run_started > spec.max_runtime_seconds:
                step_logs.append({"step": step + 1, "stage": "enrichment", "type": "timeout"})
                break

            if not self._is_usable_company_candidate(company):
                continue

            company_name = self._first_non_empty(company.get("company_name"), company.get("name"))
            website_url = self._first_non_empty(company.get("company_website"), company.get("website"))
            location = self._first_non_empty(company.get("location"), context.get("location"))
            search_query = f"\"{company_name}\" {location} founder CEO owner contact LinkedIn email"

            linkedin_url = ""
            contact_page = self._first_non_empty(company.get("contact_page"), "")
            step += 1
            search_result = await self._execute_manual_tool(
                session_id=session_id,
                trace_id=trace_id,
                agent_id=spec.agent_id,
                step=step,
                stage="enrichment_search",
                tool_name="web_search",
                arguments={
                    "query": search_query,
                    "company_name": company_name,
                    "industry": context.get("industry", ""),
                    "location": location,
                    "expand_queries": True,
                    "max_results": 10,
                    "max_results_per_query": 4,
                    "search_depth": "advanced",
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
                if "linkedin.com/in/" in candidate_url and not linkedin_url:
                    linkedin_url = candidate_url
                elif "linkedin.com/company/" in candidate_url and not linkedin_url:
                    linkedin_url = candidate_url
                elif not website_url and candidate_url and self._extract_domain(candidate_url) and "linkedin.com" not in candidate_url:
                    website_url = self._normalize_website(candidate_url)
                if any(tag in candidate_url.lower() for tag in ("contact", "about", "team", "leadership")) and not contact_page:
                    contact_page = candidate_url

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
                "contact_email": "",
                "contact_phone": "",
                "linkedin_url": linkedin_url,
                "contact_page": contact_page,
                "source": self._first_non_empty(company.get("source"), "web_search"),
            }

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
                lead.update({
                    "contact_email": self._first_non_empty(lead.get("contact_email"), website_fallback.get("contact_email")),
                    "contact_phone": self._first_non_empty(lead.get("contact_phone"), website_fallback.get("contact_phone")),
                    "contact_page": self._first_non_empty(lead.get("contact_page"), website_fallback.get("contact_page")),
                    "source": self._first_non_empty(lead.get("source"), website_fallback.get("source"), lead.get("source")),
                })

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
                    "source": self._first_non_empty(linkedin_data.get("source"), "linkedin"),
                    "value_proposition": self._first_non_empty(linkedin_data.get("value_proposition"), lead.get("value_proposition")),
                    "company_size": self._first_non_empty(linkedin_data.get("company_size"), lead.get("company_size")),
                })

            quality_input = score_lead(lead)
            lead.update(quality_input)
            lead["verification_status"] = "unreviewed"
            lead["verification_message"] = ""
            if lead.get("quality_score", 0) >= 45 and (
                lead.get("contact_email")
                or lead.get("contact_phone")
                or lead.get("linkedin_url")
                or lead.get("contact_page")
            ):
                leads.append(lead)

        leads.sort(
            key=lambda item: (
                item.get("quality_score", 0),
                1 if item.get("contact_email") else 0,
                1 if item.get("contact_person_name") or item.get("founder_name") else 0,
                1 if item.get("linkedin_url") else 0,
            ),
            reverse=True,
        )
        leads = leads[:target_lead_count]

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
            elif step.get("tool_name") == "web_research":
                result = ((step.get("result") or {}).get("output") or {})
                candidate_results.extend(result.get("results") or [])
                candidate_sites.extend(result.get("candidate_websites") or [])

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

        tool_calls = 0
        tool_history: List[Dict[str, Any]] = []
        step_logs: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt(spec, system_prompt_text, developer_prompt_text)
        invalid_decision_count = 0

        for step in range(1, spec.max_steps + 1):
            elapsed = time.perf_counter() - run_started
            if elapsed > spec.max_runtime_seconds:
                step_logs.append({"step": step, "type": "timeout", "message": "agent runtime exceeded"})
                break

            decision = await self.llm.generate_json(
                prompt=self._decision_prompt(
                    user_message=message,
                    context=context,
                    memory_context=memory_context,
                    tool_specs=tool_specs,
                    tool_history=tool_history,
                    step=step,
                    max_steps=spec.max_steps,
                ),
                system_prompt=system_prompt,
                temperature=0.2,
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
                            json_output = await self.llm.generate_json(
                                prompt=(
                                    "Convert the text below into JSON matching this schema shape.\n"
                                    f"Schema:\n{json.dumps(spec.output_schema, ensure_ascii=False)}\n\n"
                                    f"Text:\n{final_answer}"
                                ),
                                system_prompt=system_prompt,
                                temperature=0.1,
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
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
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
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
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
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }
