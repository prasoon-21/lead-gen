import json
import logging
import re
import time
import uuid
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from config.schemas import AgentSpec
from core.adapters.base import BaseLLMAdapter
from core.agents.memory import MemoryStore
from core.tools import ToolContext, ToolExecutor, ToolPolicy, ToolRegistry


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
                "latency_ms": (time.perf_counter() - run_started) * 1000,
            },
        }
