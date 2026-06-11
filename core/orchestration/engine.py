import time
import uuid
import logging
import json
from typing import Any, Dict, Optional

from config.loader import ConfigLoader
from config.schemas import WorkflowNodeSpec, WorkflowSpec
from core.agents.factory import AgentFactory
from core.orchestration.node_registry import NodeRegistry
from core.orchestration.state import get_path, map_inputs, map_outputs, set_path


class WorkflowEngine:
    _FAIL_FAST_RULES = {
        "discover_directories": {
            "metric": "directories_found",
            "path": "nodes.discover_directories.output.metadata.directories_found",
            "reason": "No relevant directories were found for this industry and location.",
            "drop_path": "nodes.discover_directories.output.metadata.dropped_reasons",
        },
        "extract_companies": {
            "metric": "companies_extracted",
            "path": "nodes.extract_companies.output.metadata.companies_extracted",
            "reason": "Directories were found, but no companies could be extracted from them.",
            "drop_path": "nodes.extract_companies.output.metadata.dropped_reasons",
        },
        "validate_companies": {
            "metric": "validated_companies",
            "path": "nodes.validate_companies.output.metadata.validated_companies",
            "reason": "Companies were extracted, but none had a valid website after validation.",
            "drop_path": "nodes.validate_companies.output.metadata.dropped_reasons",
        },
    }
    _DIRECT_DISCOVERY_FALLBACK = {
        "workflow_id": "lead_generation_directory_pipeline",
        "agent_id": "lead_company_discovery_agent",
        "resume_node": "validate_companies",
        "eligible_stages": {"discover_directories", "extract_companies"},
    }

    def __init__(
        self,
        config_loader: ConfigLoader,
        agent_factory: AgentFactory,
        node_registry: NodeRegistry,
        token_tracker=None,
    ):
        self.config_loader = config_loader
        self.agent_factory = agent_factory
        self.node_registry = node_registry
        self.token_tracker = token_tracker
        self._runtime_logger = logging.getLogger("agent.runtime")

        # Shared resources passed into every node executor at call time.
        self._resources = {
            "adapter": agent_factory.base_adapter,
            "retriever": agent_factory.retriever,
            "tool_registry": agent_factory.tool_registry,
            "config_loader": config_loader,
            "token_tracker": token_tracker,
            "agent_factory": agent_factory,
            "todo_store": agent_factory.todo_store,
        }

    def _log(self, trace_id: str, event: str, data: Dict[str, Any]) -> None:
        payload = data.get("payload", data)
        try:
            payload_preview = json.dumps(payload, ensure_ascii=False)
        except Exception:
            payload_preview = str(payload)
        if len(payload_preview) > 1200:
            payload_preview = payload_preview[:1200] + "...[truncated]"
        self._runtime_logger.info(
            event,
            extra={
                "event": event,
                "trace_id": trace_id,
                "session_id": data.get("session_id", "-"),
                "agent_id": data.get("agent_id", "-"),
                "workflow_id": data.get("workflow_id", "-"),
                "step": data.get("step", 0),
                "tool_name": data.get("tool_name", "-"),
                "status": data.get("status", "-"),
                "payload": payload_preview,
            },
        )
        if self.token_tracker:
            self.token_tracker.log_event(event, {"trace_id": trace_id, **data})

    def _next_node(self, node: WorkflowNodeSpec, state: Dict[str, Any]) -> Optional[str]:
        for transition in node.transitions:
            if self._check_condition(transition.when, state):
                return transition.to
        return None

    @staticmethod
    def _check_condition(condition: str, state: Dict[str, Any]) -> bool:
        cond = (condition or "always").strip()
        if cond == "always":
            return True
        if cond.startswith("truthy:"):
            return bool(get_path(state, cond.split(":", 1)[1], None))
        if cond.startswith("exists:"):
            return get_path(state, cond.split(":", 1)[1], None) is not None
        if cond.startswith("equals:"):
            _, path, expected = cond.split(":", 2)
            return str(get_path(state, path, "")) == expected
        if cond.startswith("not_equals:"):
            _, path, expected = cond.split(":", 2)
            return str(get_path(state, path, "")) != expected
        return False

    async def _run_node(
        self,
        node: WorkflowNodeSpec,
        state: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        node_input = map_inputs(state, node.input_map)
        executor = self.node_registry.get(node.node_type)
        return await executor.execute(node, node_input, state, trace_id, self._resources)

    def _evaluate_fail_fast(
        self,
        node_id: str,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        rule = self._FAIL_FAST_RULES.get(node_id)
        if not rule:
            return None

        metric_value = int(get_path(state, rule["path"], 0) or 0)
        if metric_value > 0:
            return None

        top_drop_reason = ""
        drop_reasons = get_path(state, rule.get("drop_path", ""), {}) or {}
        if isinstance(drop_reasons, dict) and drop_reasons:
            top_drop_reason = max(
                drop_reasons.items(),
                key=lambda item: int(item[1] or 0),
            )[0]

        reason = rule["reason"]
        if top_drop_reason:
            reason = f"{reason} Top issue: {top_drop_reason}."

        return {
            "stage": node_id,
            "metric": rule["metric"],
            "value": metric_value,
            "reason": reason,
            "top_drop_reason": top_drop_reason,
        }

    def _can_use_direct_discovery_fallback(
        self,
        workflow_id: str,
        node_id: str,
        state: Dict[str, Any],
    ) -> bool:
        config = self._DIRECT_DISCOVERY_FALLBACK
        if workflow_id != config["workflow_id"]:
            return False
        if node_id not in config["eligible_stages"]:
            return False
        if get_path(state, "workflow.fallback_company_discovery", None) is not None:
            return False
        return True

    async def _attempt_direct_company_discovery_fallback(
        self,
        *,
        workflow_id: str,
        node_id: str,
        state: Dict[str, Any],
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        config = self._DIRECT_DISCOVERY_FALLBACK
        session_id = state.get("session_id")
        message = get_path(state, "input.message", "")
        context = get_path(state, "input.context", {}) or {}

        self._log(
            trace_id,
            "workflow_fallback_start",
            {
                "workflow_id": workflow_id,
                "session_id": session_id,
                "status": f"from_stage={node_id} fallback={config['agent_id']}",
                "payload": {
                    "from_stage": node_id,
                    "fallback_agent_id": config["agent_id"],
                    "resume_node": config["resume_node"],
                },
            },
        )

        fallback_result = await self.agent_factory.run(
            agent_id=config["agent_id"],
            message=message,
            session_id=session_id,
            context=context,
        )
        companies = get_path(fallback_result, "json.companies", []) or []
        fallback_info = {
            "activated": bool(companies),
            "fallback_agent_id": config["agent_id"],
            "from_stage": node_id,
            "resume_node": config["resume_node"],
            "companies_found": len(companies),
            "tool_calls": int(get_path(fallback_result, "metadata.tool_calls", 0) or 0),
            "input_tokens": int(get_path(fallback_result, "metadata.input_tokens", 0) or 0),
            "output_tokens": int(get_path(fallback_result, "metadata.output_tokens", 0) or 0),
            "total_tokens": int(get_path(fallback_result, "metadata.total_tokens", 0) or 0),
            "latency_ms": float(get_path(fallback_result, "metadata.latency_ms", 0) or 0),
            "steps_count": len(get_path(fallback_result, "steps", []) or []),
        }

        set_path(state, "workflow.fallback_company_discovery", fallback_info)
        set_path(state, "nodes.discover_companies_fallback.output", fallback_result)
        set_path(state, "outputs.fallback_discovery_json", get_path(fallback_result, "json"))
        set_path(state, "outputs.fallback_discovery_text", get_path(fallback_result, "text"))
        set_path(state, "outputs.fallback_discovery_steps", get_path(fallback_result, "steps", []))

        if companies:
            set_path(state, "outputs.discovery_companies", companies)
            set_path(state, "outputs.discovery_steps", get_path(fallback_result, "steps", []))

        self._log(
            trace_id,
            "workflow_fallback_end",
            {
                "workflow_id": workflow_id,
                "session_id": session_id,
                "status": f"from_stage={node_id} activated={bool(companies)} companies={len(companies)}",
                "payload": fallback_info,
            },
        )
        return fallback_info

    @staticmethod
    def _build_node_summary_from_result(node_id: str, node_type: str, node_output: Dict[str, Any]) -> Dict[str, Any]:
        node_meta = node_output.get("metadata", {}) if isinstance(node_output.get("metadata"), dict) else {}
        return {
            "node_id": node_id,
            "node_type": node_type,
            "tool_calls": int(node_meta.get("tool_calls", 0) or 0),
            "input_tokens": int(node_meta.get("input_tokens", 0) or 0),
            "output_tokens": int(node_meta.get("output_tokens", 0) or 0),
            "latency_ms": float(node_meta.get("latency_ms", 0) or 0),
            "manual_flow": bool(node_meta.get("manual_flow", False)),
        }

    async def run(
        self,
        workflow_id: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec: WorkflowSpec = self.config_loader.load_workflow_spec(workflow_id)
        if not spec.start_at:
            raise ValueError(f"start_at is required in workflow spec: {workflow_id}")

        trace_id = f"WTRACE_{uuid.uuid4()}"
        state: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "session_id": session_id or f"SESS_{uuid.uuid4()}",
            "input": payload,
            "nodes": {},
        }
        if isinstance(payload, dict):
            state.update(payload)

        current_node_id = spec.start_at
        visited_steps = 0
        max_steps = max(20, len(spec.nodes) * 4)
        started = time.perf_counter()
        aggregated_tool_calls = 0
        aggregated_input_tokens = 0
        aggregated_output_tokens = 0
        node_summaries = []
        fail_fast_info: Optional[Dict[str, Any]] = None
        fallback_info: Optional[Dict[str, Any]] = None

        while current_node_id and visited_steps < max_steps:
            visited_steps += 1
            node = spec.get_node(current_node_id)
            if not node:
                raise ValueError(f"Node not found in workflow '{workflow_id}': {current_node_id}")
            node_input_preview = map_inputs(state, node.input_map)

            self._log(
                trace_id,
                "workflow_node_start",
                {
                    "workflow_id": workflow_id,
                    "session_id": state.get("session_id"),
                    "status": f"node={node.node_id} type={node.node_type}",
                    "payload": {"node_id": node.node_id, "node_type": node.node_type},
                },
            )
            self._log(
                trace_id,
                "workflow_node_input",
                {
                    "workflow_id": workflow_id,
                    "session_id": state.get("session_id"),
                    "status": f"node={node.node_id} type={node.node_type}",
                    "payload": {"node_id": node.node_id, "input": node_input_preview},
                },
            )
            try:
                node_output = await self._run_node(node, state, trace_id)
            except Exception as exc:
                self._log(
                    trace_id,
                    "workflow_node_error",
                    {
                        "workflow_id": workflow_id,
                        "session_id": state.get("session_id"),
                        "status": f"node={node.node_id} type={node.node_type} error",
                        "payload": {
                            "node_id": node.node_id,
                            "node_type": node.node_type,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    },
                )
                raise
            set_path(state, f"nodes.{node.node_id}.output", node_output)
            map_outputs(state, node_output if isinstance(node_output, dict) else {"result": node_output}, node.output_map)
            if isinstance(node_output, dict):
                node_summary = self._build_node_summary_from_result(node.node_id, node.node_type, node_output)
                aggregated_tool_calls += node_summary["tool_calls"]
                aggregated_input_tokens += node_summary["input_tokens"]
                aggregated_output_tokens += node_summary["output_tokens"]
                node_summaries.append(node_summary)
            self._log(
                trace_id,
                "workflow_node_end",
                {
                    "workflow_id": workflow_id,
                    "session_id": state.get("session_id"),
                    "status": f"node={node.node_id} type={node.node_type}",
                    "payload": {
                        "node_id": node.node_id,
                        "metadata": node_output.get("metadata", {}) if isinstance(node_output, dict) else {},
                        "output_preview": node_output,
                    },
                },
            )

            fail_fast_info = self._evaluate_fail_fast(node.node_id, state)
            if fail_fast_info:
                if self._can_use_direct_discovery_fallback(workflow_id, node.node_id, state):
                    fallback_info = await self._attempt_direct_company_discovery_fallback(
                        workflow_id=workflow_id,
                        node_id=node.node_id,
                        state=state,
                        trace_id=trace_id,
                    )
                    fallback_output = get_path(state, "nodes.discover_companies_fallback.output", {}) or {}
                    if isinstance(fallback_output, dict):
                        fallback_summary = self._build_node_summary_from_result(
                            "discover_companies_fallback",
                            "agent",
                            fallback_output,
                        )
                        aggregated_tool_calls += fallback_summary["tool_calls"]
                        aggregated_input_tokens += fallback_summary["input_tokens"]
                        aggregated_output_tokens += fallback_summary["output_tokens"]
                        node_summaries.append(fallback_summary)
                    if fallback_info and fallback_info.get("activated"):
                        fail_fast_info = None
                        current_node_id = self._DIRECT_DISCOVERY_FALLBACK["resume_node"]
                        continue

                    if fallback_info:
                        fail_fast_info["fallback_attempted"] = True
                        fail_fast_info["fallback_agent_id"] = fallback_info.get("fallback_agent_id", "")
                        fail_fast_info["fallback_companies_found"] = int(fallback_info.get("companies_found", 0) or 0)

                set_path(state, "workflow.fail_fast", fail_fast_info)
                self._log(
                    trace_id,
                    "workflow_fail_fast",
                    {
                        "workflow_id": workflow_id,
                        "session_id": state.get("session_id"),
                        "status": f"stage={fail_fast_info['stage']} metric={fail_fast_info['metric']}",
                        "payload": fail_fast_info,
                    },
                )
                current_node_id = None
                break

            current_node_id = self._next_node(node, state)

        pipeline_observability = {
            "directories_found": int(get_path(state, "nodes.discover_directories.output.metadata.directories_found", 0) or 0),
            "directory_queries": get_path(state, "nodes.discover_directories.output.metadata.generated_queries", []) or [],
            "companies_extracted": int(get_path(state, "nodes.extract_companies.output.metadata.companies_extracted", 0) or 0),
            "validated_companies": int(get_path(state, "nodes.validate_companies.output.metadata.validated_companies", 0) or 0),
            "valid_websites": int(get_path(state, "nodes.validate_companies.output.metadata.valid_websites", 0) or 0),
            "extract_enriched_companies": int(get_path(state, "nodes.enrich_contacts.output.metadata.extract_enriched_companies", 0) or 0),
            "validated_input_companies": int(get_path(state, "nodes.enrich_contacts.output.metadata.validated_input_companies", 0) or 0),
            "contact_info_count": int(get_path(state, "nodes.enrich_contacts.output.metadata.contact_info_count", 0) or 0),
            "linkedin_people_count": int(get_path(state, "nodes.enrich_contacts.output.metadata.linkedin_people_count", 0) or 0),
            "usable_leads": int(get_path(state, "nodes.score_leads.output.metadata.usable_leads", 0) or 0),
            "fallback_company_discovery": get_path(state, "workflow.fallback_company_discovery", {}) or {},
            "drop_reasons": {
                "directory_discovery": get_path(state, "nodes.discover_directories.output.metadata.dropped_reasons", {}) or {},
                "company_extraction": get_path(state, "nodes.extract_companies.output.metadata.dropped_reasons", {}) or {},
                "company_validation": get_path(state, "nodes.validate_companies.output.metadata.dropped_reasons", {}) or {},
                "contact_enrichment": get_path(state, "nodes.enrich_contacts.output.metadata.dropped_reasons", {}) or {},
                "quality_scoring": get_path(state, "nodes.score_leads.output.metadata.dropped_reasons", {}) or {},
            },
        }
        success_definition = {
            "description": "Given industry + location + role, return 10-20 usable leads with real company websites and at least one valid contact path.",
            "min_usable_leads": 10,
            "max_usable_leads": 20,
            "requires_valid_website": True,
            "requires_contact_path": True,
        }
        meets_success_definition = (
            pipeline_observability["usable_leads"] >= success_definition["min_usable_leads"]
            and pipeline_observability["usable_leads"] <= success_definition["max_usable_leads"]
        )
        if fail_fast_info:
            meets_success_definition = False

        return {
            "success": not bool(fail_fast_info),
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "session_id": state.get("session_id"),
            "state": state,
            "metadata": {
                "steps": visited_steps,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "tool_calls": aggregated_tool_calls,
                "input_tokens": aggregated_input_tokens,
                "output_tokens": aggregated_output_tokens,
                "total_tokens": aggregated_input_tokens + aggregated_output_tokens,
                "node_summaries": node_summaries,
                "pipeline_observability": pipeline_observability,
                "pipeline_failure": fail_fast_info,
                "pipeline_fallback": fallback_info,
                "success_definition": success_definition,
                "meets_success_definition": meets_success_definition,
            },
        }
