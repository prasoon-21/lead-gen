import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config.loader import ConfigLoader
from config.schemas import AgentSpec
from core.adapters.base import BaseLLMAdapter
from core.adapters.router import build_adapter_from_env
from core.agents.kernel import AgentKernel
from core.agents.memory import MemoryStore
from core.tools.defaults import build_default_tool_registry
from core.tools.registry import ToolRegistry


class AgentFactory:
    def __init__(
        self,
        config_loader: ConfigLoader,
        base_adapter: Optional[BaseLLMAdapter],
        retriever=None,
        token_tracker=None,
        tool_registry: Optional[ToolRegistry] = None,
        todo_store=None,
    ):
        self.config_loader = config_loader
        self.base_adapter = base_adapter
        self.retriever = retriever
        self.token_tracker = token_tracker
        self.tool_registry = tool_registry or build_default_tool_registry()
        self.memory_store = MemoryStore()
        self.todo_store = todo_store
        self._logger = logging.getLogger("agent.runtime")

    def _load_spec(self, agent_id: str) -> AgentSpec:
        return self.config_loader.load_agent_spec(agent_id)

    def _resolve_adapter(self, spec: AgentSpec) -> Optional[BaseLLMAdapter]:
        if not spec.model_profile:
            return self.base_adapter

        profiles = self.config_loader.load_model_profiles()
        profile = profiles.get(spec.model_profile)
        if not profile:
            return self.base_adapter

        adapter = build_adapter_from_env(
            provider_override=profile.provider,
            model_override=profile.model,
            embedding_model_override=profile.embedding_model,
        )
        return adapter or self.base_adapter

    def _load_prompt(self, path: Optional[str]) -> str:
        if not path:
            return ""
        try:
            return self.config_loader.load_prompt(path)
        except FileNotFoundError:
            return ""

    @staticmethod
    def _inject_date(prompt: str) -> str:
        """Prepend the current date so the LLM can resolve relative date references correctly."""
        if not prompt:
            return prompt
        now = datetime.now(timezone.utc)
        date_line = f"[System: Today's date is {now.strftime('%A, %d %B %Y')} (UTC). Use this as the anchor for all relative date calculations.]\n\n"
        return date_line + prompt

    async def run_with_spec(
        self,
        spec: AgentSpec,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an agent from a pre-built AgentSpec (e.g. loaded from the agent directory DB)."""
        adapter = self._resolve_adapter(spec)
        if not adapter:
            raise ValueError("LLM adapter not initialized")

        kernel = AgentKernel(
            llm=adapter,
            tool_registry=self.tool_registry,
            token_tracker=self.token_tracker,
            memory_store=self.memory_store,
        )

        session = session_id or f"SESS_{uuid.uuid4()}"
        system_prompt_text = self._inject_date(spec.system_prompt or "")
        developer_prompt_text = spec.developer_prompt or ""
        caller = (context or {}).get("user_id") or (context or {}).get("user") or "unknown"
        self._logger.info(
            "agent_run_start",
            extra={
                "event": "agent_run_start",
                "trace_id": "-",
                "session_id": session,
                "agent_id": spec.agent_id,
                "workflow_id": "-",
                "step": 0,
                "tool_name": "-",
                "status": f"caller={caller} source=directory",
            },
        )

        resources = {
            "adapter": adapter,
            "retriever": self.retriever,
            "config_loader": self.config_loader,
            "token_tracker": self.token_tracker,
            "agent_factory": self,
            "todo_store": self.todo_store,
        }
        result = await kernel.run(
            spec=spec,
            session_id=session,
            message=message,
            context=context or {},
            resources=resources,
            system_prompt_text=system_prompt_text,
            developer_prompt_text=developer_prompt_text,
        )
        self._logger.info(
            "agent_run_end",
            extra={
                "event": "agent_run_end",
                "trace_id": result.get("trace_id", "-"),
                "session_id": result.get("session_id", session),
                "agent_id": spec.agent_id,
                "workflow_id": "-",
                "step": 0,
                "tool_name": "-",
                "status": f"completed={result.get('metadata', {}).get('completed', False)}",
            },
        )
        return result

    async def run(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = self._load_spec(agent_id)
        adapter = self._resolve_adapter(spec)
        if not adapter:
            raise ValueError("LLM adapter not initialized")

        kernel = AgentKernel(
            llm=adapter,
            tool_registry=self.tool_registry,
            token_tracker=self.token_tracker,
            memory_store=self.memory_store,
        )

        session = session_id or f"SESS_{uuid.uuid4()}"
        system_prompt_text = self._inject_date(self._load_prompt(spec.system_prompt_path))
        developer_prompt_text = self._load_prompt(spec.developer_prompt_path)
        caller = (context or {}).get("user_id") or (context or {}).get("user") or "unknown"
        self._logger.info(
            "agent_run_start",
            extra={
                "event": "agent_run_start",
                "trace_id": "-",
                "session_id": session,
                "agent_id": spec.agent_id,
                "workflow_id": "-",
                "step": 0,
                "tool_name": "-",
                "status": f"caller={caller}",
            },
        )
        self._logger.info(
            "agent_prompt_loaded",
            extra={
                "event": "agent_prompt_loaded",
                "trace_id": "-",
                "session_id": session,
                "agent_id": spec.agent_id,
                "workflow_id": "-",
                "step": 0,
                "tool_name": "-",
                "status": f"system_path={spec.system_prompt_path or '-'} developer_path={spec.developer_prompt_path or '-'}",
            },
        )

        resources = {
            "adapter": adapter,
            "retriever": self.retriever,
            "config_loader": self.config_loader,
            "token_tracker": self.token_tracker,
            "agent_factory": self,
            "todo_store": self.todo_store,
        }
        result = await kernel.run(
            spec=spec,
            session_id=session,
            message=message,
            context=context or {},
            resources=resources,
            system_prompt_text=system_prompt_text,
            developer_prompt_text=developer_prompt_text,
        )
        self._logger.info(
            "agent_run_end",
            extra={
                "event": "agent_run_end",
                "trace_id": result.get("trace_id", "-"),
                "session_id": result.get("session_id", session),
                "agent_id": spec.agent_id,
                "workflow_id": "-",
                "step": 0,
                "tool_name": "-",
                "status": f"completed={result.get('metadata', {}).get('completed', False)} tool_calls={result.get('metadata', {}).get('tool_calls', 0)}",
            },
        )
        return result
