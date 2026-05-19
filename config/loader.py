import json
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from config.schemas import (
    WorkflowConfig,
    NodeConfig,
    AppConfig,
    ModelProfile,
    VectorStoreConfig,
    FilterConfig,
    FilterCondition,
    AgentSpec,
    WorkflowSpec,
    WorkflowNodeSpec,
    WorkflowTransition,
    ClassificationDimensionSpec,
    ClassificationProfileSpec,
)


class ConfigLoader:

    def __init__(
        self,
        workflows_dir: str = "config/workflow_settings",
        agents_dir: str = "config/agents",
        workflow_specs_dir: str = "config/workflows",
        workflow_profiles_dir: str = "config/workflow_profiles",
    ):
        self.workflows_dir = Path(workflows_dir)
        self.agents_dir = Path(agents_dir)
        self.workflow_specs_dir = Path(workflow_specs_dir)
        self.workflow_profiles_dir = Path(workflow_profiles_dir)
        self._cache: Dict[str, WorkflowConfig] = {}
        self._agents_cache: Dict[str, AgentSpec] = {}
        self._workflow_specs_cache: Dict[str, WorkflowSpec] = {}
        self._workflow_profiles_cache: Dict[str, ClassificationProfileSpec] = {}
        self._prompts_cache: Dict[str, str] = {}
        self._model_profiles: Optional[Dict[str, ModelProfile]] = None

        # Injected at startup by api/main.py after Firestore is ready
        self._agent_store = None
        self._workflow_store = None

    # ── DB injection (called once at app startup) ─────────────────────────────

    def set_db_stores(self, agent_store, workflow_store) -> None:
        """Inject Firestore stores so DB-first loading is enabled."""
        self._agent_store = agent_store
        self._workflow_store = workflow_store

    async def preload_from_db(self) -> None:
        """Pull all agents and workflows from Firestore into memory caches.

        Called once at startup after Firestore is initialised.  After this,
        the existing synchronous load_agent_spec / load_workflow_spec methods
        serve from cache (which may have been populated from DB), so no other
        code needs to change.
        """
        if self._agent_store:
            try:
                records = await self._agent_store.list_all()
                loaded = 0
                for record in records:
                    agent_id = record.get("agent_id")
                    if not agent_id:
                        continue
                    spec = self._parse_agent_spec(record)
                    # Inline prompt text takes precedence over file path
                    if record.get("system_prompt"):
                        spec.system_prompt_path = None
                    self._agents_cache[agent_id] = spec
                    loaded += 1
                print(f"  Loaded {loaded} agents from Firestore into cache")
            except Exception as exc:
                print(f"  Warning: Firestore agent preload failed, using files: {exc}")

        if self._workflow_store:
            try:
                records = await self._workflow_store.list_all()
                loaded = 0
                for record in records:
                    workflow_id = record.get("workflow_id")
                    if not workflow_id:
                        continue
                    spec = self._parse_workflow_spec(record)
                    self._workflow_specs_cache[workflow_id] = spec
                    loaded += 1
                print(f"  Loaded {loaded} workflows from Firestore into cache")
            except Exception as exc:
                print(f"  Warning: Firestore workflow preload failed, using files: {exc}")

    def load_workflow(self, name: str) -> WorkflowConfig:
        if name in self._cache:
            return self._cache[name]

        config_path = self._find_config_file(name)
        if not config_path:
            raise FileNotFoundError(f"Workflow config not found: {name}")

        data = self._load_file(config_path)
        workflow = self._parse_workflow(data)

        self._cache[name] = workflow
        return workflow

    def load_prompt(self, path: str) -> str:
        if path in self._prompts_cache:
            return self._prompts_cache[path]

        prompt_path = Path(path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self._prompts_cache[path] = content
        return content

    def load_agent_spec(self, agent_id: str) -> AgentSpec:
        if agent_id in self._agents_cache:
            return self._agents_cache[agent_id]

        config_path = self._find_config_file(agent_id, base_dir=self.agents_dir)
        if not config_path:
            raise FileNotFoundError(f"Agent spec not found: {agent_id}")

        data = self._load_file(config_path)
        spec = self._parse_agent_spec(data)
        self._agents_cache[agent_id] = spec
        return spec

    def load_workflow_spec(self, workflow_id: str) -> WorkflowSpec:
        if workflow_id in self._workflow_specs_cache:
            return self._workflow_specs_cache[workflow_id]

        config_path = self._find_config_file(workflow_id, base_dir=self.workflow_specs_dir)
        if not config_path:
            raise FileNotFoundError(f"Workflow spec not found: {workflow_id}")

        data = self._load_file(config_path)
        spec = self._parse_workflow_spec(data)
        self._workflow_specs_cache[workflow_id] = spec
        return spec

    def load_classification_profile(self, profile_id: str) -> ClassificationProfileSpec:
        if profile_id in self._workflow_profiles_cache:
            return self._workflow_profiles_cache[profile_id]

        config_path = self._find_config_file(profile_id, base_dir=self.workflow_profiles_dir)
        if not config_path:
            raise FileNotFoundError(f"Classification profile not found: {profile_id}")

        data = self._load_file(config_path)
        profile = self._parse_classification_profile(data)
        self._workflow_profiles_cache[profile_id] = profile
        return profile

    def load_model_profiles(self, path: str = "config/models.yaml") -> Dict[str, ModelProfile]:
        if self._model_profiles is not None:
            return self._model_profiles

        config_path = Path(path)
        if not config_path.exists():
            self._model_profiles = {}
            return self._model_profiles

        data = self._load_file(config_path) or {}
        profiles = {}
        for name, profile_data in (data.get("profiles") or {}).items():
            profiles[name] = ModelProfile(
                name=name,
                provider=profile_data.get("provider", "gemini"),
                model=profile_data.get("model", "gemini-2.5-flash"),
                temperature=profile_data.get("temperature", 0.7),
                max_tokens=profile_data.get("max_tokens", 4096),
                embedding_model=profile_data.get("embedding_model"),
            )

        self._model_profiles = profiles
        return profiles

    def parse_filter_config(self, data: Optional[Dict[str, Any]]) -> Optional[FilterConfig]:
        if not data:
            return None

        must = self._parse_filter_conditions(data.get("must", []))
        should = self._parse_filter_conditions(data.get("should", []))
        must_not = self._parse_filter_conditions(data.get("must_not", []))
        minimum_should_match = data.get("minimum_should_match")

        return FilterConfig(
            must=must,
            should=should,
            must_not=must_not,
            minimum_should_match=minimum_should_match,
        )

    def resolve_filter_placeholders(
        self,
        filter_config: FilterConfig,
        context: Dict[str, Any],
    ) -> FilterConfig:
        return FilterConfig(
            must=self._resolve_conditions(filter_config.must, context),
            should=self._resolve_conditions(filter_config.should, context),
            must_not=self._resolve_conditions(filter_config.must_not, context),
            minimum_should_match=filter_config.minimum_should_match,
        )

    def load_app_config(self, path: str = "config/app.yaml") -> AppConfig:
        config_path = Path(path)
        if not config_path.exists():
            return AppConfig()

        data = self._load_file(config_path)
        return self._parse_app_config(data)

    def _find_config_file(self, name: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        extensions = [".json", ".yaml", ".yml"]
        target_dir = base_dir or self.workflows_dir

        for ext in extensions:
            path = target_dir / f"{name}{ext}"
            if path.exists():
                return path

        return None

    def _load_file(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == ".json":
                return json.load(f)
            else:
                return yaml.safe_load(f)

    def _parse_workflow(self, data: Dict[str, Any]) -> WorkflowConfig:
        nodes = []
        for node_data in data.get("nodes", []):
            node = NodeConfig(
                name=node_data.get("name", ""),
                type=node_data.get("type", ""),
                model=node_data.get("model", "gemini-2.5-flash"),
                model_profile=node_data.get("model_profile"),
                system_prompt=node_data.get("system_prompt", ""),
                system_prompt_path=node_data.get("system_prompt_path"),
                output_format=node_data.get("output_format"),
                embedding_collection=node_data.get("embedding_collection"),
                tools=node_data.get("tools", []),
                temperature=node_data.get("temperature", 0.7),
                max_tokens=node_data.get("max_tokens", 4096)
            )
            nodes.append(node)

        vector_store = None
        vector_data = data.get("vector_store")
        if vector_data:
            vector_store = VectorStoreConfig(
                provider=vector_data.get("provider", "qdrant"),
                collection=vector_data.get("collection"),
                filters=self.parse_filter_config(vector_data.get("filters")),
            )

        return WorkflowConfig(
            workflow=data.get("workflow", ""),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            nodes=nodes,
            vector_store=vector_store,
        )

    def _parse_agent_spec(self, data: Dict[str, Any]) -> AgentSpec:
        return AgentSpec(
            agent_id=data.get("agent_id", ""),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            model_profile=data.get("model_profile"),
            system_prompt=data.get("system_prompt", ""),
            system_prompt_path=data.get("system_prompt_path"),
            developer_prompt=data.get("developer_prompt", ""),
            developer_prompt_path=data.get("developer_prompt_path"),
            response_mode=data.get("response_mode", "text"),
            output_schema=data.get("output_schema"),
            allowed_tools=data.get("allowed_tools", []) or [],
            allowed_sub_agents=data.get("allowed_sub_agents", []) or [],
            memory_window=data.get("memory_window", 6),
            max_steps=data.get("max_steps", 8),
            max_tool_calls=data.get("max_tool_calls", 6),
            max_runtime_seconds=data.get("max_runtime_seconds", 45),
            default_context=data.get("default_context", {}) or {},
            metadata=data.get("metadata", {}) or {},
        )

    def _parse_workflow_spec(self, data: Dict[str, Any]) -> WorkflowSpec:
        nodes: List[WorkflowNodeSpec] = []
        for node_data in data.get("nodes", []):
            transitions: List[WorkflowTransition] = []
            for transition_data in node_data.get("transitions", []) or []:
                transitions.append(
                    WorkflowTransition(
                        to=transition_data.get("to", ""),
                        when=transition_data.get("when", "always"),
                    )
                )

            nodes.append(
                WorkflowNodeSpec(
                    node_id=node_data.get("node_id", ""),
                    node_type=node_data.get("node_type", ""),
                    config=node_data.get("config", {}) or {},
                    input_map=node_data.get("input_map", {}) or {},
                    output_map=node_data.get("output_map", {}) or {},
                    transitions=transitions,
                )
            )

        return WorkflowSpec(
            workflow_id=data.get("workflow_id", ""),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            start_at=data.get("start_at", ""),
            nodes=nodes,
            metadata=data.get("metadata", {}) or {},
        )

    def _parse_classification_profile(self, data: Dict[str, Any]) -> ClassificationProfileSpec:
        dimensions: List[ClassificationDimensionSpec] = []
        for dimension_data in data.get("dimensions", []) or []:
            dimensions.append(
                ClassificationDimensionSpec(
                    name=dimension_data.get("name", ""),
                    description=dimension_data.get("description", ""),
                    field_type=dimension_data.get("field_type", "enum"),
                    allowed_values=dimension_data.get("allowed_values", []) or [],
                    required=dimension_data.get("required", True),
                    allow_null=dimension_data.get("allow_null", False),
                    metadata=dimension_data.get("metadata", {}) or {},
                )
            )

        return ClassificationProfileSpec(
            profile_id=data.get("profile_id", ""),
            version=data.get("version", "1.0"),
            workflow_id=data.get("workflow_id"),
            description=data.get("description", ""),
            instructions=data.get("instructions", []) or [],
            dimensions=dimensions,
            metadata=data.get("metadata", {}) or {},
        )

    def _parse_app_config(self, data: Dict[str, Any]) -> AppConfig:
        from config.schemas import ModelConfig, RetrievalConfig, LoggingConfig

        model_data = data.get("model", {})
        retrieval_data = data.get("retrieval", {})
        logging_data = data.get("logging", {})

        return AppConfig(
            model=ModelConfig(**model_data) if model_data else ModelConfig(),
            retrieval=RetrievalConfig(**retrieval_data) if retrieval_data else RetrievalConfig(),
            logging=LoggingConfig(**logging_data) if logging_data else LoggingConfig(),
            workflows_dir=data.get("workflows_dir", "config/workflow_settings")
        )

    def list_workflow_specs(self) -> List[Dict[str, Any]]:
        """Return id + description for every workflow spec YAML in the specs dir."""
        results = []
        if not self.workflow_specs_dir.exists():
            return results
        for path in sorted(self.workflow_specs_dir.glob("*.yaml")):
            try:
                data = self._load_file(path)
                results.append({
                    "workflow_id": data.get("workflow_id", path.stem),
                    "description": data.get("description", ""),
                    "version": data.get("version", "1.0"),
                    "nodes": [n.get("node_id") for n in data.get("nodes", [])],
                })
            except Exception:
                pass
        return results

    def list_agent_specs(self) -> List[Dict[str, Any]]:
        """Return id + description for every agent spec YAML in the agents dir."""
        results = []
        if not self.agents_dir.exists():
            return results
        for path in sorted(self.agents_dir.glob("*.yaml")):
            try:
                data = self._load_file(path)
                results.append({
                    "agent_id": data.get("agent_id", path.stem),
                    "description": data.get("description", ""),
                    "version": data.get("version", "1.0"),
                    "allowed_tools": data.get("allowed_tools", []),
                    "model_profile": data.get("model_profile", ""),
                })
            except Exception:
                pass
        return results

    def clear_cache(self):
        self._cache.clear()
        self._agents_cache.clear()
        self._workflow_specs_cache.clear()
        self._workflow_profiles_cache.clear()
        self._prompts_cache.clear()
        self._model_profiles = None

    def _parse_filter_conditions(self, data: List[Dict[str, Any]]) -> List[FilterCondition]:
        conditions = []
        for item in data or []:
            if isinstance(item, FilterCondition):
                conditions.append(item)
                continue
            if not isinstance(item, dict):
                continue
            field = item.get("field", "")
            if not field:
                continue
            conditions.append(
                FilterCondition(
                    field=field,
                    op=item.get("op", "eq"),
                    value=item.get("value"),
                )
            )
        return conditions

    def _resolve_conditions(self, conditions: List[FilterCondition], context: Dict[str, Any]) -> List[FilterCondition]:
        resolved = []
        for condition in conditions:
            value = self._resolve_value(condition.value, context)
            if value is None:
                continue
            if isinstance(value, list) and not value:
                continue
            resolved.append(
                FilterCondition(
                    field=condition.field,
                    op=condition.op,
                    value=value,
                )
            )
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            pattern = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

            def replace(match: re.Match) -> str:
                key = match.group(1)
                return str(context.get(key, ""))

            resolved = pattern.sub(replace, value).strip()
            return resolved or None

        if isinstance(value, list):
            resolved_list = []
            for item in value:
                resolved_item = self._resolve_value(item, context)
                if resolved_item is None or resolved_item == "":
                    continue
                resolved_list.append(resolved_item)
            return resolved_list

        if isinstance(value, dict):
            resolved_dict = {}
            for key, item in value.items():
                resolved_item = self._resolve_value(item, context)
                if resolved_item is None or resolved_item == "":
                    continue
                resolved_dict[key] = resolved_item
            return resolved_dict

        return value
