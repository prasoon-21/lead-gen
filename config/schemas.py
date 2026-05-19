from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class NodeConfig:
    name: str
    type: str
    model: str
    model_profile: Optional[str] = None
    system_prompt: str = ""
    system_prompt_path: Optional[str] = None
    output_format: Optional[str] = None
    embedding_collection: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class WorkflowConfig:
    workflow: str
    version: str = "1.0"
    description: str = ""
    nodes: List[NodeConfig] = field(default_factory=list)
    vector_store: Optional["VectorStoreConfig"] = None
    
    def get_node(self, name: str) -> Optional[NodeConfig]:
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def get_nodes_by_type(self, node_type: str) -> List[NodeConfig]:
        return [node for node in self.nodes if node.type == node_type]


@dataclass
class ModelConfig:
    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    fallback_api_keys: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class ModelProfile:
    name: str
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    embedding_model: Optional[str] = None


@dataclass
class FilterCondition:
    field: str
    op: str = "eq"
    value: Any = None


@dataclass
class FilterConfig:
    must: List[FilterCondition] = field(default_factory=list)
    should: List[FilterCondition] = field(default_factory=list)
    must_not: List[FilterCondition] = field(default_factory=list)
    minimum_should_match: Optional[int] = None


@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"
    collection: Optional[str] = None
    filters: Optional[FilterConfig] = None


@dataclass
class RetrievalConfig:
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "aurika_campus_content"
    embedding_model: str = "models/text-embedding-004"
    top_k: int = 10
    score_threshold: float = 0.5


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_retrieval: bool = True
    log_tokens: bool = True
    export_to_csv: bool = True
    csv_path: str = "logs/token_usage.csv"
    export_to_json: bool = True
    json_path: str = "logs/agent_logs.jsonl"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    workflows_dir: str = "config/workflows"


@dataclass
class AgentSpec:
    agent_id: str
    version: str = "1.0"
    description: str = ""
    model_profile: Optional[str] = None
    system_prompt: str = ""
    system_prompt_path: Optional[str] = None
    developer_prompt: str = ""
    developer_prompt_path: Optional[str] = None
    response_mode: str = "text"
    output_schema: Optional[Dict[str, Any]] = None
    allowed_tools: List[str] = field(default_factory=list)
    allowed_sub_agents: List[str] = field(default_factory=list)
    memory_window: int = 6
    max_steps: int = 8
    max_tool_calls: int = 6
    max_runtime_seconds: int = 45
    default_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTransition:
    to: str
    when: str = "always"


@dataclass
class WorkflowNodeSpec:
    node_id: str
    node_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    input_map: Dict[str, str] = field(default_factory=dict)
    output_map: Dict[str, str] = field(default_factory=dict)
    transitions: List[WorkflowTransition] = field(default_factory=list)


@dataclass
class WorkflowSpec:
    workflow_id: str
    version: str = "1.0"
    description: str = ""
    start_at: str = ""
    nodes: List[WorkflowNodeSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Optional[WorkflowNodeSpec]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None


@dataclass
class ClassificationDimensionSpec:
    name: str
    description: str
    field_type: str = "enum"
    allowed_values: List[str] = field(default_factory=list)
    required: bool = True
    allow_null: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationProfileSpec:
    profile_id: str
    version: str = "1.0"
    workflow_id: Optional[str] = None
    description: str = ""
    instructions: List[str] = field(default_factory=list)
    dimensions: List[ClassificationDimensionSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_dimension(self, name: str) -> Optional[ClassificationDimensionSpec]:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        return None
