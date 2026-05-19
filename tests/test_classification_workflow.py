import asyncio

from config.loader import ConfigLoader
from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.orchestration.engine import WorkflowEngine
from core.tools.registry import ToolRegistry


class FakeAdapter(BaseLLMAdapter):
    def __init__(self):
        super().__init__(api_key="x", model_name="fake-model")
        self.last_prompt = ""
        self.last_system_prompt = ""

    async def generate(self, prompt: str, system_prompt=None, temperature: float = 0.7, max_tokens: int = 4096):
        return LLMResponse(text="ok", input_tokens=1, output_tokens=1, model=self.model_name, latency_ms=1)

    async def generate_with_vision(self, prompt: str, images, system_prompt=None, temperature: float = 0.7):
        return LLMResponse(text="vision", input_tokens=1, output_tokens=1, model=self.model_name, latency_ms=1)

    async def generate_json(self, prompt: str, system_prompt=None, temperature: float = 0.3):
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt or ""
        return {
            "needs_ticket": True,
            "request_type": "post_sales_installation",
            "product_family": "heater",
            "customer_role": "end_customer",
            "customer_stage": "installation",
            "issue_area": "installation_help",
            "error_code": None,
            "priority": "medium",
            "ticket_queue": "support_installation",
            "ticket_title": "Heater installation help request",
            "ticket_summary": "Customer needs help while installing the heater.",
            "confidence": 0.93,
        }

    async def embed(self, text: str):
        return [0.1, 0.2]


class StubAgentFactory:
    def __init__(self, adapter):
        self.base_adapter = adapter
        self.retriever = None
        self.todo_store = None
        self.tool_registry = ToolRegistry()


def test_classification_workflow_uses_profile_dimensions():
    adapter = FakeAdapter()
    engine = WorkflowEngine(
        config_loader=ConfigLoader(),
        agent_factory=StubAgentFactory(adapter),
    )

    result = asyncio.run(
        engine.run(
            workflow_id="email_auto_ticketing_v1",
            payload={
                "conversation_text": "I bought your heater and I am stuck during installation.",
                "subject": "Need heater installation help",
                "customer_email": "customer@example.com",
            },
        )
    )

    classification = result["state"]["outputs"]["classification"]
    assert classification["needs_ticket"] is True
    assert classification["product_family"] == "heater"
    assert "request_type" in adapter.last_system_prompt
    assert "Need heater installation help" in adapter.last_prompt
