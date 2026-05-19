import asyncio

from config.schemas import AgentSpec
from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.agents.kernel import AgentKernel
from core.tools.base import BaseTool, ToolContext
from core.tools.registry import ToolRegistry


class FakeAdapter(BaseLLMAdapter):
    def __init__(self):
        super().__init__(api_key="x", model_name="fake-model")
        self.calls = 0

    async def generate(self, prompt: str, system_prompt=None, temperature: float = 0.7, max_tokens: int = 4096):
        return LLMResponse(text="ok", input_tokens=1, output_tokens=1, model=self.model_name, latency_ms=1)

    async def generate_with_vision(self, prompt: str, images, system_prompt=None, temperature: float = 0.7):
        return LLMResponse(text="vision", input_tokens=1, output_tokens=1, model=self.model_name, latency_ms=1)

    async def generate_json(self, prompt: str, system_prompt=None, temperature: float = 0.3):
        self.calls += 1
        if self.calls == 1:
            return {"type": "tool_call", "thought": "Need helper tool", "tool_name": "echo_tool", "arguments": {"text": "hi"}}
        return {"type": "final_answer", "thought": "done", "final_answer": "completed"}

    async def embed(self, text: str):
        return [0.1, 0.2]


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echo input"
    input_schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def run(self, arguments, context: ToolContext):
        return {"echo": arguments["text"]}


def test_kernel_decision_loop():
    adapter = FakeAdapter()
    registry = ToolRegistry()
    registry.register(EchoTool())
    kernel = AgentKernel(llm=adapter, tool_registry=registry)

    spec = AgentSpec(
        agent_id="test_agent",
        allowed_tools=["echo_tool"],
        max_steps=4,
        max_tool_calls=2,
    )

    result = asyncio.run(
        kernel.run(
            spec=spec,
            session_id="sess-1",
            message="test",
            context={},
            resources={},
        )
    )
    assert result["metadata"]["completed"] is True
    assert result["metadata"]["tool_calls"] == 1
    assert result["text"] == "completed"
