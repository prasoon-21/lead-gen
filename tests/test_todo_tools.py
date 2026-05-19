import asyncio

from core.tools.base import ToolContext
from core.tools.internal.todo_tools import TodoCreateTool, TodoListTool, TodoUpdateTool


class FakeTodoStore:
    def __init__(self):
        self.created = []
        self.updated = []

    def ensure_store(self):
        return {"spreadsheet_id": "fake"}

    def create_task(self, payload, trace_id="", session_id="", source_agent=""):
        task = {"task_id": "TASK_1", **payload}
        self.created.append(task)
        return task

    def list_tasks(self, filters=None):
        return [{"task_id": "TASK_1", "title": "Sample", "status": "pending"}]

    def update_task(self, task_id, updates, trace_id="", session_id=""):
        task = {"task_id": task_id, **updates}
        self.updated.append(task)
        return task


def _ctx(confirmed_signal: bool = False):
    return ToolContext(
        session_id="SESS_TEST",
        trace_id="TRACE_TEST",
        resources={"todo_store": FakeTodoStore()},
        state={
            "agent_id": "task_manager_agent",
            "_user_confirmation_signal": confirmed_signal,
            "_latest_user_message": "yes confirm" if confirmed_signal else "add task",
        },
    )


def test_todo_create_requires_confirmation():
    tool = TodoCreateTool()
    result = asyncio.run(tool.execute({"title": "Finish PR"}, _ctx()))
    assert result.ok is False
    assert result.error["code"] == "confirmation_required"


def test_todo_create_with_confirmation():
    tool = TodoCreateTool()
    result = asyncio.run(tool.execute({"title": "Finish PR", "confirmed": True}, _ctx(confirmed_signal=True)))
    assert result.ok is True
    assert result.data["task"]["task_id"] == "TASK_1"


def test_todo_list_works():
    tool = TodoListTool()
    result = asyncio.run(tool.execute({"status": "pending"}, _ctx()))
    assert result.ok is True
    assert result.data["count"] == 1


def test_todo_update_requires_confirmation():
    tool = TodoUpdateTool()
    result = asyncio.run(
        tool.execute(
            {"task_id": "TASK_1", "updates": {"priority": "high"}},
            _ctx(),
        )
    )
    assert result.ok is False
    assert result.error["code"] == "confirmation_required"
