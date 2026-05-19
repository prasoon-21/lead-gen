import asyncio
from typing import Any, Dict

from core.tools.base import BaseTool, ToolContext, ToolExecutionError


def _require_todo_store(context: ToolContext):
    store = context.resources.get("todo_store")
    if not store:
        raise ToolExecutionError(
            "Todo storage is not configured",
            code="todo_store_unavailable",
            details={"hint": "Initialize TodoSheetStore and pass it in tool resources."},
        )
    return store


def _require_confirmation(arguments: Dict[str, Any], action: str, context: ToolContext):
    agent_flag = bool(arguments.get("confirmed", False))
    user_signal = bool(context.state.get("_user_confirmation_signal", False))
    if agent_flag and user_signal:
        return
    raise ToolExecutionError(
        f"Confirmation is required before '{action}'",
        code="confirmation_required",
        details={
            "action": action,
            "required": {
                "tool_argument": "confirmed=true",
                "latest_user_message_contains_confirmation": True,
            },
        },
    )


class TodoEnsureStoreTool(BaseTool):
    name = "todo_ensure_store"
    description = "Ensure todo spreadsheet and required worksheets/headers exist."
    input_schema = {"type": "object", "properties": {}}

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        store = _require_todo_store(context)
        try:
            data = await asyncio.to_thread(store.ensure_store)
        except Exception as exc:
            raise ToolExecutionError(str(exc), code="todo_store_error") from exc
        return {"store": data}


class TodoCreateTool(BaseTool):
    name = "todo_create"
    description = "Create a todo task. Requires explicit confirmation."
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string"},
            "project": {"type": "string"},
            "status": {"type": "string"},
            "priority": {"type": "string"},
            "due_date": {"type": "string"},
            "tags": {"type": "array"},
            "notes": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["title"],
    }

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        _require_confirmation(arguments, "create_task", context)
        store = _require_todo_store(context)
        payload = {
            "title": arguments.get("title"),
            "description": arguments.get("description", ""),
            "category": arguments.get("category", ""),
            "project": arguments.get("project", ""),
            "status": arguments.get("status", "pending"),
            "priority": arguments.get("priority", "medium"),
            "due_date": arguments.get("due_date", ""),
            "tags": arguments.get("tags", []),
            "notes": arguments.get("notes", ""),
        }
        try:
            task = await asyncio.to_thread(
                store.create_task,
                payload,
                context.trace_id,
                context.session_id,
                context.state.get("agent_id", ""),
            )
        except Exception as exc:
            raise ToolExecutionError(str(exc), code="todo_store_error") from exc
        return {"task": task}


class TodoListTool(BaseTool):
    name = "todo_list"
    description = "List todo tasks with optional filters."
    input_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "category": {"type": "string"},
            "project": {"type": "string"},
            "search": {"type": "string"},
            "limit": {"type": "integer"},
            "include_done": {"type": "boolean"},
        },
    }

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        store = _require_todo_store(context)
        filters = dict(arguments or {})
        try:
            tasks = await asyncio.to_thread(store.list_tasks, filters)
        except Exception as exc:
            raise ToolExecutionError(str(exc), code="todo_store_error") from exc
        return {"tasks": tasks, "count": len(tasks)}


class TodoUpdateTool(BaseTool):
    name = "todo_update"
    description = "Update a todo task by task_id. Requires explicit confirmation."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "updates": {"type": "object"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["task_id", "updates"],
    }

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        _require_confirmation(arguments, "update_task", context)
        store = _require_todo_store(context)
        try:
            task = await asyncio.to_thread(
                store.update_task,
                arguments["task_id"],
                arguments.get("updates", {}),
                context.trace_id,
                context.session_id,
            )
        except Exception as exc:
            raise ToolExecutionError(str(exc), code="todo_store_error") from exc
        return {"task": task}


class TodoCompleteTool(BaseTool):
    name = "todo_complete"
    description = "Mark a task as done. Requires explicit confirmation."
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "completion_note": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["task_id"],
    }

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        _require_confirmation(arguments, "complete_task", context)
        store = _require_todo_store(context)
        try:
            task = await asyncio.to_thread(
                store.complete_task,
                arguments["task_id"],
                arguments.get("completion_note", ""),
                context.trace_id,
                context.session_id,
            )
        except Exception as exc:
            raise ToolExecutionError(str(exc), code="todo_store_error") from exc
        return {"task": task}
