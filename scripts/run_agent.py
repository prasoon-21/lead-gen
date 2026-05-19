"""
Terminal test runner for any agent loaded by name from config/agents/.

Usage:
    python scripts/run_agent.py task_manager_agent "list my tasks"
    python scripts/run_agent.py task_manager_agent   # interactive REPL mode

Tool-call events are printed to the terminal in real time via the agent.runtime
logger — no API server needed.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# ── make sure project root is on sys.path ─────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── ANSI colour helpers ────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
MAGENTA = "\033[35m"
BLUE   = "\033[34m"

EVENT_COLOURS: Dict[str, str] = {
    "agent_run_start":      BOLD + CYAN,
    "agent_run_end":        BOLD + CYAN,
    "agent_prompt_loaded":  DIM + CYAN,
    "agent_decision":       BLUE,
    "agent_tool_call":      BOLD + GREEN,
    "agent_final_answer":   BOLD + GREEN,
    "agent_fallback":       BOLD + RED,
    "tool_execute_start":   YELLOW,
    "tool_execute_end":     GREEN,
    "tool_execute_timeout": RED,
    "tool_blocked":         RED,
    "tool_missing":         RED,
}


# ── custom log formatter ───────────────────────────────────────────────────────
class AgentTerminalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event     = getattr(record, "event",      record.getMessage())
        tool_name = getattr(record, "tool_name",  "-")
        step      = getattr(record, "step",       0)
        agent_id  = getattr(record, "agent_id",   "-")
        status    = getattr(record, "status",     "-")
        trace_id  = getattr(record, "trace_id",   "-")

        colour = EVENT_COLOURS.get(event, "")
        ts = time.strftime("%H:%M:%S")

        tool_part = f"  tool={BOLD}{tool_name}{RESET}" if tool_name != "-" else ""
        step_part = f"  step={step}" if step else ""

        line = (
            f"{DIM}{ts}{RESET} "
            f"{colour}{event:<24}{RESET} "
            f"agent={MAGENTA}{agent_id}{RESET}"
            f"{step_part}"
            f"{tool_part}"
            f"  {DIM}{status}{RESET}"
        )

        if event == "agent_tool_call" and tool_name != "-":
            line = (
                f"\n{DIM}{'─'*60}{RESET}\n"
                f"{GREEN}{BOLD}  TOOL CALLED: {tool_name}{RESET}\n"
                f"  step={step}  agent={agent_id}\n"
                f"  {DIM}{status}{RESET}\n"
                f"{DIM}{'─'*60}{RESET}"
            )
        elif event == "agent_final_answer":
            line = f"\n{GREEN}{BOLD}  FINAL ANSWER{RESET}  step={step}  agent={agent_id}\n"
        elif event == "agent_run_end":
            line = f"\n{CYAN}{BOLD}  RUN COMPLETE{RESET}  agent={agent_id}  {DIM}{status}{RESET}\n"

        return line


def _setup_logging(level: int = logging.DEBUG) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(AgentTerminalFormatter())
    handler.setLevel(level)

    logger = logging.getLogger("agent.runtime")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


# ── stub todo store (no Firestore needed) ─────────────────────────────────────
class _StubTodoStore:
    def __init__(self):
        self._tasks: Dict[str, Any] = {}
        self._next = 1

    def ensure_store(self):
        return {"spreadsheet_id": "local-stub"}

    def create_task(self, payload, trace_id="", session_id="", source_agent=""):
        task_id = f"TASK_{self._next:03d}"
        self._next += 1
        task = {"task_id": task_id, **payload}
        self._tasks[task_id] = task
        return task

    def list_tasks(self, filters=None):
        tasks = list(self._tasks.values())
        if filters and (status := filters.get("status")):
            tasks = [t for t in tasks if t.get("status") == status]
        return tasks

    def update_task(self, task_id, updates, trace_id="", session_id=""):
        if task_id not in self._tasks:
            return {"task_id": task_id, "error": "not found"}
        self._tasks[task_id].update(updates)
        return self._tasks[task_id]

    def complete_task(self, task_id, trace_id="", session_id=""):
        return self.update_task(task_id, {"status": "done"})


# ── runner ────────────────────────────────────────────────────────────────────
async def run_turn(agent_id: str, message: str, session_id: str) -> Dict[str, Any]:
    from config.loader import ConfigLoader
    from core.agents.factory import AgentFactory
    from core.adapters.router import build_adapter_from_env

    loader  = ConfigLoader()
    adapter = build_adapter_from_env()
    factory = AgentFactory(
        config_loader=loader,
        base_adapter=adapter,
        todo_store=_StubTodoStore(),
    )

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}  Agent : {MAGENTA}{agent_id}{RESET}")
    print(f"{BOLD}  Input : {RESET}{message}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")

    result = await factory.run(
        agent_id=agent_id,
        message=message,
        session_id=session_id,
    )

    # ── pretty-print result ────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    if result.get("text"):
        print(f"{BOLD}  Response:{RESET}\n  {result['text']}")
    elif result.get("json"):
        print(f"{BOLD}  Response (JSON):{RESET}")
        print("  " + json.dumps(result["json"], indent=2).replace("\n", "\n  "))

    meta = result.get("metadata", {})
    print(
        f"\n{DIM}  tool_calls={meta.get('tool_calls', 0)}"
        f"  completed={meta.get('completed')}"
        f"  latency={meta.get('latency_ms', 0):.0f}ms"
        f"  trace={result.get('trace_id', '-')}{RESET}"
    )

    steps = result.get("steps", [])
    tool_steps = [s for s in steps if s.get("type") == "tool_call"]
    if tool_steps:
        print(f"\n{BOLD}  Tool call log:{RESET}")
        for s in tool_steps:
            ok     = s.get("result", {}).get("ok", "?")
            colour = GREEN if ok else RED
            print(
                f"  {DIM}step {s['step']:>2}{RESET}  "
                f"{colour}{BOLD}{s['tool_name']:<28}{RESET}"
                f"  ok={ok}"
                f"  {DIM}{json.dumps(s.get('arguments', {}), ensure_ascii=False)[:80]}{RESET}"
            )
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")
    return result


def _repl(agent_id: str, session_id: str) -> None:
    print(f"\n{BOLD}Entering REPL for agent '{agent_id}' (session {session_id}){RESET}")
    print(f"{DIM}Type 'exit' or Ctrl-C to quit.{RESET}\n")
    while True:
        try:
            msg = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if not msg:
            continue
        asyncio.run(run_turn(agent_id, msg, session_id))


def main() -> None:
    _setup_logging()
    args = sys.argv[1:]
    if not args:
        print(f"Usage: python scripts/run_agent.py <agent_id> [message]")
        sys.exit(1)

    agent_id   = args[0]
    session_id = f"SESS_CLI_{int(time.time())}"

    if len(args) >= 2:
        message = " ".join(args[1:])
        asyncio.run(run_turn(agent_id, message, session_id))
    else:
        _repl(agent_id, session_id)


if __name__ == "__main__":
    main()
