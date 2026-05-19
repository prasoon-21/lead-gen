from collections import defaultdict
from typing import Dict, List, Optional


class MemoryStore:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._memory: Dict[str, List[dict]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._memory[session_id].append({"role": role, "content": content})
        max_messages = self.max_turns * 2
        if len(self._memory[session_id]) > max_messages:
            self._memory[session_id] = self._memory[session_id][-max_messages:]

    def get_context(self, session_id: str, limit: Optional[int] = None) -> str:
        messages = self._memory.get(session_id, [])
        if not messages:
            return ""
        max_messages = (limit or self.max_turns) * 2
        messages = messages[-max_messages:]

        lines = ["--- Conversation Memory ---"]
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        lines.append("--- End Memory ---")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        if session_id in self._memory:
            del self._memory[session_id]
