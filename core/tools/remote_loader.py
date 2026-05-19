"""Loads remote tool definitions from config/remote_tools/*.yaml and registers
them as RemoteToolWrapper instances so any agent can call external project APIs
as if they were local tools.

Each YAML file in config/remote_tools/ defines one remote tool:

    tool_name: create_product
    description: Creates a new product in Aurika OS

    # Option A — full URL hardcoded (with optional env override):
    endpoint_url: https://aurika-os.com/tools/create_product
    endpoint_url_env: AURIKA_FULL_URL_ENV   # replaces endpoint_url entirely when set

    # Option B — base URL from env + path from YAML (recommended):
    base_url_env: AURIKA_OS_BASE_API_URL    # e.g. http://localhost:3000
    endpoint_url: /api/internal/tasks       # path appended to the base URL

    auth_header_name: X-Internal-Key     # optional
    auth_header_env: AURIKA_OS_API_KEY   # env var that holds the raw token (no "Bearer " prefix)
    auth_header_prefix: "Bearer "        # optional prefix prepended to the token
    timeout_seconds: 30
    request_body_mode: wrapped           # or raw_arguments
    static_headers:
      X-Aurika-Project: aurika_os
    argument_headers:
      x-aurika-tenant-id: tenant_id
    omit_arguments_from_body:
      - tenant_id
    input_schema:
      type: object
      properties:
        name:
          type: string
          description: Product name
        price:
          type: number
          description: Product price in INR
      required:
        - name
        - price
"""

from pathlib import Path
import os
from typing import TYPE_CHECKING, List, Optional

import yaml

from core.tools.external.remote_tool import RemoteToolWrapper

if TYPE_CHECKING:
    from core.db.tool_store import ToolStore


def _spec_to_tool(spec: dict) -> Optional[RemoteToolWrapper]:
    """Build a RemoteToolWrapper from a spec dict (YAML or Firestore document)."""
    endpoint_url = spec.get("endpoint_url")
    endpoint_url_env = spec.get("endpoint_url_env")
    base_url_env = spec.get("base_url_env")

    if endpoint_url_env:
        # Full URL override from env (replaces endpoint_url entirely)
        endpoint_url = os.getenv(endpoint_url_env, endpoint_url)
    elif base_url_env:
        # Base URL from env + path from endpoint_url in the YAML
        base = os.getenv(base_url_env, "").rstrip("/")
        if base:
            endpoint_url = base + "/" + (endpoint_url or "").lstrip("/")

    if not spec or not spec.get("tool_name") or not endpoint_url:
        return None
    return RemoteToolWrapper(
        name=spec["tool_name"],
        description=spec.get("description", ""),
        endpoint_url=endpoint_url,
        input_schema=spec.get("input_schema", {"type": "object", "properties": {}}),
        auth_header_name=spec.get("auth_header_name"),
        auth_header_env=spec.get("auth_header_env"),
        auth_header_prefix=spec.get("auth_header_prefix", ""),
        timeout_seconds=int(spec.get("timeout_seconds", 30)),
        method=spec.get("method", "POST"),
        request_body_mode=spec.get("request_body_mode", "wrapped"),
        static_headers=spec.get("static_headers"),
        argument_headers=spec.get("argument_headers"),
        omit_arguments_from_body=spec.get("omit_arguments_from_body"),
    )


def load_remote_tools(remote_tools_dir: str = "config/remote_tools") -> List[RemoteToolWrapper]:
    """Discover and instantiate all remote tools from YAML files in remote_tools_dir.

    Files prefixed with '_' (e.g. _example.yaml) are treated as documentation
    templates and are skipped.
    """
    tools: List[RemoteToolWrapper] = []
    directory = Path(remote_tools_dir)

    if not directory.exists():
        return tools

    for yaml_file in sorted(directory.glob("*.yaml")):
        if yaml_file.name.startswith("_"):
            continue  # skip example / template files

        try:
            with open(yaml_file, "r") as fh:
                spec = yaml.safe_load(fh)
            tool = _spec_to_tool(spec)
            if tool:
                tools.append(tool)
                print(f"  Registered remote tool (file): {tool.name}  →  {tool.endpoint_url}")
            else:
                print(f"  Skipping remote tool file (missing tool_name or endpoint_url): {yaml_file.name}")
        except Exception as exc:
            print(f"  Warning: Failed to load remote tool from {yaml_file.name}: {exc}")

    return tools


async def load_remote_tools_from_db(tool_store: "ToolStore") -> List[RemoteToolWrapper]:
    """Load remote tool registrations from Firestore and return as RemoteToolWrapper list.

    This supplements (does not replace) file-based tools.  Called at startup
    after Firestore is ready, results are registered into the ToolRegistry.
    """
    tools: List[RemoteToolWrapper] = []
    try:
        records = await tool_store.list_all()
        for spec in records:
            tool = _spec_to_tool(spec)
            if tool:
                tools.append(tool)
                print(f"  Registered remote tool (db): {tool.name}  →  {tool.endpoint_url}")
    except Exception as exc:
        print(f"  Warning: Firestore remote tool load failed: {exc}")
    return tools
