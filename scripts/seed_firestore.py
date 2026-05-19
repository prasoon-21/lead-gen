"""
Seed script — pushes standard library agents and workflows from YAML files
into Firestore so the DB becomes the single source of truth at runtime.

Run once (or after adding new standard agents/workflows):
    python scripts/seed_firestore.py

Flags:
    --force     Overwrite existing records (default: skip if already exists)
    --dry-run   Print what would be seeded without writing to Firestore

Records are tagged with  is_standard=True  so the admin UI can distinguish
standard library entries from business-specific custom ones.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def seed(force: bool = False, dry_run: bool = False):
    from config.loader import ConfigLoader
    from core.db.firestore_client import FirestoreClient
    from core.db.agent_store import AgentStore
    from core.db.workflow_store import WorkflowStore

    business_id = os.getenv("FIRESTORE_BUSINESS_ID", "aurika-agentic-core")
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Seeding Firestore for business: {business_id}\n")

    loader = ConfigLoader()
    client = FirestoreClient(business_id=business_id)
    agent_store = AgentStore(client)
    workflow_store = WorkflowStore(client)

    # ── Seed agents ───────────────────────────────────────────────────────────
    print("── Agents ──────────────────────────────────────────────────────")
    agent_specs = loader.list_agent_specs()

    for meta in agent_specs:
        agent_id = meta["agent_id"]
        try:
            spec = loader.load_agent_spec(agent_id)
        except Exception as exc:
            print(f"  ✗ SKIP  {agent_id}  (load error: {exc})")
            continue

        # Inline the prompt text so the DB record is self-contained
        system_prompt = spec.system_prompt or ""
        if not system_prompt and spec.system_prompt_path:
            try:
                system_prompt = loader.load_prompt(spec.system_prompt_path)
            except FileNotFoundError:
                pass

        record = AgentStore.from_yaml_spec({
            "agent_id": spec.agent_id,
            "version": spec.version,
            "description": spec.description,
            "model_profile": spec.model_profile,
            "system_prompt": system_prompt,
            "response_mode": spec.response_mode,
            "output_schema": spec.output_schema,
            "allowed_tools": spec.allowed_tools,
            "allowed_sub_agents": spec.allowed_sub_agents,
            "memory_window": spec.memory_window,
            "max_steps": spec.max_steps,
            "max_tool_calls": spec.max_tool_calls,
            "max_runtime_seconds": spec.max_runtime_seconds,
            "default_context": spec.default_context,
            "metadata": spec.metadata,
            "is_standard": True,
        })

        if not dry_run:
            exists = await agent_store.exists(agent_id)
            if exists and not force:
                print(f"  ○ SKIP  {agent_id}  (already in DB, use --force to overwrite)")
                continue
            await agent_store.save(agent_id, record)

        action = "WRITE" if not dry_run else "WOULD WRITE"
        print(f"  ✓ {action}  {agent_id}")

    # ── Seed workflows ────────────────────────────────────────────────────────
    print("\n── Workflows ───────────────────────────────────────────────────")
    workflow_specs = loader.list_workflow_specs()

    for meta in workflow_specs:
        workflow_id = meta["workflow_id"]
        try:
            spec = loader.load_workflow_spec(workflow_id)
        except Exception as exc:
            print(f"  ✗ SKIP  {workflow_id}  (load error: {exc})")
            continue

        record = WorkflowStore.from_yaml_spec({
            "workflow_id": spec.workflow_id,
            "version": spec.version,
            "description": spec.description,
            "start_at": spec.start_at,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "config": n.config,
                    "input_map": n.input_map,
                    "output_map": n.output_map,
                    "transitions": [
                        {"to": t.to, "when": t.when} for t in n.transitions
                    ],
                }
                for n in spec.nodes
            ],
            "metadata": spec.metadata,
            "is_standard": True,
        })

        if not dry_run:
            exists = await workflow_store.exists(workflow_id)
            if exists and not force:
                print(f"  ○ SKIP  {workflow_id}  (already in DB, use --force to overwrite)")
                continue
            await workflow_store.save(workflow_id, record)

        action = "WRITE" if not dry_run else "WOULD WRITE"
        print(f"  ✓ {action}  {workflow_id}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Firestore with standard agents and workflows")
    parser.add_argument("--force", action="store_true", help="Overwrite existing records")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    args = parser.parse_args()

    asyncio.run(seed(force=args.force, dry_run=args.dry_run))
