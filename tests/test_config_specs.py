from config.loader import ConfigLoader


def test_load_agent_spec():
    loader = ConfigLoader()
    spec = loader.load_agent_spec("research_assistant")
    assert spec.agent_id == "research_assistant"
    assert "web_search" in spec.allowed_tools


def test_load_workflow_spec():
    loader = ConfigLoader()
    spec = loader.load_workflow_spec("research_pipeline")
    assert spec.workflow_id == "research_pipeline"
    assert spec.start_at == "research_agent"
    assert spec.get_node("research_agent") is not None


def test_load_todo_agent_and_workflow_spec():
    loader = ConfigLoader()
    agent_spec = loader.load_agent_spec("task_manager_agent")
    workflow_spec = loader.load_workflow_spec("task_manager_pipeline")
    assert agent_spec.agent_id == "task_manager_agent"
    assert "todo_create" in agent_spec.allowed_tools
    assert "aurika_tasks_create_task" in agent_spec.allowed_tools
    assert workflow_spec.start_at == "ensure_todo_store"


def test_load_classification_profile():
    loader = ConfigLoader()
    profile = loader.load_classification_profile("email_auto_ticketing_v1")
    assert profile.profile_id == "email_auto_ticketing_v1"
    assert profile.get_dimension("request_type") is not None
