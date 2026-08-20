from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.jobs.handlers import github_event


@pytest.mark.parametrize(
    ("event_type", "deployment_signal", "workflow_calls", "deployment_calls"),
    [
        ("workflow_run", "WORKFLOW_RUN", 1, 0),
        ("workflow_run", "DEPLOYMENT", 0, 0),
        ("deployment_status", "DEPLOYMENT", 0, 1),
        ("deployment_status", "WORKFLOW_RUN", 0, 0),
    ],
)
def test_live_deployment_events_follow_repository_signal(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    deployment_signal: str,
    workflow_calls: int,
    deployment_calls: int,
) -> None:
    workflow = MagicMock(return_value=uuid4())
    deployment = MagicMock(return_value=uuid4())
    monkeypatch.setattr(
        github_event.deployment_normalizer,
        "upsert_deployment_from_workflow_run",
        workflow,
    )
    monkeypatch.setattr(
        github_event.deployment_normalizer,
        "upsert_deployment_from_deployment_status",
        deployment,
    )

    payload = (
        {
            "action": "completed",
            "workflow_run": {"conclusion": "success"},
        }
        if event_type == "workflow_run"
        else {"deployment": {}, "deployment_status": {"state": "success"}}
    )
    github_event._dispatch(
        MagicMock(),
        event_type,
        "completed",
        payload,
        uuid4(),
        uuid4(),
        deployment_signal,
        MagicMock(),
    )

    assert workflow.call_count == workflow_calls
    assert deployment.call_count == deployment_calls
