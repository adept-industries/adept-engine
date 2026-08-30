from types import SimpleNamespace
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


def test_open_pull_request_event_scores_current_provider_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_id = uuid4()
    repository_id = uuid4()
    pull_request_id = uuid4()
    current_pull_request = {
        "id": 100,
        "number": 42,
        "state": "open",
        "changed_files": 1,
        "additions": 8,
        "deletions": 2,
    }
    commits = [{"sha": "abc", "commit": {"message": "Fix race"}}]
    files = [{"filename": "app/main.py", "additions": 8, "deletions": 2}]
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_pull_request.return_value = current_pull_request
    client.list_pull_request_commits.return_value = commits
    client.list_pull_request_files.return_value = files
    score = MagicMock()

    monkeypatch.setattr(github_event, "GithubClient", MagicMock(return_value=client))
    monkeypatch.setattr(
        github_event,
        "load_github_repository",
        MagicMock(
            return_value=SimpleNamespace(
                installation_id=99,
                owner_login="adept-industries",
                name="adept-engine",
            )
        ),
    )
    monkeypatch.setattr(
        github_event.pr_normalizer,
        "upsert_pull_request",
        MagicMock(return_value=pull_request_id),
    )
    monkeypatch.setattr(github_event, "calculate_and_persist_pull_request_risk", score)
    monkeypatch.setattr(github_event, "_pull_request_is_merged", MagicMock(return_value=False))

    github_event._handle_pull_request(
        MagicMock(),
        {"pull_request": {"number": 42, "state": "open", "additions": 1}},
        "synchronize",
        workspace_id,
        repository_id,
        MagicMock(),
    )

    client.get_pull_request.assert_called_once_with("adept-industries", "adept-engine", 42)
    client.list_pull_request_commits.assert_called_once_with("adept-industries", "adept-engine", 42)
    client.list_pull_request_files.assert_called_once_with("adept-industries", "adept-engine", 42)
    score.assert_called_once()
    assert score.call_args.args[4:] == (current_pull_request, files, commits)


def test_closed_pull_request_event_normalizes_without_rescoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_pull_request.return_value = {
        "id": 100,
        "number": 42,
        "state": "closed",
        "merged": False,
    }
    client.list_pull_request_commits.return_value = []
    score = MagicMock()

    monkeypatch.setattr(github_event, "GithubClient", MagicMock(return_value=client))
    monkeypatch.setattr(
        github_event,
        "load_github_repository",
        MagicMock(
            return_value=SimpleNamespace(
                installation_id=99,
                owner_login="adept-industries",
                name="adept-engine",
            )
        ),
    )
    monkeypatch.setattr(
        github_event.pr_normalizer,
        "upsert_pull_request",
        MagicMock(return_value=uuid4()),
    )
    monkeypatch.setattr(github_event, "calculate_and_persist_pull_request_risk", score)
    monkeypatch.setattr(github_event, "_pull_request_is_merged", MagicMock(return_value=False))

    github_event._handle_pull_request(
        MagicMock(),
        {"pull_request": {"number": 42}},
        "closed",
        uuid4(),
        uuid4(),
        MagicMock(),
    )

    client.list_pull_request_files.assert_not_called()
    score.assert_not_called()


def test_issue_event_routes_to_issue_normalizer(monkeypatch: pytest.MonkeyPatch) -> None:
    issue_id = uuid4()
    normalize = MagicMock(return_value=issue_id)
    monkeypatch.setattr(github_event.issue_normalizer, "upsert_github_issue", normalize)

    workspace_id = uuid4()
    repository_id = uuid4()
    database_engine = MagicMock()
    payload = {"issue": {"id": 100, "number": 7, "title": "Broken build"}}
    github_event._dispatch(
        database_engine,
        "issues",
        "opened",
        payload,
        workspace_id,
        repository_id,
        None,
        MagicMock(),
    )

    normalize.assert_called_once_with(
        database_engine,
        workspace_id,
        repository_id,
        payload["issue"],
    )
