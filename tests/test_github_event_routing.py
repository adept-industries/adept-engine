from datetime import UTC, datetime
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
    reviews = [{"id": 7, "state": "APPROVED", "submitted_at": "2026-09-01T10:00:00Z"}]
    client.list_pull_request_reviews.return_value = reviews
    upsert = MagicMock(return_value=pull_request_id)
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
    monkeypatch.setattr(github_event.pr_normalizer, "upsert_pull_request", upsert)
    monkeypatch.setattr(github_event, "calculate_and_persist_pull_request_risk", score)
    monkeypatch.setattr(github_event, "_pull_request_merged_at", MagicMock(return_value=None))

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
    client.list_pull_request_reviews.assert_called_once_with("adept-industries", "adept-engine", 42)
    assert upsert.call_args.args[3:] == (current_pull_request, "synchronize", commits, reviews)


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
    monkeypatch.setattr(github_event, "_pull_request_merged_at", MagicMock(return_value=None))

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


def _review_repository() -> SimpleNamespace:
    return SimpleNamespace(installation_id=99, owner_login="adept-industries", name="adept-engine")


def test_review_event_replaces_reviews_from_provider_list(monkeypatch: pytest.MonkeyPatch) -> None:
    pull_request_id = uuid4()
    reviews = [{"id": 1, "state": "DISMISSED", "submitted_at": "2026-09-01T10:00:00Z"}]
    client = MagicMock()
    client.__enter__.return_value = client
    client.list_pull_request_reviews.return_value = reviews
    replace = MagicMock()
    full_sync = MagicMock()
    database_engine = MagicMock()
    monkeypatch.setattr(github_event, "GithubClient", MagicMock(return_value=client))
    monkeypatch.setattr(
        github_event, "load_github_repository", MagicMock(return_value=_review_repository())
    )
    monkeypatch.setattr(github_event, "_pull_request_id", MagicMock(return_value=pull_request_id))
    monkeypatch.setattr(github_event.pr_normalizer, "replace_pull_request_reviews", replace)
    monkeypatch.setattr(github_event, "_handle_pull_request", full_sync)
    monkeypatch.setattr(github_event, "_pull_request_merged_at", MagicMock(return_value=None))

    github_event._dispatch(
        database_engine,
        "pull_request_review",
        "dismissed",
        {"action": "dismissed", "pull_request": {"number": 42}},
        uuid4(),
        uuid4(),
        "WORKFLOW_RUN",
        MagicMock(),
    )

    client.list_pull_request_reviews.assert_called_once_with("adept-industries", "adept-engine", 42)
    replace.assert_called_once_with(database_engine, pull_request_id, reviews)
    full_sync.assert_not_called()


def test_review_event_for_unknown_pull_request_imports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_sync = MagicMock()
    replace = MagicMock()
    monkeypatch.setattr(github_event, "_pull_request_id", MagicMock(return_value=None))
    monkeypatch.setattr(github_event, "_handle_pull_request", full_sync)
    monkeypatch.setattr(github_event.pr_normalizer, "replace_pull_request_reviews", replace)
    workspace_id = uuid4()
    repository_id = uuid4()
    webhook_pr = {"number": 42}

    github_event._dispatch(
        MagicMock(),
        "pull_request_review",
        "submitted",
        {"pull_request": webhook_pr},
        workspace_id,
        repository_id,
        "WORKFLOW_RUN",
        MagicMock(),
    )

    full_sync.assert_called_once()
    assert full_sync.call_args.args[1:5] == (
        {"pull_request": webhook_pr},
        "synchronize",
        workspace_id,
        repository_id,
    )
    replace.assert_not_called()


@pytest.mark.parametrize(
    ("event_type", "action"),
    [("pull_request_review", "requested"), ("pull_request_review_comment", "created")],
)
def test_review_events_without_cycle_time_signal_make_no_provider_calls(
    monkeypatch: pytest.MonkeyPatch, event_type: str, action: str
) -> None:
    github_client = MagicMock()
    lookup = MagicMock()
    monkeypatch.setattr(github_event, "GithubClient", github_client)
    monkeypatch.setattr(github_event, "_pull_request_id", lookup)

    github_event._dispatch(
        MagicMock(),
        event_type,
        action,
        {"pull_request": {"number": 42}},
        uuid4(),
        uuid4(),
        "WORKFLOW_RUN",
        MagicMock(),
    )

    github_client.assert_not_called()
    lookup.assert_not_called()


def test_review_change_on_merged_pull_request_queues_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged_at = datetime(2026, 9, 23, 17, tzinfo=UTC)
    client = MagicMock()
    client.__enter__.return_value = client
    client.list_pull_request_reviews.return_value = []
    enqueue = MagicMock()
    monkeypatch.setattr(github_event, "GithubClient", MagicMock(return_value=client))
    monkeypatch.setattr(
        github_event, "load_github_repository", MagicMock(return_value=_review_repository())
    )
    monkeypatch.setattr(github_event, "_pull_request_id", MagicMock(return_value=uuid4()))
    monkeypatch.setattr(github_event.pr_normalizer, "replace_pull_request_reviews", MagicMock())
    monkeypatch.setattr(github_event, "_pull_request_merged_at", MagicMock(return_value=merged_at))
    monkeypatch.setattr(github_event, "enqueue_recalculate_metrics_job", enqueue)
    workspace_id = uuid4()
    repository_id = uuid4()

    github_event._dispatch(
        MagicMock(),
        "pull_request_review",
        "submitted",
        {"pull_request": {"number": 42}},
        workspace_id,
        repository_id,
        "WORKFLOW_RUN",
        MagicMock(),
    )

    enqueue.assert_called_once()
    assert enqueue.call_args.args[1:] == (workspace_id, repository_id)
    assert enqueue.call_args.kwargs == {"affected_at": merged_at}


def test_ready_for_review_uses_the_delivery_transition_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    # The fresh read reflects a push made an hour after the PR became ready.
    client.get_pull_request.return_value = {
        "id": 100,
        "number": 42,
        "state": "closed",
        "merged": False,
        "updated_at": "2026-09-02T09:00:00Z",
    }
    client.list_pull_request_commits.return_value = []
    client.list_pull_request_reviews.return_value = []
    upsert = MagicMock(return_value=uuid4())

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
    monkeypatch.setattr(github_event.pr_normalizer, "upsert_pull_request", upsert)
    monkeypatch.setattr(github_event, "_pull_request_merged_at", MagicMock(return_value=None))

    github_event._handle_pull_request(
        MagicMock(),
        {"pull_request": {"number": 42, "updated_at": "2026-09-02T08:00:00Z"}},
        "ready_for_review",
        uuid4(),
        uuid4(),
        MagicMock(),
    )

    assert upsert.call_args.kwargs["ready_for_review_at"] == datetime(2026, 9, 2, 8, tzinfo=UTC)


def test_other_pull_request_actions_do_not_set_a_ready_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_pull_request.return_value = {"id": 100, "number": 42, "state": "closed"}
    client.list_pull_request_commits.return_value = []
    client.list_pull_request_reviews.return_value = []
    upsert = MagicMock(return_value=uuid4())

    monkeypatch.setattr(github_event, "GithubClient", MagicMock(return_value=client))
    monkeypatch.setattr(
        github_event,
        "load_github_repository",
        MagicMock(return_value=SimpleNamespace(installation_id=99, owner_login="o", name="r")),
    )
    monkeypatch.setattr(github_event.pr_normalizer, "upsert_pull_request", upsert)
    monkeypatch.setattr(github_event, "_pull_request_merged_at", MagicMock(return_value=None))

    github_event._handle_pull_request(
        MagicMock(),
        {"pull_request": {"number": 42, "updated_at": "2026-09-02T08:00:00Z"}},
        "edited",
        uuid4(),
        uuid4(),
        MagicMock(),
    )

    assert upsert.call_args.kwargs["ready_for_review_at"] is None
