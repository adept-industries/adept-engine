from unittest.mock import MagicMock
from uuid import uuid4

from app.normalization.jira_issues import upsert_jira_issue


def test_jira_issue_without_provider_id_is_not_persisted() -> None:
    database_engine = MagicMock()

    upsert_jira_issue(
        database_engine,
        uuid4(),
        uuid4(),
        {"key": "ADEPT-1", "fields": []},
    )

    database_engine.begin.assert_not_called()


def test_jira_issue_allows_nullable_nested_fields() -> None:
    database_engine = MagicMock()
    connection = database_engine.begin.return_value.__enter__.return_value

    upsert_jira_issue(
        database_engine,
        uuid4(),
        uuid4(),
        {
            "id": "20001",
            "key": "ADEPT-1",
            "fields": {
                "issuetype": {"name": "Bug"},
                "status": {"name": "Open"},
                "priority": None,
                "summary": "Nullable provider field",
            },
        },
    )

    parameters = connection.execute.call_args.args[1]
    assert parameters["issue_type"] == "Bug"
    assert parameters["status_name"] == "Open"
    assert parameters["priority_name"] is None
