import base64
import json

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import SecretStr

from app.core.config import Settings
from app.providers import ProviderError, ProviderPermanentError
from app.providers.github import GithubClient, _generate_app_jwt
from app.providers.jira import JiraClient, refresh_oauth_token


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"postgres_password": SecretStr("test")}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _github_settings() -> Settings:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return _settings(
        github_app_id="1234",
        github_app_private_key_base64=SecretStr(base64.b64encode(pem).decode()),
    )


def test_github_app_jwt_contains_signed_header_payload_and_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    settings = _settings(
        github_app_id="1234",
        github_app_private_key_base64=SecretStr(base64.b64encode(pem).decode()),
    )
    monkeypatch.setattr("app.providers.github.time.time", lambda: 1_800_000_000.0)

    token = _generate_app_jwt(settings)
    header_segment, payload_segment, signature_segment = token.split(".")

    def decode_segment(segment: str) -> bytes:
        return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))

    assert json.loads(decode_segment(header_segment)) == {"alg": "RS256", "typ": "JWT"}
    assert json.loads(decode_segment(payload_segment)) == {
        "iat": 1_799_999_940,
        "exp": 1_800_000_540,
        "iss": "1234",
    }
    private_key.public_key().verify(
        decode_segment(signature_segment),
        f"{header_segment}.{payload_segment}".encode("ascii"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_github_client_authenticates_and_pages_installation_repositories() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        assert request.headers["Authorization"] == "Bearer installation-token"
        return httpx.Response(
            200,
            json={
                "total_count": 101,
                "repositories": [{"id": number} for number in range(100)],
            },
        )

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with GithubClient(_github_settings(), 99, http_client=http_client) as client:
        result = client.list_installation_repositories(1)

    assert len(result.items) == 100
    assert result.next_page == 2
    assert requests[0].url.path == "/app/installations/99/access_tokens"
    assert requests[1].url.params["page"] == "1"


def test_github_workflow_backfill_can_query_all_branches() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(200, json={"total_count": 0, "workflow_runs": []})

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with GithubClient(_github_settings(), 99, http_client=http_client) as client:
        client.list_workflow_runs(
            "adept-industries",
            "adept-engine",
            1,
            branch=None,
            created_from="2026-08-01T00:00:00Z",
            created_to="2026-08-02T00:00:00Z",
        )

    assert requests[1].url.path == "/repos/adept-industries/adept-engine/actions/runs"
    assert "branch" not in requests[1].url.params


def test_github_client_collects_all_pull_request_file_pages() -> None:
    file_pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        assert request.url.path == "/repos/adept-industries/adept-engine/pulls/42/files"
        page = request.url.params["page"]
        file_pages.append(page)
        count = 100 if page == "1" else 2
        return httpx.Response(
            200,
            json=[{"filename": f"src/file-{page}-{index}.py"} for index in range(count)],
        )

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with GithubClient(_github_settings(), 99, http_client=http_client) as client:
        files = client.list_pull_request_files("adept-industries", "adept-engine", 42)

    assert len(files) == 102
    assert file_pages == ["1", "2"]


def test_github_client_lists_open_issues_with_stable_pagination() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(200, json=[{"id": number} for number in range(100)])

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with GithubClient(_github_settings(), 99, http_client=http_client) as client:
        result = client.list_open_issues("adept-industries", "adept-engine", 2)

    request = requests[1]
    assert request.url.path == "/repos/adept-industries/adept-engine/issues"
    assert request.url.params["state"] == "open"
    assert request.url.params["page"] == "2"
    assert result.next_page == 3


def test_github_rate_limit_preserves_bounded_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(429, headers={"Retry-After": "1200"})

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with (
        GithubClient(_github_settings(), 99, http_client=http_client) as client,
        pytest.raises(ProviderError) as raised,
    ):
        client.list_installation_repositories(1)

    assert raised.value.retry_after_seconds == 900


def test_github_primary_rate_limit_uses_reset_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.github.time.time", lambda: 1_000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(
            403,
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "2200",
            },
        )

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with (
        GithubClient(_github_settings(), 99, http_client=http_client) as client,
        pytest.raises(ProviderError) as raised,
    ):
        client.list_installation_repositories(1)

    assert raised.value.retry_after_seconds == 900


def test_github_permission_forbidden_remains_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(403)

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with (
        GithubClient(_github_settings(), 99, http_client=http_client) as client,
        pytest.raises(ProviderPermanentError),
    ):
        client.list_installation_repositories(1)


def test_jira_client_uses_cloud_api_path_and_paginates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer access-token"
        assert request.url.path == "/ex/jira/cloud-1/rest/api/3/project/search"
        return httpx.Response(
            200,
            json={
                "values": [{"id": "100", "key": "ADEPT", "name": "Adept"}],
                "isLast": False,
                "total": 2,
            },
        )

    http_client = httpx.Client(
        base_url="https://api.atlassian.com", transport=httpx.MockTransport(handler)
    )
    with JiraClient(_settings(), "cloud-1", "access-token", http_client=http_client) as client:
        result = client.list_projects(0, max_results=1)

    assert result.items[0]["key"] == "ADEPT"
    assert result.next_page == 1


def test_jira_client_searches_unresolved_project_issues_with_token_pagination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer access-token"
        assert request.url.path == "/ex/jira/cloud-1/rest/api/3/search/jql"
        assert request.url.params["jql"] == (
            'project = "ADEPT" AND resolution IS EMPTY ORDER BY updated DESC'
        )
        assert request.url.params["nextPageToken"] == "next-token"
        assert "summary" in request.url.params["fields"]
        return httpx.Response(
            200,
            json={"issues": [{"id": "10001", "key": "ADEPT-1"}], "nextPageToken": "last"},
        )

    http_client = httpx.Client(
        base_url="https://api.atlassian.com", transport=httpx.MockTransport(handler)
    )
    with JiraClient(_settings(), "cloud-1", "access-token", http_client=http_client) as client:
        result = client.list_unresolved_issues("ADEPT", "next-token")

    assert result.items[0]["key"] == "ADEPT-1"
    assert result.next_page_token == "last"


def test_jira_refresh_rotates_refresh_token_and_scopes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "read:jira-work offline_access",
            },
        )

    client = httpx.Client(
        base_url="https://auth.atlassian.com", transport=httpx.MockTransport(handler)
    )
    result = refresh_oauth_token(
        _settings(jira_client_id="client", jira_client_secret=SecretStr("secret")),
        "old-refresh",
        http_client=client,
    )

    assert result.access_token == "new-access"
    assert result.refresh_token == "new-refresh"
    assert result.scopes == ["read:jira-work", "offline_access"]


def test_jira_webhook_exists_paginates_until_stored_id_is_found() -> None:
    starts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(request.url.params["startAt"])
        if request.url.params["startAt"] == "0":
            return httpx.Response(
                200,
                json={"values": [{"id": 11}], "isLast": False, "maxResults": 1},
            )
        return httpx.Response(
            200,
            json={"values": [{"id": 991}], "isLast": True, "maxResults": 1},
        )

    http_client = httpx.Client(
        base_url="https://api.atlassian.com", transport=httpx.MockTransport(handler)
    )
    with JiraClient(_settings(), "cloud-1", "access-token", http_client=http_client) as client:
        assert client.webhook_exists(991) is True

    assert starts == ["0", "1"]


def test_jira_rejects_non_retryable_provider_response() -> None:
    client = httpx.Client(
        base_url="https://api.atlassian.com",
        transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
    )
    with (
        JiraClient(_settings(), "cloud-1", "bad-token", http_client=client) as jira,
        pytest.raises(ProviderPermanentError),
    ):
        jira.list_projects(0)
