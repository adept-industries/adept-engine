import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine

import app.main as main_module


@pytest.mark.integration
def test_ready_requires_flyway_v7(
    database_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "get_database_engine", lambda: database_engine)

    response = TestClient(main_module.app).get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "READY", "schemaVersion": "7"}


def test_ready_returns_503_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = create_engine(
        "postgresql+psycopg://adept:adept@127.0.0.1:1/unavailable",
        connect_args={"connect_timeout": 1},
    )
    monkeypatch.setattr(main_module, "get_database_engine", lambda: unavailable)

    response = TestClient(main_module.app).get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "database or Flyway schema is not ready"}
    unavailable.dispose()
