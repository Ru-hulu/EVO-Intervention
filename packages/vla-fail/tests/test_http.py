import pytest

# Optional HTTP dependencies must be checked before importing the server.
# ruff: noqa: E402

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from vla_fail import ACCDetector
from vla_fail import LLMDDetector
from vla_fail.server import create_app


@pytest.fixture
def client():
    model = LLMDDetector().fit([[[1, 8], [4, 3]], [[1, 12], [4, 7]], [[3, 8], [6, 3]], [[3, 12], [6, 7]]])
    model.threshold = 10
    return TestClient(create_app(llmd=model, acc=ACCDetector(v_min=0.01, threshold=0.5)))


def test_llmd_endpoint_matches_hand_example(client):
    response = client.post("/v1/llmd/score", json={"tokens": [[3, 12], [8, 9]]})
    assert response.status_code == 200
    assert response.json()["score"] == pytest.approx(13, abs=2e-5)
    assert response.json()["triggered"] is True


def test_stateless_acc_does_not_mix_clients(client):
    request = {
        "previous_positions": [[0.1, 0.2, 0.1], [0.2, 0.2, 0.1], [0.3, 0.2, 0.1]],
        "current_positions": [[0.1, 0.24, 0.1], [0.2, 0.24, 0.1], [0.3, 0.24, 0.1]],
        "executed_steps": 1,
        "previous_smoothed_score": 0.02,
    }
    first = client.post("/v1/acc/score", json=request)
    assert first.status_code == 200
    second = client.post("/v1/acc/score", json=request)
    assert second.json() == first.json()
    del request["previous_smoothed_score"]
    assert client.post("/v1/acc/score", json=request).status_code == 422


def test_joint_warmup_is_explicit(client):
    response = client.post(
        "/v1/intervention/evaluate", json={"tokens": [[2, 10], [5, 5]], "positions": [[0, 0, 0], [1, 0, 0]]}
    )
    assert response.status_code == 200
    assert response.json()["need_intervention"] is None
    assert response.json()["unavailable"] == ["acc_history_missing"]


def test_llmd_only_http():
    model = LLMDDetector().fit([[[0, 0]], [[1, 1]]])
    client = TestClient(create_app(llmd=model))
    response = client.post("/v1/intervention/evaluate", json={"tokens": [[0, 0]]})
    assert response.status_code == 200
    assert response.json()["need_intervention"] is None
    assert client.get("/health").json()["acc_enabled"] is False


@pytest.mark.parametrize("tokens", [[], [[1]], [[1, 2, 3]], [[1, 2], [3]]])
def test_bad_shapes_return_422(client, tokens):
    response = client.post("/v1/llmd/score", json={"tokens": tokens})
    assert response.status_code == 422


def test_missing_inputs_and_unknown_fields_return_422(client):
    assert client.post("/v1/intervention/evaluate", json={"tokens": [[2, 10], [5, 5]]}).status_code == 422
    assert client.post("/v1/llmd/score", json={"tokens": [[2, 10], [5, 5]], "task": "cup"}).status_code == 422


def test_unfitted_model_returns_409():
    client = TestClient(create_app(llmd=LLMDDetector()))
    assert client.post("/v1/llmd/score", json={"tokens": [[0, 0]]}).status_code == 409
