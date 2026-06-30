"""Tests for main_websocket.py: FastAPI routes, HTTP endpoints."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from unmute.main_websocket import (
    _check_server_status,
    _ws_to_http,
    app,
    general_exception_handler,
)


class TestWsToHttp:
    def test_ws_to_http(self):
        assert _ws_to_http("ws://localhost:8080") == "http://localhost:8080"

    def test_wss_to_https(self):
        assert _ws_to_http("wss://example.com") == "https://example.com"


class TestCheckServerStatus:
    def test_server_up(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("unmute.main_websocket.requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            result = _check_server_status("http://localhost:8080")
            assert result is True

    def test_server_down(self):
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("unmute.main_websocket.requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            result = _check_server_status("http://localhost:8080")
            assert result is False

    def test_server_unreachable(self):
        with patch("unmute.main_websocket.requests") as mock_requests:
            mock_requests.exceptions = type('Ex', (), {'RequestException': Exception})
            mock_requests.get.side_effect = mock_requests.exceptions.RequestException("Connection refused")
            result = _check_server_status("http://localhost:8080")
            assert result is False


class TestHttpRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    @pytest.mark.asyncio
    async def test_health_all_up(self, client):
        with patch("unmute.main_websocket._get_health") as mock_health:
            from unmute.main_websocket import HealthStatus

            async def fake_health(_):
                return HealthStatus(tts_up=True, stt_up=True, llm_up=True)

            mock_health.side_effect = fake_health
            response = client.get("/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert data["tts_up"] is True

    @pytest.mark.asyncio
    async def test_health_unhealthy(self, client):
        with patch("unmute.main_websocket._get_health") as mock_health:
            from unmute.main_websocket import HealthStatus

            async def fake_health(_):
                return HealthStatus(tts_up=False, stt_up=True, llm_up=True)

            mock_health.side_effect = fake_health
            response = client.get("/v1/health")
            data = response.json()
            assert data["ok"] is False

    def test_voices_returns_list(self, client):
        response = client.get("/v1/voices")
        assert response.status_code == 200
        voices_data = response.json()
        assert isinstance(voices_data, list)

    def test_voices_all_good(self, client):
        response = client.get("/v1/voices")
        voices_data = response.json()
        for voice in voices_data:
            assert "name" in voice or "source" in voice


@pytest.mark.asyncio
class TestExceptionHandlers:
    async def test_general_exception_handler(self):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
        request = Request(scope)
        exc = RuntimeError("Something broke")
        response = await general_exception_handler(request, exc)
        assert response.status_code == 500
        body = json.loads(response.body)
        assert body["detail"] == "Internal server error"
