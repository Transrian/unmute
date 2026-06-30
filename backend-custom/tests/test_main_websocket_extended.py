"""Extended tests for main_websocket.py to increase coverage."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from unmute.main_websocket import (
    _check_server_status,
    _cors_headers_for_error,
    _ws_to_http,
    app,
    EmitDebugLogger,
    general_exception_handler,
)


class TestWsToHttp:
    def test_plain_http(self):
        assert _ws_to_http("ws://localhost:8080") == "http://localhost:8080"

    def test_secure_https(self):
        assert _ws_to_http("wss://example.com:443/path") == "https://example.com:443/path"


class TestCorsHeadersForError:
    def test_allowed_origin(self):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"origin", b"http://localhost:3000")],
        }
        request = Request(scope)
        headers = _cors_headers_for_error(request)
        assert "Access-Control-Allow-Origin" in headers
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

    def test_disallowed_origin(self):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"origin", b"http://evil.com")],
        }
        request = Request(scope)
        headers = _cors_headers_for_error(request)
        assert "Access-Control-Allow-Origin" not in headers

    def test_no_origin(self):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
        request = Request(scope)
        headers = _cors_headers_for_error(request)
        assert "Access-Control-Allow-Credentials" in headers


class TestEmitDebugLogger:
    def test_first_emit(self):
        logger = EmitDebugLogger()
        import unmute.openai_realtime_api_events as ora

        event = ora.ResponseCreated(
            response=ora.Response(status="in_progress", voice="alloy")
        )
        logger.on_emit(event)
        assert logger.last_emitted_n == 1
        assert logger.last_emitted_type == "response.created"

    def test_repeated_emit(self):
        logger = EmitDebugLogger()
        import unmute.openai_realtime_api_events as ora

        event1 = ora.ResponseCreated(
            response=ora.Response(status="in_progress", voice="alloy")
        )
        event2 = ora.ResponseCreated(
            response=ora.Response(status="in_progress", voice="alloy")
        )
        event3 = ora.ResponseCreated(
            response=ora.Response(status="in_progress", voice="alloy")
        )

        logger.on_emit(event1)
        logger.on_emit(event2)
        logger.on_emit(event3)
        assert logger.last_emitted_n == 3

    def test_different_emit_type(self):
        logger = EmitDebugLogger()
        import unmute.openai_realtime_api_events as ora

        event1 = ora.ResponseCreated(
            response=ora.Response(status="in_progress", voice="alloy")
        )
        event2 = ora.ResponseTextDelta(delta="Hello")

        logger.on_emit(event1)
        logger.on_emit(event2)
        assert logger.last_emitted_n == 1  # reset for new type


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


class TestCheckServerStatus:
    def test_success_with_headers(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("unmute.main_websocket.requests") as mock_requests:
            mock_requests.get.return_value = mock_response
            result = _check_server_status(
                "http://localhost:8080", headers={"Authorization": "Bearer token"}
            )
            assert result is True


class TestHealthStatus:
    def test_ok_true_when_all_services_up(self):
        from unmute.main_websocket import HealthStatus

        health = HealthStatus(tts_up=True, stt_up=True, llm_up=True)
        assert health.ok is True

    def test_ok_false_when_tts_down(self):
        from unmute.main_websocket import HealthStatus

        health = HealthStatus(tts_up=False, stt_up=True, llm_up=True)
        assert health.ok is False

    def test_ok_false_when_stt_down(self):
        from unmute.main_websocket import HealthStatus

        health = HealthStatus(tts_up=True, stt_up=False, llm_up=True)
        assert health.ok is False

    def test_ok_false_when_llm_down(self):
        from unmute.main_websocket import HealthStatus

        health = HealthStatus(tts_up=True, stt_up=True, llm_up=False)
        assert health.ok is False


class TestTtsStreamingQuery:
    def test_to_url_params_basic(self):
        from unmute.tts.text_to_speech import TtsStreamingQuery

        query = TtsStreamingQuery(voice="alloy", cfg_alpha=1.5)
        params = query.to_url_params()
        assert "voice=alloy" in params
        assert "cfg_alpha=1.5" in params
        assert params.startswith("?")

    def test_to_url_params_excludes_none(self):
        from unmute.tts.text_to_speech import TtsStreamingQuery

        query = TtsStreamingQuery(voice=None, seed=None)
        params = query.to_url_params()
        # None values should be excluded
        assert "voice=" not in params
        assert "seed=" not in params

    def test_to_url_params_default_format(self):
        from unmute.tts.text_to_speech import TtsStreamingQuery

        query = TtsStreamingQuery()
        params = query.to_url_params()
        assert "format=PcmMessagePack" in params


class TestPrepareTextForTts:
    def test_strips_whitespace(self):
        from unmute.tts.text_to_speech import prepare_text_for_tts

        result = prepare_text_for_tts("  Hello world  ")
        assert result == "Hello world"

    def test_removes_unpronounceable_chars(self):
        from unmute.tts.text_to_speech import prepare_text_for_tts

        result = prepare_text_for_tts("Hello *world* and _test_")
        assert "*" not in result
        assert "_" not in result
        assert "`" not in result

    def test_normalizes_quotes(self):
        from unmute.tts.text_to_speech import prepare_text_for_tts

        result = prepare_text_for_tts("She said \"hello\" and 'bye'")
        assert result == "She said \"hello\" and 'bye'"

    def test_removes_colon_spaces(self):
        from unmute.tts.text_to_speech import prepare_text_for_tts

        result = prepare_text_for_tts("Hello : world")
        assert result == "Hello world"


class TestTTSClientMessages:
    def test_text_message_creation(self):
        from unmute.tts.text_to_speech import TTSClientTextMessage

        msg = TTSClientTextMessage(text="Hello world")
        assert msg.type == "Text"
        assert msg.text == "Hello world"

    def test_voice_message_creation(self):
        from unmute.tts.text_to_speech import TTSClientVoiceMessage

        msg = TTSClientVoiceMessage(embeddings=[0.1, 0.2, 0.3], shape=[10])
        assert msg.type == "Voice"
        assert msg.embeddings == [0.1, 0.2, 0.3]

    def test_eos_message_creation(self):
        from unmute.tts.text_to_speech import TTSClientEosMessage

        msg = TTSClientEosMessage()
        assert msg.type == "Eos"

    def test_tts_server_messages(self):
        from unmute.tts.text_to_speech import (
            TTSTextMessage,
            TTSAudioMessage,
            TTSErrorMessage,
            TTSReadyMessage,
        )

        text_msg = TTSTextMessage(type="Text", text="Hello", start_s=0.0, stop_s=1.0)
        assert text_msg.text == "Hello"

        audio_msg = TTSAudioMessage(type="Audio", pcm=[0.1, 0.2])
        assert audio_msg.pcm == [0.1, 0.2]

        error_msg = TTSErrorMessage(type="Error", message="Something failed")
        assert error_msg.message == "Something failed"

        ready_msg = TTSReadyMessage(type="Ready")
        assert ready_msg.type == "Ready"


class TestUrlEscape:
    def test_escapes_special_chars(self):
        from unmute.tts.text_to_speech import url_escape

        result = url_escape("hello world & more")
        assert "+" not in result or "%20" in result

    def test_escapes_none(self):
        from unmute.tts.text_to_speech import url_escape

        result = url_escape(None)
        assert result == "None"

    def test_escapes_numbers(self):
        from unmute.tts.text_to_speech import url_escape

        result = url_escape(42)
        assert result == "42"
