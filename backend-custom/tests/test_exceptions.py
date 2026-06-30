"""Tests for exceptions.py."""

import unmute.openai_realtime_api_events as ora

from unmute.exceptions import (
    MissingServiceAtCapacity,
    MissingServiceTimeout,
    WebSocketClosedError,
    make_ora_error,
)


class TestMissingServiceAtCapacity:
    def test_init(self):
        exc = MissingServiceAtCapacity("stt")
        assert exc.service == "stt"
        assert "stt" in str(exc)

    def test_is_exception(self):
        exc = MissingServiceAtCapacity("tts")
        assert exc.service == "tts"
        assert isinstance(exc, Exception)


class TestMissingServiceTimeout:
    def test_init(self):
        exc = MissingServiceTimeout("tts")
        assert exc.service == "tts"
        assert "tts" in str(exc)

    def test_is_exception(self):
        exc = MissingServiceTimeout("llm")
        assert exc.service == "llm"
        assert isinstance(exc, Exception)


class TestWebSocketClosedError:
    def test_init(self):
        exc = WebSocketClosedError()
        assert isinstance(exc, Exception)


class TestMakeOraError:
    def test_simple_error(self):
        error = make_ora_error(type="fatal", message="Something broke")
        assert isinstance(error, ora.Error)
        assert error.error.type == "fatal"
        assert error.error.message == "Something broke"
        assert error.error.code is None
        assert error.error.param is None

    def test_warning_error(self):
        error = make_ora_error(type="warning", message="Slow service")
        assert error.error.type == "warning"
        assert error.error.message == "Slow service"
