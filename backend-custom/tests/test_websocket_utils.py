"""Tests for websocket_utils: http_to_ws, ws_to_http."""

from unmute.websocket_utils import http_to_ws, ws_to_http


class TestHttpToWs:
    def test_http_to_ws_simple(self):
        assert http_to_ws("http://localhost:8080") == "ws://localhost:8080"

    def test_https_to_wss(self):
        assert http_to_ws("https://example.com") == "wss://example.com"

    def test_already_ws_unchanged(self):
        assert http_to_ws("ws://localhost:8080") == "ws://localhost:8080"

    def test_already_wss_unchanged(self):
        assert http_to_ws("wss://example.com") == "wss://example.com"

    def test_with_path_and_query(self):
        url = "http://stt:8080/api/asr-streaming?foo=bar"
        result = http_to_ws(url)
        assert result == "ws://stt:8080/api/asr-streaming?foo=bar"


class TestWsToHttp:
    def test_ws_to_http_simple(self):
        assert ws_to_http("ws://localhost:8080") == "http://localhost:8080"

    def test_wss_to_https(self):
        assert ws_to_http("wss://example.com") == "https://example.com"

    def test_already_http_unchanged(self):
        assert ws_to_http("http://localhost:8080") == "http://localhost:8080"

    def test_already_https_unchanged(self):
        assert ws_to_http("https://example.com") == "https://example.com"

    def test_with_path(self):
        assert (
            ws_to_http("ws://tts:8080/api/tts_streaming")
            == "http://tts:8080/api/tts_streaming"
        )

    def test_roundtrip_http(self):
        url = "http://localhost:8080/path"
        assert ws_to_http(http_to_ws(url)) == url

    def test_roundtrip_https(self):
        url = "https://example.com/path"
        assert ws_to_http(http_to_ws(url)) == url
