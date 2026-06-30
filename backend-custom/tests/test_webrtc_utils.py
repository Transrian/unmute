"""Tests for webrtc_utils.py."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetCloudflareRTCConfiguration:
    def test_success(self):
        with (
            patch("unmute.webrtc_utils.os.environ", {"TURN_KEY_ID": "test-id", "TURN_KEY_API_TOKEN": "test-token"}),
            patch("unmute.webrtc_utils.requests.post") as mock_post,
        ):
            mock_post.return_value.ok = True
            mock_post.return_value.json.return_value = {"iceServers": []}

            from unmute.webrtc_utils import get_cloudflare_rtc_configuration

            result = get_cloudflare_rtc_configuration()
            assert result == {"iceServers": []}

    def test_failure(self):
        with (
            patch("unmute.webrtc_utils.os.environ", {"TURN_KEY_ID": "test-id", "TURN_KEY_API_TOKEN": "test-token"}),
            patch("unmute.webrtc_utils.requests.post") as mock_post,
        ):
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response

            from unmute.webrtc_utils import get_cloudflare_rtc_configuration

            with pytest.raises(Exception, match="Failed to get TURN credentials"):
                get_cloudflare_rtc_configuration()

    def test_request_body(self):
        with (
            patch("unmute.webrtc_utils.os.environ", {"TURN_KEY_ID": "test-id", "TURN_KEY_API_TOKEN": "test-token"}),
            patch("unmute.webrtc_utils.requests.post") as mock_post,
        ):
            mock_post.return_value.ok = True
            mock_post.return_value.json.return_value = {"iceServers": []}

            from unmute.webrtc_utils import get_cloudflare_rtc_configuration

            get_cloudflare_rtc_configuration()

            call_args = mock_post.call_args
            assert call_args.kwargs["json"] == {"ttl": 86400}
            assert call_args.kwargs["headers"]["Content-Type"] == "application/json"
