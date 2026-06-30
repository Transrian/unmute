"""Tests for tts/voice_cloning.py."""

from unittest.mock import MagicMock, patch

import pytest

from unmute.tts.voice_cloning import clone_voice


class TestCloneVoice:
    def test_returns_custom_prefix(self):
        mock_response = MagicMock()
        mock_response.content = b"fake-embedding-data"
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.tts.voice_cloning.requests") as mock_requests,
            patch("unmute.tts.voice_cloning.voice_embeddings_cache"),
        ):
            mock_requests.post.return_value = mock_response

            result = clone_voice(b"fake-audio")

            assert result.startswith("custom:")
            # Should be a UUID
            uuid_part = result[len("custom:"):]
            import uuid

            uuid.UUID(uuid_part)  # Should not raise

    def test_calls_voice_cloning_server(self):
        mock_response = MagicMock()
        mock_response.content = b"fake-embedding-data"
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.tts.voice_cloning.requests") as mock_requests,
            patch("unmute.tts.voice_cloning.voice_embeddings_cache"),
        ):
            mock_requests.post.return_value = mock_response

            clone_voice(b"fake-audio")

            mock_requests.post.assert_called_once()
            call_args = mock_requests.post.call_args
            assert b"fake-audio" == call_args[1]["data"]
            assert call_args[1]["headers"]["Content-Type"] == "application/octet-stream"

    def test_stores_in_cache(self):
        mock_response = MagicMock()
        mock_response.content = b"fake-embedding-data"
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.tts.voice_cloning.requests") as mock_requests,
            patch("unmute.tts.voice_cloning.voice_embeddings_cache") as mock_cache,
        ):
            mock_requests.post.return_value = mock_response

            clone_voice(b"fake-audio")

            mock_cache.set.assert_called_once()
            mock_cache.cleanup.assert_called_once()

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")

        with (
            patch("unmute.tts.voice_cloning.requests") as mock_requests,
        ):
            mock_requests.post.return_value = mock_response

            with pytest.raises(Exception, match="HTTP 500"):
                clone_voice(b"fake-audio")
