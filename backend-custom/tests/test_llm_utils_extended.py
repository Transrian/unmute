"""Extended tests for llm/llm_utils.py to increase coverage.

Tests preprocess_messages_for_llm, rechunk_to_words, VLLMStream, autoselect_model,
and other utility functions.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unmute.llm.llm_utils import (
    INTERRUPTION_CHAR,
    USER_SILENCE_MARKER,
)


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_messages_for_llm
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessMessagesForLLM:
    def test_removes_empty_interruption_messages(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": INTERRUPTION_CHAR},  # should be removed
            {"role": "user", "content": "Bye"},
        ]
        result = preprocess_messages_for_llm(history)
        # The interruption-only message should be removed,
        # and the two adjacent user messages get concatenated
        assert len(result) == 1
        assert result[0]["content"] == "Hello Bye"

    def test_removes_interruption_suffix(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Nice to meet you—"},
        ]
        result = preprocess_messages_for_llm(history)
        assert result[1]["content"] == "Nice to meet you"

    def test_concatenates_adjacent_same_roles(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        result = preprocess_messages_for_llm(history)
        assert len(result) == 1
        assert result[0]["content"] == "Hello How are you?"

    def test_keeps_system_and_assistant(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "system", "content": "Be nice"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = preprocess_messages_for_llm(history)
        # Should add a dummy user message for Gemma compatibility
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello."
        assert result[2]["role"] == "assistant"

    def test_no_dummy_message_when_user_after_system(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": "Hello"},
        ]
        result = preprocess_messages_for_llm(history)
        assert len(result) == 2
        # No dummy user message needed

    def test_removes_silence_marker_prefix(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": USER_SILENCE_MARKER + " and then some words"},
        ]
        result = preprocess_messages_for_llm(history)
        # The silence marker should be stripped from the content
        assert result[1]["content"] == " and then some words"

    def test_keeps_silence_marker_alone(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": USER_SILENCE_MARKER},
        ]
        result = preprocess_messages_for_llm(history)
        # When the marker is alone, it should be kept
        assert result[1]["content"] == USER_SILENCE_MARKER

    def test_empty_history(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        result = preprocess_messages_for_llm([])
        assert result == []

    def test_deep_copies_messages(self):
        from unmute.llm.llm_utils import preprocess_messages_for_llm

        history = [
            {"role": "user", "content": "Hello"},
        ]
        original_content = history[0]["content"]
        result = preprocess_messages_for_llm(history)
        # Mutating result should not affect original
        result[0]["content"] = "Modified"
        assert history[0]["content"] == original_content


# ─────────────────────────────────────────────────────────────────────────────
# rechunk_to_words
# ─────────────────────────────────────────────────────────────────────────────


class TestRechunkToWords:
    @pytest.mark.asyncio
    async def test_rechunks_to_whole_words(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            yield "He"
            yield "ll"
            yield "o "
            yield "w"
            yield "or"
            yield "ld"

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_preserves_spaces_as_prefix(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            yield "Hello world"

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_merges_multiple_spaces(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            yield "Hello  world"

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_empty_final_buffer_not_yielded(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            yield "Hello "
            yield "  "  # just spaces

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == ["Hello"]

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            # yield nothing
            if False:
                yield ""

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == []

    @pytest.mark.asyncio
    async def test_no_spaces_returns_single_chunk(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            yield "Hello"

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == ["Hello"]

    @pytest.mark.asyncio
    async def test_chunked_word_assembly(self):
        from unmute.llm.llm_utils import rechunk_to_words

        async def iterator():
            yield "H"
            yield "e"
            yield "l"
            yield "l"
            yield "o"
            yield " "
            yield "w"
            yield "o"
            yield "r"
            yield "l"
            yield "d"

        words = []
        async for word in rechunk_to_words(iterator()):
            words.append(word)
        assert words == ["Hello", " world"]


# ─────────────────────────────────────────────────────────────────────────────
# get_openai_client
# ─────────────────────────────────────────────────────────────────────────────


class TestGetOpenaiClient:
    def test_returns_async_openai_client(self):
        from unmute.llm.llm_utils import get_openai_client

        client = get_openai_client()
        assert client.base_url is not None

    def test_custom_server_url(self):
        from unmute.llm.llm_utils import get_openai_client

        client = get_openai_client(server_url="http://custom:8000")
        assert "custom" in str(client.base_url)

    def test_none_api_key_uses_dummy(self):
        from unmute.llm.llm_utils import get_openai_client

        client = get_openai_client(api_key=None)
        assert client.api_key == "EMPTY"


# ─────────────────────────────────────────────────────────────────────────────
# VLLMStream
# ─────────────────────────────────────────────────────────────────────────────


class TestVLLMStream:
    def test_init_sets_model(self):
        from unittest.mock import patch as mock_patch
        from unmute.llm.llm_utils import VLLMStream

        fake_client = MagicMock()
        with mock_patch("unmute.llm.llm_utils.autoselect_model", return_value="test-model"):
            stream = VLLMStream(fake_client, temperature=0.5)
            assert stream.model == "test-model"
            assert stream.temperature == 0.5

    @pytest.mark.asyncio
    async def test_chat_completion_streams_content(self):
        from unmute.llm.llm_utils import VLLMStream

        fake_client = MagicMock()

        class FakeStream:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

        mock_delta1 = MagicMock()
        mock_delta1.content = "Hello"
        mock_choice1 = MagicMock()
        mock_choice1.delta = mock_delta1
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [mock_choice1]

        mock_delta2 = MagicMock()
        mock_delta2.content = " world"
        mock_choice2 = MagicMock()
        mock_choice2.delta = mock_delta2
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [mock_choice2]

        # Empty choices chunk (OpenRouter keep-alive)
        mock_chunk_empty = MagicMock()
        mock_chunk_empty.choices = []

        fake_stream = FakeStream([mock_chunk_empty, mock_chunk1, mock_chunk2])
        fake_client.chat.completions.create = AsyncMock(return_value=fake_stream)

        with patch("unmute.llm.llm_utils.autoselect_model", return_value="test-model"):
            stream = VLLMStream(fake_client, temperature=0.5)
            chunks = []
            async for content in stream.chat_completion([{"role": "user", "content": "hi"}]):
                chunks.append(content)
            assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_completion_skips_empty_content(self):
        from unmute.llm.llm_utils import VLLMStream

        fake_client = MagicMock()

        class FakeStream:
            def __init__(self, chunks):
                self._chunks = chunks
                self._index = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._chunks):
                    raise StopAsyncIteration
                chunk = self._chunks[self._index]
                self._index += 1
                return chunk

        # Delta with None content (happens on first message)
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]

        fake_stream = FakeStream([mock_chunk])
        fake_client.chat.completions.create = AsyncMock(return_value=fake_stream)

        with patch("unmute.llm.llm_utils.autoselect_model", return_value="test-model"):
            stream = VLLMStream(fake_client, temperature=0.5)
            chunks = []
            async for content in stream.chat_completion([{"role": "user", "content": "hi"}]):
                chunks.append(content)
            assert chunks == []


# ─────────────────────────────────────────────────────────────────────────────
# autoselect_model
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoselectModel:
    def test_returns_env_model_when_set(self):
        from unmute.llm.llm_utils import autoselect_model

        with patch("unmute.llm.llm_utils.KYUTAI_LLM_MODEL", "my-custom-model"):
            # Clear the cache to re-evaluate
            autoselect_model.cache_clear()
            result = autoselect_model()
            assert result == "my-custom-model"

    def test_autoselect_raises_on_multiple_models(self):
        from unmute.llm.llm_utils import autoselect_model

        autoselect_model.cache_clear()

        with patch("unmute.llm.llm_utils.KYUTAI_LLM_MODEL", None):
            fake_model_list = MagicMock()
            fake_model_list.data = [
                MagicMock(id="model1"),
                MagicMock(id="model2"),
            ]

            with patch("unmute.llm.llm_utils.OpenAI") as mock_openai:
                mock_openai.return_value.models.list.return_value = fake_model_list
                with pytest.raises(ValueError, match="multiple models"):
                    autoselect_model()
