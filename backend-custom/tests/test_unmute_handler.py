"""Tests for unmute_handler.py: UnmuteHandler (selected methods)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from unmute.kyutai_constants import SAMPLE_RATE


def _patch_handler_deps():
    """Return context managers for patching UnmuteHandler dependencies."""
    return (
        patch("unmute.unmute_handler.Chatbot"),
        patch("unmute.unmute_handler.get_openai_client"),
    )


class TestUnmuteHandlerInit:
    @pytest.mark.asyncio
    async def test_init(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            assert handler.n_samples_received == 0

    @pytest.mark.asyncio
    async def test_audio_received_sec(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.n_samples_received = SAMPLE_RATE  # 1 second
            assert handler.audio_received_sec() == 1.0


class TestUnmuteHandlerDeterminePause:
    @pytest.mark.asyncio
    async def test_no_stt_returns_false(self):
        """When stt is None, determine_pause should return False."""
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            # stt is None by default (no quest)
            assert handler.determine_pause() is False

    @pytest.mark.asyncio
    async def test_not_user_speaking_returns_false(self):
        """When conversation state is not user_speaking, pause should be False."""
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler
            from unmute.stt.exponential_moving_average import ExponentialMovingAverage

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"

            # Create a real STT-like object
            fake_stt = MagicMock()
            fake_stt.pause_prediction = ExponentialMovingAverage(
                attack_time=0.01, release_time=0.01, initial_value=0.9
            )
            # Store in quest manager
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            assert handler.determine_pause() is False

    @pytest.mark.asyncio
    async def test_pause_detected_high_prediction(self):
        """When pause prediction > 0.6 and user is speaking, pause should be True."""
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler
            from unmute.stt.exponential_moving_average import ExponentialMovingAverage

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"

            fake_stt = MagicMock()
            fake_stt.pause_prediction = ExponentialMovingAverage(
                attack_time=0.01, release_time=0.01, initial_value=0.9
            )
            fake_stt.sent_samples = SAMPLE_RATE
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt
            handler.stt_last_message_time = 0.0

            assert handler.determine_pause() is True

    @pytest.mark.asyncio
    async def test_no_pause_low_prediction(self):
        """When pause prediction < 0.6, pause should be False."""
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler
            from unmute.stt.exponential_moving_average import ExponentialMovingAverage

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"

            fake_stt = MagicMock()
            fake_stt.pause_prediction = ExponentialMovingAverage(
                attack_time=0.01, release_time=0.01, initial_value=0.3
            )
            fake_stt.sent_samples = SAMPLE_RATE
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt
            handler.stt_last_message_time = 0.0

            assert handler.determine_pause() is False


class TestUnmuteHandlerInterruptBot:
    @pytest.mark.asyncio
    async def test_interrupt_bot_wrong_state(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "waiting_for_user"
            handler._clear_queue = None

            with pytest.raises(RuntimeError, match="Can't interrupt"):
                await handler.interrupt_bot()


class TestUnmuteHandlerDetectLongSilence:
    @pytest.mark.asyncio
    async def test_no_silence_yet(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "waiting_for_user"
            handler.waiting_for_user_start_time = handler.audio_received_sec()
            # Should not trigger silence (just started)
            await handler.detect_long_silence()

    @pytest.mark.asyncio
    async def test_silence_detected(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            chatbot_mock = MagicMock()
            chatbot_mock.conversation_state.return_value = "waiting_for_user"

            async def fake_add(delta, role, **kwargs):
                pass

            chatbot_mock.add_chat_message_delta = fake_add
            handler.chatbot = chatbot_mock

            # Simulate 10 seconds of silence
            handler.n_samples_received = int(SAMPLE_RATE * 10)
            handler.waiting_for_user_start_time = 0.0

            await handler.detect_long_silence()


class TestUnmuteHandlerCheckForBotGoodbye:
    @pytest.mark.asyncio
    async def test_goodbye_detected(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [
                {"role": "system", "content": "Be nice."},
                {"role": "user", "content": "Bye!"},
                {"role": "assistant", "content": "See you later! Bye!"},
            ]

            handler.output_queue = asyncio.Queue()
            await handler.check_for_bot_goodbye()

            item = await handler.output_queue.get()
            from fastrtc import CloseStream
            assert isinstance(item, CloseStream)

    @pytest.mark.asyncio
    async def test_no_goodbye(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [
                {"role": "system", "content": "Be nice."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]

            handler.output_queue = asyncio.Queue()
            await handler.check_for_bot_goodbye()

            # Queue should be empty
            assert handler.output_queue.empty()


class TestUnmuteHandlerUpdateSession:
    @pytest.mark.asyncio
    async def test_update_instructions(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler
            import unmute.openai_realtime_api_events as ora
            from unmute.llm.system_prompt import ConstantInstructions

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.set_instructions = MagicMock()

            session = ora.SessionConfig(
                instructions=ConstantInstructions(),
                voice="alloy",
            )
            await handler.update_session(session)

            handler.chatbot.set_instructions.assert_called_once()
            assert handler.tts_voice == "alloy"

    @pytest.mark.asyncio
    async def test_update_voice(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler
            import unmute.openai_realtime_api_events as ora

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()

            session = ora.SessionConfig(
                instructions=None,
                voice="gertrude",
            )
            await handler.update_session(session)

            assert handler.tts_voice == "gertrude"
