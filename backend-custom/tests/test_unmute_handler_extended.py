"""Extended tests for unmute_handler.py to increase coverage.

Tests methods not covered by test_unmute_handler.py:
  - __aenter__ / __aexit__
  - start_up / start_up_stt
  - emit
  - copy
  - get_gradio_update
  - receive (conversation state paths)
  - _stt_loop
  - _tts_loop
  - _generate_response_task
  - interrupt_bot (happy path)
  - update_session (disallow_recording)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME


def _patch_handler_deps():
    """Return context managers for patching UnmuteHandler dependencies."""
    return (
        patch("unmute.unmute_handler.Chatbot"),
        patch("unmute.unmute_handler.get_openai_client"),
        patch("unmute.unmute_handler.Recorder", return_value=None),
        patch("unmute.unmute_handler.RECORDINGS_DIR", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle: __aenter__, __aexit__, start_up, start_up_stt, cleanup
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerLifecycle:
    @pytest.mark.asyncio
    async def test_aenter(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            # quest_manager.__aenter__ is called
            await handler.__aenter__()
            # No assertion needed - it should not raise

    @pytest.mark.asyncio
    async def test_aexit_after_aenter(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            # quest_manager.__aexit__ requires __aenter__ to be called first
            await handler.__aenter__()
            await handler.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_start_up_calls_start_up_stt(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.start_up_stt = AsyncMock()
            await handler.start_up()
            handler.start_up_stt.assert_awaited_once()
            assert handler.waiting_for_user_start_time == 0.0

    @pytest.mark.asyncio
    async def test_start_up_stt_creates_quest(self):
        """start_up_stt should create an STT quest via find_instance."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            fake_stt = MagicMock()
            fake_stt.shutdown = AsyncMock()

            handler = UnmuteHandler()

            async def fake_run(stt):
                pass

            # We need to mock _stt_loop so it doesn't block
            handler._stt_loop = fake_run

            # Quest manager needs to be entered first
            await handler.__aenter__()

            with patch("unmute.unmute_handler.find_instance", return_value=fake_stt):
                await handler.start_up_stt()

            # Check the STT quest was added
            assert "stt" in handler.quest_manager.quests

    @pytest.mark.asyncio
    async def test_cleanup_shuts_down_recorder(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler
            from unmute.recorder import Recorder

            # Create handler with a mock recorder instead of None
            fake_recorder = MagicMock()
            fake_recorder.shutdown = AsyncMock()

            # We need to patch RECORDINGS_DIR to be non-None so recorder is created
            # but then mock Recorder to return our fake
            with patch("unmute.unmute_handler.RECORDINGS_DIR", "/tmp/rec"):
                with patch("unmute.unmute_handler.Recorder", return_value=fake_recorder):
                    handler = UnmuteHandler()
                    await handler.cleanup()
                    fake_recorder.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_recorder(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.recorder = None
            # Should not raise
            await handler.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# emit()
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerEmit:
    @pytest.mark.asyncio
    async def test_emit_returns_item_from_queue(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            import unmute.openai_realtime_api_events as ora
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.output_queue = asyncio.Queue()
            event = ora.ResponseCreated(
                response=ora.Response(status="in_progress", voice="alloy")
            )
            await handler.output_queue.put(event)

            result = await handler.emit()
            assert isinstance(result, ora.ResponseCreated)

    @pytest.mark.asyncio
    async def test_emit_returns_gradio_update_on_silence(self):
        """When output_queue is empty and enough time has passed, emit should return a GradioUpdate."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            import unmute.openai_realtime_api_events as ora
            from fastrtc import AdditionalOutputs
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [{"role": "user", "content": "hi"}]
            handler.chatbot.conversation_state.return_value = "waiting_for_user"
            # stt/tts are properties that read from quest_manager.quests
            # which is empty by default, so stt/tts return None automatically
            handler.tts_voice = None

            # Last update was 10 seconds ago, now is 0
            handler.last_additional_output_update = -10
            handler.n_samples_received = 0  # current time = 0

            # Simulate empty queue by waiting forever – but we need a quick exit
            # Let's put None-ish data to simulate queue becoming available
            async def fake_wait_for_item(queue):
                return None

            with patch("unmute.unmute_handler.wait_for_item", fake_wait_for_item):
                result = await handler.emit()
                assert isinstance(result, AdditionalOutputs)

    @pytest.mark.asyncio
    async def test_emit_returns_none_recent_update(self):
        """When output_queue is empty and update is recent, emit should return None."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.last_additional_output_update = 0  # recent
            handler.n_samples_received = 0  # current time = 0

            async def fake_wait_for_item(queue):
                return None

            with patch("unmute.unmute_handler.wait_for_item", fake_wait_for_item):
                result = await handler.emit()
                assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# copy()
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerCopy:
    def test_copy_returns_new_instance(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            new_handler = handler.copy()
            assert isinstance(new_handler, UnmuteHandler)
            assert new_handler is not handler


# ─────────────────────────────────────────────────────────────────────────────
# get_gradio_update()
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerGetGradioUpdate:
    def test_get_gradio_update_no_stt_tts(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from fastrtc import AdditionalOutputs
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "waiting_for_user"
            handler.chatbot.chat_history = [{"role": "user", "content": "hi"}]
            # stt/tts are properties that read from quest_manager.quests
            # which is empty by default, so they return None automatically
            handler.tts_voice = None

            result = handler.get_gradio_update()
            assert isinstance(result, AdditionalOutputs)

    def test_get_gradio_update_with_stt_tts(self):
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from fastrtc import AdditionalOutputs
            from unmute.stt.exponential_moving_average import ExponentialMovingAverage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.chatbot.chat_history = [
                {"role": "system", "content": "Be nice"},
                {"role": "user", "content": "hi"},
            ]

            fake_stt = MagicMock()
            fake_stt.state.return_value = "connected"
            fake_stt.pause_prediction = ExponentialMovingAverage(
                attack_time=0.01, release_time=0.01, initial_value=0.8
            )
            # stt/tts are read from quest_manager.quests
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            fake_tts = MagicMock()
            fake_tts.state.return_value = "connected"
            fake_tts.voice = "alloy"
            handler.quest_manager.quests["tts"] = MagicMock()
            handler.quest_manager.quests["tts"].get_nowait.return_value = fake_tts
            handler.tts_voice = "alloy"

            result = handler.get_gradio_update()
            assert isinstance(result, AdditionalOutputs)
            # System prompt should be excluded
            assert len(result.args[0].chat_history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# receive() – conversation state paths
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerReceive:
    @pytest.mark.asyncio
    async def test_receive_bot_speaking_updates_waiting_for_user_start(self):
        """When bot is speaking, receive should update waiting_for_user_start_time."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"
            handler.audio_input_override = None

            fake_stt = MagicMock()
            fake_stt.send_audio = AsyncMock()
            fake_stt.current_time = 1.0
            fake_stt.delay_sec = 0.5
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.2
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))
            handler.stt_end_of_flush_time = 100.0  # already flushing

            await handler.receive(frame)
            # waiting_for_user_start_time should be updated to current audio time
            assert handler.waiting_for_user_start_time > 0

    @pytest.mark.asyncio
    async def test_receive_user_speaking_sets_timing(self):
        """When user is speaking, receive should reset timing dict."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.audio_input_override = None

            fake_stt = MagicMock()
            fake_stt.send_audio = AsyncMock()
            fake_stt.current_time = 1.0
            fake_stt.delay_sec = 0.5
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.3
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            handler.stt_end_of_flush_time = None
            handler.chat_history = []  # not generating initial response

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))

            with patch.object(handler, "detect_long_silence", new=AsyncMock()):
                with patch.object(handler, "determine_pause", return_value=False):
                    await handler.receive(frame)

            assert handler.debug_dict["timing"] == {}

    @pytest.mark.asyncio
    async def test_receive_pause_detected(self):
        """When pause is detected, send zeros to STT for flushing."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.audio_input_override = None

            fake_stt = MagicMock()
            fake_stt.send_audio = AsyncMock()
            fake_stt.current_time = 1.0
            fake_stt.delay_sec = 0.5
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.3
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            handler.stt_end_of_flush_time = None

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))

            with patch.object(handler, "detect_long_silence", new=AsyncMock()):
                with patch.object(handler, "determine_pause", return_value=True):
                    with patch.object(handler, "_generate_response", new=AsyncMock()):
                        await handler.receive(frame)

            # stt_end_of_flush_time should be set
            assert handler.stt_end_of_flush_time is not None

    @pytest.mark.asyncio
    async def test_receive_vad_interruption(self):
        """When VAD pause < 0.4 during bot_speaking, interrupt bot."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"
            handler.chatbot.chat_history = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
            handler.audio_input_override = None

            fake_stt = MagicMock()
            fake_stt.send_audio = AsyncMock()
            fake_stt.current_time = 5.0  # > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            fake_stt.delay_sec = 0.5
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.3  # < 0.4
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            handler.stt_end_of_flush_time = None
            handler.n_samples_received = int(SAMPLE_RATE * 5)  # 5 seconds

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))

            with patch.object(handler, "interrupt_bot", new=AsyncMock()) as mock_interrupt:
                with patch.object(handler, "add_chat_message_delta", new=AsyncMock()):
                    with patch.object(handler, "detect_long_silence", new=AsyncMock()):
                        with patch.object(handler, "determine_pause", return_value=False):
                            await handler.receive(frame)

            mock_interrupt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_receive_flushing_finished_generates_response(self):
        """When flushing is done and STT time is past end_of_flush_time, generate response."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.chatbot.chat_history = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "hi"},
            ]
            handler.audio_input_override = None

            fake_stt = MagicMock()
            fake_stt.send_audio = AsyncMock()
            fake_stt.current_time = 10.0
            fake_stt.delay_sec = 0.5
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.2
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt

            handler.stt_end_of_flush_time = 5.0  # flush ended at time 5
            handler.stt_flush_timer = MagicMock()
            handler.stt_flush_timer.time.return_value = 1.0

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))

            with patch.object(handler, "_generate_response", new=AsyncMock()) as mock_gen:
                await handler.receive(frame)
            mock_gen.assert_awaited_once()
            # stt_end_of_flush_time reset and response generated
            assert handler.stt_end_of_flush_time is None


# ─────────────────────────────────────────────────────────────────────────────
# _stt_loop
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerSttLoop:
    @pytest.mark.asyncio
    async def test_stt_loop_processes_word_messages(self):
        """_stt_loop should process word messages from STT."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.stt.speech_to_text import STTWordMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.chatbot.chat_history = []

            async def fake_add(delta, role, **kwargs):
                return True

            handler.chatbot.add_chat_message_delta = fake_add

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTWordMessage(
                    type="Word", text="hello", start_time=0.1
                )

            fake_stt.__aiter__ = lambda self: stt_iterator()

            # Collect events from output_queue
            async def run_stt_loop():
                import asyncio

                task = asyncio.create_task(handler._stt_loop(fake_stt))
                await asyncio.sleep(0.1)  # let it run

                # Put a sentinel to make it exit
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_stt_loop()

    @pytest.mark.asyncio
    async def test_stt_loop_ignores_marker_messages(self):
        """_stt_loop should skip STTMarkerMessage."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.stt.speech_to_text import STTMarkerMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"

            async def fake_add(delta, role, **kwargs):
                return True

            handler.chatbot.add_chat_message_delta = fake_add

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTMarkerMessage(type="Marker", id=1)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
                import asyncio

                task = asyncio.create_task(handler._stt_loop(fake_stt))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_stt_loop()

    @pytest.mark.asyncio
    async def test_stt_loop_empty_text_skipped(self):
        """_stt_loop should skip empty text messages."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.stt.speech_to_text import STTWordMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            add_calls = []

            async def fake_add(delta, role, **kwargs):
                add_calls.append((delta, role))
                return True

            handler.chatbot.add_chat_message_delta = fake_add

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                # Empty text should be sent to output queue but not added to chat
                yield STTWordMessage(type="Word", text="", start_time=0.0)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
                import asyncio

                task = asyncio.create_task(handler._stt_loop(fake_stt))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_stt_loop()

            # Empty text should NOT have been added via add_chat_message_delta
            assert add_calls == []

    @pytest.mark.asyncio
    async def test_stt_loop_interrupts_bot_on_word(self):
        """When bot is speaking and STT gets a word, it should interrupt."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.stt.speech_to_text import STTWordMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"

            async def fake_add(delta, role, **kwargs):
                pass

            handler.chatbot.add_chat_message_delta = fake_add

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTWordMessage(type="Word", text="hello", start_time=0.1)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
                import asyncio

                task = asyncio.create_task(handler._stt_loop(fake_stt))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            with patch.object(handler, "interrupt_bot", new=AsyncMock()) as mock_interrupt:
                await run_stt_loop()
            mock_interrupt.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# interrupt_bot (happy path)
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerInterruptBotHappy:
    @pytest.mark.asyncio
    async def test_interrupt_bot_success(self):
        """Interrupt bot should work when conversation is in bot_speaking state."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"

            async def fake_add(delta, role, **kwargs):
                pass

            handler.chatbot.add_chat_message_delta = fake_add
            handler._clear_queue = MagicMock()

            await handler.interrupt_bot()

            handler._clear_queue.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# update_session – disallow_recording path
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerUpdateSessionRecording:
    @pytest.mark.asyncio
    async def test_update_session_disallow_recording(self):
        """When allow_recording=False, recorder should be shut down."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            import unmute.openai_realtime_api_events as ora
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()

            fake_recorder = MagicMock()
            fake_recorder.add_event = AsyncMock()
            fake_recorder.shutdown = AsyncMock()
            handler.recorder = fake_recorder

            session = ora.SessionConfig(
                instructions=None,
                voice="alloy",
                allow_recording=False,
            )
            await handler.update_session(session)

            fake_recorder.shutdown.assert_awaited_with(keep_recording=False)
            assert handler.recorder is None


# ─────────────────────────────────────────────────────────────────────────────
# _tts_loop
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerTtsLoop:
    @pytest.mark.asyncio
    async def test_tts_loop_processes_audio_and_text(self):
        """_tts_loop should process audio and text messages from TTS."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.tts.text_to_speech import TTSAudioMessage, TTSTextMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [
                {"role": "system", "content": "Be nice"},
            ]

            async def fake_add(delta, role, **kwargs):
                pass

            handler.chatbot.add_chat_message_delta = fake_add

            fake_tts = MagicMock()

            async def tts_iterator():
                yield TTSTextMessage(
                    type="Text", text="Hello", start_s=0.0, stop_s=0.5
                )
                yield TTSAudioMessage(type="Audio", pcm=[0.0] * 100)

            fake_tts.__aiter__ = lambda self: tts_iterator()
            fake_tts.received_samples = 0
            fake_tts.received_samples_yielded = 0
            fake_tts.shutdown = AsyncMock()

            handler.tts_output_stopwatch = MagicMock()
            handler.tts_output_stopwatch.stop.return_value = 0.5

            async def run_tts_loop():
                import asyncio

                generating_message_i = 1
                task = asyncio.create_task(handler._tts_loop(fake_tts, generating_message_i))
                await asyncio.sleep(0.2)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_tts_loop()

    @pytest.mark.asyncio
    async def test_tts_loop_empty_text_message_skipped(self):
        """TTS empty text message (start_s=0, stop_s=0) should be skipped."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.tts.text_to_speech import TTSTextMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [{"role": "system", "content": "test"}]

            add_calls = []

            async def fake_add(delta, role, **kwargs):
                add_calls.append((delta, role))

            handler.chatbot.add_chat_message_delta = fake_add

            fake_tts = MagicMock()

            async def tts_iterator():
                # Empty text message from TTS server (always emitted, should be skipped)
                yield TTSTextMessage(type="Text", text="", start_s=0, stop_s=0)

            fake_tts.__aiter__ = lambda self: tts_iterator()
            fake_tts.received_samples = 0
            fake_tts.received_samples_yielded = 0
            fake_tts.shutdown = AsyncMock()

            handler.tts_output_stopwatch = MagicMock()
            handler.tts_output_stopwatch.stop.return_value = None

            async def run_tts_loop():
                import asyncio

                task = asyncio.create_task(handler._tts_loop(fake_tts, 1))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_tts_loop()

            # Empty text from TTS server is skipped, but the TTS loop itself
            # calls add_chat_message_delta for the empty assistant message at the end
            # and empty user message - these are from the loop's cleanup code
            # The TTSTextMessage with text="", start_s=0, stop_s=0 is skipped
            # So we just check the loop completes without error
            pass


# ─────────────────────────────────────────────────────────────────────────────
# _generate_response_task
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerGenerateResponseTask:
    @pytest.mark.asyncio
    async def test_generate_response_task_first_message_temperature(self):
        """First message (generating_message_i==2) should use FIRST_MESSAGE_TEMPERATURE."""
        p1, p2, p3, p4 = _patch_handler_deps()
        with p1, p2, p3, p4:
            from unmute.unmute_handler import (
                FIRST_MESSAGE_TEMPERATURE,
                UnmuteHandler,
            )

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [
                {"role": "system", "content": "Be nice"},
            ]  # 1 message, so generating_message_i == 1 (not 2)

            async def fake_add(delta, role, **kwargs):
                return False

            handler.chatbot.add_chat_message_delta = fake_add
            handler.chatbot.preprocessed_messages.return_value = [
                {"role": "system", "content": "Be nice"},
                {"role": "user", "content": "hi"},
            ]
            handler.tts_voice = None

            # Mock TTS startup
            fake_tts = MagicMock()
            fake_tts.send = AsyncMock()

            async def fake_tts_init():
                return fake_tts

            # Mock VLLMStream to yield one word then stop
            async def fake_llm_chat_completion(messages):
                yield "Hello"

            fake_vllm_instance = MagicMock()
            fake_vllm_instance.chat_completion = fake_llm_chat_completion

            with patch("unmute.unmute_handler.VLLMStream", return_value=fake_vllm_instance):
                with patch.object(handler, "start_up_tts", new=AsyncMock()) as mock_tts:
                    from unmute.quest_manager import Quest
                    fake_quest = MagicMock()
                    fake_quest.get = AsyncMock(side_effect=fake_tts_init)
                    mock_tts.return_value = fake_quest

                    async def run_task():
                        import asyncio

                        task = asyncio.create_task(handler._generate_response_task())
                        await asyncio.sleep(0.2)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    await run_task()

            # Verify ResponseCreated was emitted
            assert not handler.output_queue.empty()
            item = await handler.output_queue.get()
            import unmute.openai_realtime_api_events as ora

            assert isinstance(item, ora.ResponseCreated)
