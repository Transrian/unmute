"""Extended tests for unmute_handler.py to increase coverage."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME


def _patch_handler_deps():
    return (
        patch("unmute.unmute_handler.Chatbot"),
        patch("unmute.unmute_handler.get_openai_client"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerLifecycle:
    @pytest.mark.asyncio
    async def test_aenter(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            await handler.__aenter__()

    @pytest.mark.asyncio
    async def test_aexit_after_aenter(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            await handler.__aenter__()
            await handler.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_start_up_calls_start_up_stt(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.start_up_stt = AsyncMock()
            await handler.start_up()
            handler.start_up_stt.assert_awaited_once()
            assert handler.waiting_for_user_start_time == 0.0

    @pytest.mark.asyncio
    async def test_start_up_stt_creates_quest(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            fake_stt = MagicMock()
            fake_stt.shutdown = AsyncMock()
            handler = UnmuteHandler()
            handler._stt_loop = lambda stt: asyncio.sleep(0)
            await handler.__aenter__()
            with patch("unmute.unmute_handler.find_instance", return_value=fake_stt):
                await handler.start_up_stt()
            assert "stt" in handler.quest_manager.quests

# ─────────────────────────────────────────────────────────────────────────────
# emit()
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerEmit:
    @pytest.mark.asyncio
    async def test_emit_returns_item_from_queue(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
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


# ─────────────────────────────────────────────────────────────────────────────
# copy()
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerCopy:
    def test_copy_returns_new_instance(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            new_handler = handler.copy()
            assert isinstance(new_handler, UnmuteHandler)
            assert new_handler is not handler


# ─────────────────────────────────────────────────────────────────────────────
# receive() - conversation state paths
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerReceive:
    @pytest.mark.asyncio
    async def test_receive_bot_speaking_updates_waiting_for_user_start(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
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
            handler.stt_end_of_flush_time = 100.0

            await handler.receive(frame)
            assert handler.waiting_for_user_start_time > 0

    @pytest.mark.asyncio
    async def test_receive_pause_detected(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
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

            assert handler.stt_end_of_flush_time is not None

    @pytest.mark.asyncio
    async def test_receive_vad_interruption(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
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
            fake_stt.current_time = 5.0
            fake_stt.delay_sec = 0.5
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.3
            handler.quest_manager.quests["stt"] = MagicMock()
            handler.quest_manager.quests["stt"].get_nowait.return_value = fake_stt
            handler.stt_end_of_flush_time = None
            handler.n_samples_received = int(SAMPLE_RATE * 5)

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))
            with patch.object(handler, "interrupt_bot", new=AsyncMock()) as mock_interrupt:
                with patch.object(handler, "add_chat_message_delta", new=AsyncMock()):
                    with patch.object(handler, "detect_long_silence", new=AsyncMock()):
                        with patch.object(handler, "determine_pause", return_value=False):
                            await handler.receive(frame)
            mock_interrupt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_receive_flushing_finished_generates_response(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
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
            handler.stt_end_of_flush_time = 5.0
            handler.stt_flush_timer = MagicMock()
            handler.stt_flush_timer.time.return_value = 1.0

            frame = (SAMPLE_RATE, np.zeros((1, SAMPLES_PER_FRAME), dtype=np.int16))
            with patch.object(handler, "_generate_response", new=AsyncMock()) as mock_gen:
                await handler.receive(frame)
            mock_gen.assert_awaited_once()
            assert handler.stt_end_of_flush_time is None


# ─────────────────────────────────────────────────────────────────────────────
# _stt_loop
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerSttLoop:
    @pytest.mark.asyncio
    async def test_stt_loop_processes_word_messages(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.stt.speech_to_text import STTWordMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.chatbot.chat_history = []
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0)

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTWordMessage(type="Word", text="hello", start_time=0.1)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
                task = asyncio.create_task(handler._stt_loop(fake_stt))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_stt_loop()

    @pytest.mark.asyncio
    async def test_stt_loop_ignores_marker_messages(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.stt.speech_to_text import STTMarkerMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0)

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTMarkerMessage(type="Marker", id=1)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
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
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.stt.speech_to_text import STTWordMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "user_speaking"
            add_calls = []
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: add_calls.append((delta, role)) or asyncio.sleep(0)

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTWordMessage(type="Word", text="", start_time=0.0)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
                task = asyncio.create_task(handler._stt_loop(fake_stt))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_stt_loop()
            assert add_calls == []

    @pytest.mark.asyncio
    async def test_stt_loop_interrupts_bot_on_word(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.stt.speech_to_text import STTWordMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0)

            fake_stt = MagicMock()
            fake_stt.pause_prediction = MagicMock()
            fake_stt.pause_prediction.value = 0.0

            async def stt_iterator():
                yield STTWordMessage(type="Word", text="hello", start_time=0.1)

            fake_stt.__aiter__ = lambda self: stt_iterator()

            async def run_stt_loop():
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
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.conversation_state.return_value = "bot_speaking"
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0)
            handler._clear_queue = MagicMock()

            await handler.interrupt_bot()
            handler._clear_queue.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# _tts_loop
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerTtsLoop:
    @pytest.mark.asyncio
    async def test_tts_loop_processes_audio_and_text(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.tts.text_to_speech import TTSAudioMessage, TTSTextMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [{"role": "system", "content": "Be nice"}]
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0)

            fake_tts = MagicMock()

            async def tts_iterator():
                yield TTSTextMessage(type="Text", text="Hello", start_s=0.0, stop_s=0.5)
                yield TTSAudioMessage(type="Audio", pcm=[0.0] * 100)

            fake_tts.__aiter__ = lambda self: tts_iterator()
            fake_tts.received_samples = 0
            fake_tts.received_samples_yielded = 0
            fake_tts.shutdown = AsyncMock()
            handler.tts_output_stopwatch = MagicMock()
            handler.tts_output_stopwatch.stop.return_value = 0.5

            async def run_tts_loop():
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
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            from unmute.tts.text_to_speech import TTSTextMessage
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [{"role": "system", "content": "test"}]
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0)

            fake_tts = MagicMock()

            async def tts_iterator():
                yield TTSTextMessage(type="Text", text="", start_s=0, stop_s=0)

            fake_tts.__aiter__ = lambda self: tts_iterator()
            fake_tts.received_samples = 0
            fake_tts.received_samples_yielded = 0
            fake_tts.shutdown = AsyncMock()
            handler.tts_output_stopwatch = MagicMock()
            handler.tts_output_stopwatch.stop.return_value = None

            async def run_tts_loop():
                task = asyncio.create_task(handler._tts_loop(fake_tts, 1))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await run_tts_loop()


# ─────────────────────────────────────────────────────────────────────────────
# _generate_response_task
# ─────────────────────────────────────────────────────────────────────────────


class TestUnmuteHandlerGenerateResponseTask:
    @pytest.mark.asyncio
    async def test_generate_response_task_emits_response_created(self):
        p1, p2 = _patch_handler_deps()
        with p1, p2:
            import unmute.openai_realtime_api_events as ora
            from unmute.unmute_handler import UnmuteHandler

            handler = UnmuteHandler()
            handler.chatbot = MagicMock()
            handler.chatbot.chat_history = [{"role": "system", "content": "Be nice"}]
            handler.chatbot.add_chat_message_delta = lambda delta, role, **kw: asyncio.sleep(0) or False
            handler.chatbot.preprocessed_messages.return_value = [
                {"role": "system", "content": "Be nice"},
                {"role": "user", "content": "hi"},
            ]
            handler.tts_voice = None

            fake_tts = MagicMock()
            fake_tts.send = AsyncMock()

            async def fake_tts_init():
                return fake_tts

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
                        task = asyncio.create_task(handler._generate_response_task())
                        await asyncio.sleep(0.2)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    await run_task()

            assert not handler.output_queue.empty()
            item = await handler.output_queue.get()
            assert isinstance(item, ora.ResponseCreated)
