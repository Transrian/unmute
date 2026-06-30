"""Tests for openai_realtime_api_events.py: Event models."""

import unmute.openai_realtime_api_events as ora


class TestRandomId:
    def test_format(self):
        result = ora.random_id("event")
        assert result.startswith("event_")
        assert len(result) == 6 + 21  # prefix + underscore + 21 chars

    def test_different_calls(self):
        id1 = ora.random_id("event")
        id2 = ora.random_id("event")
        assert id1 != id2


class TestBaseEvent:
    def test_cannot_instantiate_directly(self):
        from unmute.openai_realtime_api_events import BaseEvent
        from typing import Literal

        class ConcreteEvent(BaseEvent[Literal["test.type"]]):
            pass

        # Concrete subclasses should work
        event = ConcreteEvent()
        assert event.type == "test.type"


class TestError:
    def test_create_error(self):
        error = ora.Error(
            error=ora.ErrorDetails(
                type="invalid_request_error",
                message="Bad request",
            )
        )
        assert error.type == "error"
        assert error.error.type == "invalid_request_error"
        assert error.error.message == "Bad request"

    def test_error_details_optional_fields(self):
        details = ora.ErrorDetails(type="error", message="msg")
        assert details.code is None
        assert details.param is None
        assert details.details is None


class TestSessionConfig:
    def test_session_config_basic(self):
        config = ora.SessionConfig(
            instructions=None,
            voice="alloy",
            allow_recording=True,
        )
        assert config.voice == "alloy"
        assert config.allow_recording is True

    def test_session_config_with_instructions(self):
        from unmute.llm.system_prompt import ConstantInstructions

        config = ora.SessionConfig(
            instructions=ConstantInstructions(),
            voice=None,
            allow_recording=False,
        )
        assert config.instructions is not None
        assert config.allow_recording is False


class TestSessionUpdate:
    def test_session_update(self):
        event = ora.SessionUpdate(
            session=ora.SessionConfig(
                instructions=None,
                voice="alloy",
                allow_recording=True,
            )
        )
        assert event.type == "session.update"
        assert event.session.voice == "alloy"


class TestSessionUpdated:
    def test_session_updated(self):
        event = ora.SessionUpdated(
            session=ora.SessionConfig(
                instructions=None,
                voice="alloy",
                allow_recording=True,
            )
        )
        assert event.type == "session.updated"


class TestInputAudioBufferAppend:
    def test_audio_append(self):
        event = ora.InputAudioBufferAppend(audio="dGVzdA==")
        assert event.type == "input_audio_buffer.append"
        assert event.audio == "dGVzdA=="


class TestInputAudioBufferSpeechEvents:
    def test_speech_started(self):
        event = ora.InputAudioBufferSpeechStarted()
        assert event.type == "input_audio_buffer.speech_started"

    def test_speech_stopped(self):
        event = ora.InputAudioBufferSpeechStopped()
        assert event.type == "input_audio_buffer.speech_stopped"


class TestResponse:
    def test_response_in_progress(self):
        response = ora.Response(
            status="in_progress",
            voice="alloy",
            chat_history=[],
        )
        assert response.object == "realtime.response"
        assert response.status == "in_progress"

    def test_response_with_history(self):
        response = ora.Response(
            status="completed",
            voice="alloy",
            chat_history=[{"role": "user", "content": "Hello"}],
        )
        assert len(response.chat_history) == 1


class TestResponseCreated:
    def test_response_created(self):
        event = ora.ResponseCreated(
            response=ora.Response(
                status="in_progress",
                voice="alloy",
            )
        )
        assert event.type == "response.created"


class TestResponseTextDelta:
    def test_text_delta(self):
        event = ora.ResponseTextDelta(delta="hello")
        assert event.type == "response.text.delta"
        assert event.delta == "hello"


class TestResponseTextDone:
    def test_text_done(self):
        event = ora.ResponseTextDone(text="Full response text")
        assert event.type == "response.text.done"
        assert event.text == "Full response text"


class TestResponseAudioDelta:
    def test_audio_delta(self):
        event = ora.ResponseAudioDelta(delta="dGVzdA==")
        assert event.type == "response.audio.delta"


class TestResponseAudioDone:
    def test_audio_done(self):
        event = ora.ResponseAudioDone()
        assert event.type == "response.audio.done"


class TestTranscriptionDelta:
    def test_transcription_delta(self):
        event = ora.ConversationItemInputAudioTranscriptionDelta(
            delta="hello",
            start_time=0.5,
        )
        assert event.type == "conversation.item.input_audio_transcription.delta"
        assert event.delta == "hello"
        assert event.start_time == 0.5


class TestUnmuteResponseTextDeltaReady:
    def test_text_delta_ready(self):
        event = ora.UnmuteResponseTextDeltaReady(delta="hello")
        assert event.type == "unmute.response.text.delta.ready"


class TestUnmuteResponseAudioDeltaReady:
    def test_audio_delta_ready(self):
        event = ora.UnmuteResponseAudioDeltaReady(number_of_samples=960)
        assert event.type == "unmute.response.audio.delta.ready"
        assert event.number_of_samples == 960


class TestUnmuteInterruptedByVAD:
    def test_interrupted_by_vad(self):
        event = ora.UnmuteInterruptedByVAD()
        assert event.type == "unmute.interrupted_by_vad"


class TestUnmuteAnonymizedInput:
    def test_anonymized_input(self):
        event = ora.UnmuteInputAudioBufferAppendAnonymized(
            number_of_samples=960
        )
        assert event.type == "unmute.input_audio_buffer.append_anonymized"
        assert event.number_of_samples == 960


class TestEventModelDump:
    def test_event_model_dump_json(self):
        event = ora.ResponseTextDelta(delta="hello")
        import json

        data = json.loads(event.model_dump_json())
        assert data["type"] == "response.text.delta"
        assert data["delta"] == "hello"
