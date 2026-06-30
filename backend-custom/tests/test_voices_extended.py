"""Extended tests for tts/voices.py to increase coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestVoiceSampleDiscriminator:
    def test_file_source_discriminator(self):
        from unmute.tts.voices import FileVoiceSource, VoiceSample

        voice = VoiceSample(
            name="Test",
            source=FileVoiceSource(path_on_server="/test.wav"),
        )
        assert isinstance(voice.source, FileVoiceSource)

    def test_freesound_source_discriminator(self):
        from unmute.tts.voices import FreesoundVoiceSource, VoiceSample

        voice = VoiceSample(
            name="Test",
            source=FreesoundVoiceSource(
                url="https://freesound.org/sounds/12345/",
                path_on_server="free/test.wav",
                sound_instance={"id": 12345, "name": "test", "username": "user", "license": "CC-BY"},
            ),
        )
        assert isinstance(voice.source, FreesoundVoiceSource)


class TestVoiceListSave:
    def test_save_excludes_none_values(self, tmp_path: Path):
        """Saved YAML should exclude None values."""
        from unmute.tts.voices import FileVoiceSource, VoiceList, VoiceSample

        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text("")

        vl = VoiceList()
        vl.path = voices_yaml
        vl.voices = [
            VoiceSample(name="Test", good=True, source=FileVoiceSource(path_on_server="a.wav")),
        ]

        vl.save()

        content = voices_yaml.read_text()
        assert "None" not in content

    def test_save_preserves_order(self, tmp_path: Path):
        """Save should sort voices: good first, then undecided, then bad."""
        from unmute.tts.voices import FileVoiceSource, VoiceList, VoiceSample

        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text("")

        vl = VoiceList()
        vl.path = voices_yaml
        vl.voices = [
            VoiceSample(name="Bad", good=False, source=FileVoiceSource(path_on_server="b.wav")),
            VoiceSample(name="Undecided", good=None, source=FileVoiceSource(path_on_server="c.wav")),
            VoiceSample(name="Good", good=True, source=FileVoiceSource(path_on_server="a.wav")),
        ]

        vl.save()

        from ruamel.yaml import YAML

        with voices_yaml.open() as f:
            saved = YAML().load(f)

        assert saved[0]["name"] == "Good"
        assert saved[1]["name"] == "Undecided"
        assert saved[2]["name"] == "Bad"


class TestTtsMessageAdapter:
    def test_validates_text_message(self):
        from unmute.tts.text_to_speech import TTSMessageAdapter, TTSTextMessage

        msg = TTSMessageAdapter.validate_python(
            {"type": "Text", "text": "Hello", "start_s": 0.0, "stop_s": 1.0}
        )
        assert isinstance(msg, TTSTextMessage)
        assert msg.text == "Hello"

    def test_validates_audio_message(self):
        from unmute.tts.text_to_speech import TTSMessageAdapter, TTSAudioMessage

        msg = TTSMessageAdapter.validate_python(
            {"type": "Audio", "pcm": [0.1, 0.2, 0.3]}
        )
        assert isinstance(msg, TTSAudioMessage)
        assert msg.pcm == [0.1, 0.2, 0.3]

    def test_validates_error_message(self):
        from unmute.tts.text_to_speech import TTSMessageAdapter, TTSErrorMessage

        msg = TTSMessageAdapter.validate_python(
            {"type": "Error", "message": "Something went wrong"}
        )
        assert isinstance(msg, TTSErrorMessage)

    def test_validates_ready_message(self):
        from unmute.tts.text_to_speech import TTSMessageAdapter, TTSReadyMessage

        msg = TTSMessageAdapter.validate_python({"type": "Ready"})
        assert isinstance(msg, TTSReadyMessage)


class TestTtsClientMessageAdapter:
    def test_validates_text_client_message(self):
        from unmute.tts.text_to_speech import TTSClientMessageAdapter, TTSClientTextMessage

        msg = TTSClientMessageAdapter.validate_python(
            {"type": "Text", "text": "Hello"}
        )
        assert isinstance(msg, TTSClientTextMessage)

    def test_validates_voice_client_message(self):
        from unmute.tts.text_to_speech import (
            TTSClientMessageAdapter,
            TTSClientVoiceMessage,
        )

        msg = TTSClientMessageAdapter.validate_python(
            {"type": "Voice", "embeddings": [0.1, 0.2], "shape": [10]}
        )
        assert isinstance(msg, TTSClientVoiceMessage)

    def test_validates_eos_client_message(self):
        from unmute.tts.text_to_speech import TTSClientMessageAdapter, TTSClientEosMessage

        msg = TTSClientMessageAdapter.validate_python({"type": "Eos"})
        assert isinstance(msg, TTSClientEosMessage)


class TestAudioBufferSec:
    def test_audio_buffer_sec_constant(self):
        from unmute.tts.text_to_speech import AUDIO_BUFFER_SEC
        from unmute.kyutai_constants import FRAME_TIME_SEC

        assert AUDIO_BUFFER_SEC == FRAME_TIME_SEC * 4


class TestTTSMessageTypes:
    def test_message_type_discriminator(self):
        from unmute.tts.text_to_speech import TTSMessage, TTSTextMessage

        # Test that the type discriminator works
        msg: TTSMessage = TTSTextMessage(type="Text", text="Hello", start_s=0, stop_s=1)
        assert isinstance(msg, TTSTextMessage)
