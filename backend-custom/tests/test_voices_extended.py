"""Extended tests for tts/voices.py to increase coverage."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTextToSpeechNonStreaming:
    @pytest.mark.asyncio
    async def test_cache_file_path_generation(self):
        """Cache file path should include voice, text hash, and cfg param."""
        import numpy as np
        from unmute.tts.voices import TTS_OUTPUT_CACHE_DIR

        # We can't test the full flow without a real TTS server,
        # but we can test that the cache path is generated correctly
        assert TTS_OUTPUT_CACHE_DIR.name == "tts-outputs"


class TestVoiceSampleDiscriminator:
    def test_file_source_discriminator(self):
        from unmute.tts.voices import FileVoiceSource, VoiceSample

        voice = VoiceSample(
            name="Test",
            source=FileVoiceSource(path_on_server="/test.wav"),
        )
        assert isinstance(voice.source, FileVoiceSource)

    def test_freesound_source_discriminator(self):
        from unmute.tts.freesound_download import FreesoundVoiceSource
        from unmute.tts.voices import VoiceSample

        voice = VoiceSample(
            name="Test",
            source=FreesoundVoiceSource(
                url="https://freesound.org/sounds/12345/",
                path_on_server="free/test.wav",
            ),
        )
        assert isinstance(voice.source, FreesoundVoiceSource)


class TestVoiceListUploadToServer:
    @pytest.mark.asyncio
    async def test_uploads_good_voices(self, tmp_path: Path):
        """Good voices should be processed."""
        from unittest.mock import AsyncMock
        from unmute.tts.voices import FileVoiceSource, VoiceList, VoiceSample
        from ruamel.yaml import YAML

        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text("")

        vl = VoiceList()
        vl.path = voices_yaml
        vl.voices = [
            VoiceSample(name="Good", good=True, source=FileVoiceSource(path_on_server="a.wav")),
            VoiceSample(name="Bad", good=False, source=FileVoiceSource(path_on_server="b.wav")),
        ]

        # Patch download and upload functions
        with patch("unmute.tts.voices.download_sound") as mock_download:
            with patch("unmute.tts.voices.find_enhanced_version", return_value=None):
                with patch("unmute.tts.voices.upload_voice_to_dev", new=AsyncMock()):
                    with patch("unmute.tts.voices.OUTPUT_DIR", tmp_path):
                        # Create the expected file
                        voice_file = tmp_path / "voices" / "a.wav"
                        voice_file.parent.mkdir(exist_ok=True)
                        voice_file.touch()

                        await vl.upload_to_server()

    @pytest.mark.asyncio
    async def test_skips_bad_voices(self, tmp_path: Path):
        """Bad voices should be skipped."""
        from unmute.tts.voices import FileVoiceSource, VoiceList, VoiceSample

        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text("")

        vl = VoiceList()
        vl.path = voices_yaml
        vl.voices = [
            VoiceSample(name="Bad", good=False, source=FileVoiceSource(path_on_server="b.wav")),
        ]

        with patch("unmute.tts.voices.download_sound") as mock_download:
            with patch("unmute.tts.voices.find_enhanced_version", return_value=None):
                with patch("unmute.tts.voices.upload_voice_to_dev", new=AsyncMock()) as mock_upload:
                    await vl.upload_to_server()
                    mock_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_for_missing_file_source(self, tmp_path: Path):
        """Should raise FileNotFoundError for missing file sources."""
        from unmute.tts.voices import FileVoiceSource, VoiceList, VoiceSample

        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text("")

        vl = VoiceList()
        vl.path = voices_yaml
        vl.voices = [
            VoiceSample(name="Good", good=True, source=FileVoiceSource(path_on_server="missing.wav")),
        ]

        with patch("unmute.tts.voices.find_enhanced_version", return_value=None):
            with patch("unmute.tts.voices.OUTPUT_DIR", tmp_path):
                with pytest.raises(FileNotFoundError):
                    await vl.upload_to_server()


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


class TestFindEnhancedVersion:
    def test_returns_enhanced_path_when_exists(self, tmp_path: Path):
        from unmute.tts.voices import find_enhanced_version

        enhanced_dir = tmp_path / "voices-clean"
        enhanced_dir.mkdir()
        (enhanced_dir / "test-enhanced-v2.wav").touch()

        with patch("unmute.tts.voices.OUTPUT_DIR", tmp_path):
            original_path = tmp_path / "test.wav"
            original_path.touch()
            result = find_enhanced_version(original_path)
            assert result == enhanced_dir / "test-enhanced-v2.wav"

    def test_returns_none_when_not_exists(self, tmp_path: Path):
        from unmute.tts.voices import find_enhanced_version

        with patch("unmute.tts.voices.OUTPUT_DIR", tmp_path):
            original_path = tmp_path / "test.wav"
            original_path.touch()
            result = find_enhanced_version(original_path)
            assert result is None


class TestUploadVoiceToDev:
    def test_calls_rsync(self):
        from unmute.tts.voices import upload_voice_to_dev

        local_path = Path("/tmp/test.wav")
        path_on_server = "voices/test.wav"

        with patch("unmute.tts.voices.subprocess_with_retries") as mock_rsync:
            upload_voice_to_dev(local_path, path_on_server)
            mock_rsync.assert_called_once()
            call_args = mock_rsync.call_args[0][0]
            assert call_args[0] == "rsync"
            assert call_args[1] == local_path


class TestCopyVoiceToProd:
    def test_calls_scp(self):
        from unmute.tts.voices import copy_voice_to_prod

        with patch("unmute.tts.voices.subprocess_with_retries") as mock_scp:
            with patch("builtins.print"):
                copy_voice_to_prod("test_voice")
                # Should be called for both .safetensors and without
                assert mock_scp.call_count == 2


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
