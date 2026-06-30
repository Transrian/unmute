"""Tests for tts/voices.py."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unmute.llm.system_prompt import ConstantInstructions


class TestVoiceSample:
    def test_create_with_file_source(self):
        from unmute.tts.voices import FileVoiceSource, VoiceSample

        voice = VoiceSample(
            name="Test Voice",
            comment="A test",
            good=True,
            instructions=ConstantInstructions(),
            source=FileVoiceSource(
                path_on_server="/path/to/voice.wav",
                description="Test description",
            ),
        )
        assert voice.name == "Test Voice"
        assert voice.good is True
        assert voice.source.path_on_server == "/path/to/voice.wav"

    def test_create_minimal(self):
        from unmute.tts.voices import FileVoiceSource, VoiceSample

        voice = VoiceSample(
            name="Test",
            source=FileVoiceSource(path_on_server="/voice.wav"),
        )
        assert voice.name == "Test"
        assert voice.good is None

    def test_forbids_extra_fields(self):
        from unmute.tts.voices import FileVoiceSource, VoiceSample

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            VoiceSample(
                name="Test",
                source=FileVoiceSource(path_on_server="/voice.wav"),
                extra_field="should fail",
            )


class TestFileVoiceSource:
    def test_default_source_type(self):
        from unmute.tts.voices import FileVoiceSource

        source = FileVoiceSource(path_on_server="/path/to/voice.wav")
        assert source.source_type == "file"
        assert source.description is None

    def test_with_description(self):
        from unmute.tts.voices import FileVoiceSource

        source = FileVoiceSource(
            path_on_server="/path.wav",
            description="A nice voice",
            description_link="https://example.com",
        )
        assert source.description == "A nice voice"
        assert source.description_link == "https://example.com"


class TestVoiceList:
    def test_load_voices(self, tmp_path: Path):
        from unmute.tts.voices import VoiceList

        # Create a minimal voices.yaml
        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text(
            """
- name: Test Voice
  good: true
  source:
    source_type: file
    path_on_server: /path/to/voice.wav
"""
        )

        with patch.object(VoiceList, "__init__", lambda self: None):
            vl = VoiceList()
            vl.path = voices_yaml
            from ruamel.yaml import YAML

            with voices_yaml.open() as f:
                from unmute.tts.voices import VoiceSample

                vl.voices = [VoiceSample(**sound) for sound in YAML().load(f)]

        assert len(vl.voices) == 1
        assert vl.voices[0].name == "Test Voice"
        assert vl.voices[0].good is True

    def test_save_voices_sorted(self, tmp_path: Path):
        from unmute.tts.voices import FileVoiceSource, VoiceList, VoiceSample

        voices_yaml = tmp_path / "voices.yaml"
        voices_yaml.write_text("")

        vl = VoiceList()
        vl.path = voices_yaml
        vl.voices = [
            VoiceSample(name="Good", good=True, source=FileVoiceSource(path_on_server="a.wav")),
            VoiceSample(name="Bad", good=False, source=FileVoiceSource(path_on_server="b.wav")),
            VoiceSample(name="Undecided", good=None, source=FileVoiceSource(path_on_server="c.wav")),
        ]

        vl.save()

        from ruamel.yaml import YAML

        with voices_yaml.open() as f:
            saved = YAML().load(f)

        # Should be sorted: good first, then undecided, then bad
        assert saved[0]["name"] == "Good"
        assert saved[1]["name"] == "Undecided"
        assert saved[2]["name"] == "Bad"



