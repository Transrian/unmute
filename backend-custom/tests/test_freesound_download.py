"""Tests for tts/freesound_download.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from unmute.tts.freesound_download import (
    FreesoundSoundInstance,
    FreesoundVoiceSource,
    SoundPreviews,
    get_sound_id_from_url,
    to_filename_friendly,
)


class TestSoundPreviews:
    def test_create_with_underscores(self):
        previews = SoundPreviews(
            preview_hq_mp3="http://example.com/hq.mp3",
            preview_lq_mp3="http://example.com/lq.mp3",
            preview_hq_ogg="http://example.com/hq.ogg",
            preview_lq_ogg="http://example.com/lq.ogg",
        )
        assert previews.preview_hq_mp3 == "http://example.com/hq.mp3"

    def test_create_with_dashes(self):
        previews = SoundPreviews(
            **{
                "preview-hq-mp3": "http://example.com/hq.mp3",
                "preview-lq-mp3": "http://example.com/lq.mp3",
                "preview-hq-ogg": "http://example.com/hq.ogg",
                "preview-lq-ogg": "http://example.com/lq.ogg",
            }
        )
        # Aliases should map underscores to dashes
        assert previews.preview_hq_mp3 == "http://example.com/hq.mp3"


class TestToFilenameFriendly:
    def test_normal_text(self):
        assert to_filename_friendly("Hello World") == "hello-world"

    def test_special_characters_removed(self):
        assert to_filename_friendly("Hello, World!") == "hello-world"

    def test_multiple_spaces(self):
        assert to_filename_friendly("Hello    World") == "hello-world"

    def test_lowercase_conversion(self):
        assert to_filename_friendly("HELLO WORLD") == "hello-world"

    def test_already_friendly(self):
        assert to_filename_friendly("hello-world") == "hello-world"


class TestFreesoundSoundInstance:
    def test_create(self):
        instance = FreesoundSoundInstance(
            id=12345,
            name="Test Sound",
            username="testuser",
            license="http://creativecommons.org/publicdomain/zero/1.0/",
        )
        assert instance.id == 12345
        assert instance.name == "Test Sound"

    def test_get_filename(self):
        instance = FreesoundSoundInstance(
            id=12345,
            name="Test Sound!",
            username="testuser",
            license="http://creativecommons.org/publicdomain/zero/1.0/",
        )
        assert instance.get_filename() == "12345_test-sound.mp3"

    def test_get_filename_special_chars(self):
        instance = FreesoundSoundInstance(
            id=67890,
            name="Weird [Sound] (Amazing)!",
            username="testuser",
            license="http://creativecommons.org/publicdomain/zero/1.0/",
        )
        # Special chars removed ([ ] ( ) !), spaces become dashes
        assert instance.get_filename() == "67890_weird-sound-amazing.mp3"

    def test_exclude_previews(self):
        instance = FreesoundSoundInstance(
            id=12345,
            name="Test",
            username="testuser",
            license="http://creativecommons.org/publicdomain/zero/1.0/",
        )
        dumped = instance.model_dump()
        assert "previews" not in dumped


class TestFreesoundVoiceSource:
    def test_create(self):
        source = FreesoundVoiceSource(
            url="https://freesound.org/people/user/sounds/12345/",
            path_on_server="path/to/sound.mp3",
        )
        assert source.source_type == "freesound"
        assert source.url == "https://freesound.org/people/user/sounds/12345/"
        assert source.start_time is None

    def test_with_start_time(self):
        source = FreesoundVoiceSource(
            url="https://freesound.org/people/user/sounds/12345/",
            start_time=5,
            path_on_server="path/to/sound.mp3",
        )
        assert source.start_time == 5

    def test_with_sound_instance(self):
        instance = FreesoundSoundInstance(
            id=12345,
            name="Test",
            username="testuser",
            license="http://creativecommons.org/publicdomain/zero/1.0/",
        )
        source = FreesoundVoiceSource(
            url="https://freesound.org/people/user/sounds/12345/",
            sound_instance=instance,
            path_on_server="path/to/sound.mp3",
        )
        assert source.sound_instance.id == 12345


class TestGetSoundIdFromUrl:
    def test_valid_url(self):
        assert get_sound_id_from_url(
            "https://freesound.org/people/balloonhead/sounds/785958/"
        ) == 785958

    def test_url_with_query_params(self):
        assert get_sound_id_from_url(
            "https://freesound.org/people/balloonhead/sounds/785958/?utm=something"
        ) == 785958

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="Invalid Freesound URL"):
            get_sound_id_from_url("https://freesound.org/invalid")

    def test_invalid_url_no_sounds_path(self):
        with pytest.raises(ValueError, match="Invalid Freesound URL"):
            get_sound_id_from_url("https://freesound.org/people/user/")


class TestGetSoundInstance:
    def test_from_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": 785958,
            "name": "Test Sound",
            "username": "testuser",
            "license": "http://creativecommons.org/publicdomain/zero/1.0/",
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.tts.freesound_download.requests.get", return_value=mock_response),
            patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}),
        ):
            from unmute.tts.freesound_download import get_sound_instance

            result = get_sound_instance(
                "https://freesound.org/people/user/sounds/785958/"
            )
            assert result.id == 785958

    def test_from_id(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": 12345,
            "name": "Test Sound",
            "username": "testuser",
            "license": "http://creativecommons.org/publicdomain/zero/1.0/",
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.tts.freesound_download.requests.get", return_value=mock_response),
            patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}),
        ):
            from unmute.tts.freesound_download import get_sound_instance

            result = get_sound_instance(12345)
            assert result.id == 12345

    def test_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with (
            patch("unmute.tts.freesound_download.requests.get", return_value=mock_response),
            patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}),
        ):
            from unmute.tts.freesound_download import get_sound_instance

            with pytest.raises(Exception, match="404 Not Found"):
                get_sound_instance(99999)


class TestDownloadSound:
    def test_license_check(self, tmp_path: Path):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": 12345,
            "name": "Test Sound",
            "username": "testuser",
            "license": "http://creativecommons.org/licenses/by-nc/4.0/",  # Non-commercial!
            "previews": {
                "preview-hq-mp3": "http://example.com/test.mp3",
                "preview-lq-mp3": "http://example.com/test-lq.mp3",
                "preview-hq-ogg": "http://example.com/test.ogg",
                "preview-lq-ogg": "http://example.com/test-lq.ogg",
            },
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("unmute.tts.freesound_download.requests.get", return_value=mock_response),
            patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}),
            patch("unmute.tts.freesound_download.OUTPUT_DIR", tmp_path),
        ):
            from unmute.tts.freesound_download import download_sound

            with pytest.raises(ValueError, match="license"):
                download_sound("https://freesound.org/people/user/sounds/12345/")
