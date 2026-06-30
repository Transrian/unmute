"""Tests for tts/voice_donation.py."""

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from unmute.tts.voice_donation import (
    CONSTANT_PREFIX,
    VoiceDonationMetadata,
    VoiceDonationSubmission,
    VoiceDonationVerification,
    generate_verification,
    submit_voice_donation,
)


class TestVoiceDonationVerification:
    def test_model_fields(self):
        verification = VoiceDonationVerification(
            id="test-id",
            text="Some text",
            created_at_timetamp=1234567890.0,
        )
        assert verification.id == "test-id"
        assert verification.text == "Some text"


class TestVoiceDonationSubmission:
    def test_default_values(self):
        submission = VoiceDonationSubmission(
            email="test@example.com",
            nickname="TestUser",
            verification_id=uuid.uuid4(),
        )
        assert submission.format_version == "1.1"
        assert submission.license == "CC0"
        assert submission.transcription_from_client is None

    def test_with_transcription(self):
        submission = VoiceDonationSubmission(
            email="test@example.com",
            nickname="TestUser",
            verification_id=uuid.uuid4(),
            transcription_from_client="Some text",
        )
        assert submission.transcription_from_client == "Some text"


class TestGenerateVerification:
    def test_has_constant_prefix(self):
        verification = generate_verification()
        assert verification.text.startswith(CONSTANT_PREFIX)

    def test_has_uuid(self):
        verification = generate_verification()
        uuid.UUID(verification.id)  # Should not raise

    def test_has_timestamp(self):
        verification = generate_verification()
        assert verification.created_at_timetamp > 0

    def test_has_sentences(self):
        verification = generate_verification()
        # After the constant prefix, there should be two sentences
        after_prefix = verification.text[len(CONSTANT_PREFIX) + 1:]
        assert len(after_prefix.strip()) > 0


class TestSubmitVoiceDonation:
    @pytest.mark.asyncio
    async def test_file_too_small(self, tmp_path: Path):
        with (
            patch("unmute.tts.voice_donation.VOICE_DONATION_DIR", tmp_path),
            patch(
                "unmute.tts.voice_donation.voice_donation_verification_cache",
                _FakeVoiceDonationCache(),
            ),
        ):
            submission = VoiceDonationSubmission(
                email="test@example.com",
                nickname="TestUser",
                verification_id=uuid.uuid4(),
            )
            with pytest.raises(ValueError, match="too small"):
                submit_voice_donation(submission, b"tiny")

    def test_file_too_large(self, tmp_path: Path):
        large_file = b"\x00" * (5 * 1024 * 1024)  # 5 MB
        with (
            patch("unmute.tts.voice_donation.VOICE_DONATION_DIR", tmp_path),
            patch(
                "unmute.tts.voice_donation.voice_donation_verification_cache",
                _FakeVoiceDonationCache(),
            ),
        ):
            submission = VoiceDonationSubmission(
                email="test@example.com",
                nickname="TestUser",
                verification_id=uuid.uuid4(),
            )
            with pytest.raises(ValueError, match="too large"):
                submit_voice_donation(submission, large_file)

    def test_nickname_too_long(self, tmp_path: Path):
        # File size check happens before nickname check, so use a large enough file
        with (
            patch("unmute.tts.voice_donation.VOICE_DONATION_DIR", tmp_path),
            patch(
                "unmute.tts.voice_donation.voice_donation_verification_cache",
                _FakeVoiceDonationCache(),
            ),
        ):
            submission = VoiceDonationSubmission(
                email="test@example.com",
                nickname="A" * 31,
                verification_id=uuid.uuid4(),
            )
            # Need > 0.1 MB to pass the size check first
            with pytest.raises(ValueError, match="too long"):
                submit_voice_donation(submission, b"0" * (1024 * 1024))

    def test_verification_not_found(self, tmp_path: Path):
        empty_cache = _FakeVoiceDonationCache()
        with (
            patch("unmute.tts.voice_donation.VOICE_DONATION_DIR", tmp_path),
            patch(
                "unmute.tts.voice_donation.voice_donation_verification_cache",
                empty_cache,
            ),
        ):
            submission = VoiceDonationSubmission(
                email="test@example.com",
                nickname="TestUser",
                verification_id=uuid.uuid4(),
            )
            # Need > 0.1 MB to pass the size check first
            with pytest.raises(ValueError, match="Couldn't find verification"):
                submit_voice_donation(submission, b"0" * (1024 * 1024))


class TestVoiceDonationMetadata:
    def test_create(self):
        import datetime

        metadata = VoiceDonationMetadata(
            submission=VoiceDonationSubmission(
                email="test@example.com",
                nickname="TestUser",
                verification_id=uuid.uuid4(),
            ),
            verification=VoiceDonationVerification(
                id="test-id",
                text="Some text",
                created_at_timetamp=1234567890.0,
            ),
            timestamp=1234567890.0,
            timestamp_str="2024-01-01T00:00:00",
        )
        assert metadata.timestamp == 1234567890.0
        assert metadata.submission.email == "test@example.com"


class _FakeVoiceDonationCache:
    """Fake cache for testing voice donation."""

    def __init__(self, verification_id: str | None = None):
        self._data = {}
        if verification_id:
            verification = VoiceDonationVerification(
                id=verification_id,
                text="Test verification",
                created_at_timetamp=1234567890.0,
            )
            self._data[verification_id] = verification.model_dump_json()

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value):
        self._data[key] = value

    def delete(self, key: str):
        self._data.pop(key, None)

    def cleanup(self):
        pass
