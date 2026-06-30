"""Tests for recorder.py: Recorder, RecorderEvent."""

import json
from pathlib import Path

import pytest

import unmute.openai_realtime_api_events as ora
from unmute.recorder import Recorder, RecorderEvent, make_filename


class TestRecorderEvent:
    def test_create_event(self):
        event = ora.ResponseTextDelta(delta="hello")
        recorder_event = RecorderEvent(
            timestamp_wall=1234567890.0,
            event_sender="server",
            data=event,
        )
        assert recorder_event.event_sender == "server"
        assert recorder_event.data.type == "response.text.delta"

    def test_model_dump_json(self):
        event = ora.ResponseTextDelta(delta="hello")
        recorder_event = RecorderEvent(
            timestamp_wall=1234567890.0,
            event_sender="client",
            data=event,
        )
        json_str = recorder_event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["event_sender"] == "client"


class TestMakeFilename:
    def test_format(self):
        filename = make_filename()
        parts = filename.split("_")
        assert len(parts) == 3  # YYYY-MM-DD_HH-MM-SS_uuid
        # Check date-like prefix
        assert len(parts[0]) == 10  # YYYY-MM-DD

    def test_uniqueness(self):
        f1 = make_filename()
        f2 = make_filename()
        assert f1 != f2


class TestRecorder:
    @pytest.mark.asyncio
    async def test_add_event(self, tmp_path: Path):
        recorder = Recorder(tmp_path)
        event = ora.ResponseTextDelta(delta="hello")
        await recorder.add_event("server", event)
        await recorder.shutdown()

        # Check the file was created
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        # Check the file content
        with open(jsonl_files[0]) as f:
            line = f.readline().strip()
            data = json.loads(line)
            assert data["event_sender"] == "server"
            assert data["data"]["type"] == "response.text.delta"

    @pytest.mark.asyncio
    async def test_multiple_events(self, tmp_path: Path):
        recorder = Recorder(tmp_path)
        await recorder.add_event("server", ora.ResponseTextDelta(delta="hello"))
        await recorder.add_event("client", ora.SessionUpdate(
            session=ora.SessionConfig(
                instructions=None,
                voice="alloy",
                allow_recording=True,
            )
        ))
        await recorder.shutdown()

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        with open(jsonl_files[0]) as f:
            lines = f.readlines()
            assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_shutdown_no_consent(self, tmp_path: Path):
        recorder = Recorder(tmp_path)
        await recorder.add_event("server", ora.ResponseTextDelta(delta="hello"))
        await recorder.shutdown(keep_recording=False)

        # File should be deleted
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_consent(self, tmp_path: Path):
        recorder = Recorder(tmp_path)
        await recorder.add_event("server", ora.ResponseTextDelta(delta="hello"))
        await recorder.shutdown(keep_recording=True)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1

    @pytest.mark.asyncio
    async def test_add_event_without_open_file(self, tmp_path: Path):
        recorder = Recorder(tmp_path)
        # First add_event opens the file lazily
        await recorder.add_event("server", ora.ResponseTextDelta(delta="hello"))
        assert recorder.opened_file is not None

    @pytest.mark.asyncio
    async def test_shutdown_without_events(self, tmp_path: Path):
        recorder = Recorder(tmp_path)
        await recorder.shutdown()
        # Should not raise even if no events were added
