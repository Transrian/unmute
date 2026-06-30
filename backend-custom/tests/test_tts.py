"""Tests for tts/text_to_speech.py: TTS message types, prepare_text_for_tts."""


from unmute.tts.text_to_speech import (
    TTSAudioMessage,
    TTSClientEosMessage,
    TTSClientTextMessage,
    TTSClientVoiceMessage,
    TTSErrorMessage,
    TTSReadyMessage,
    TTSTextMessage,
    TtsStreamingQuery,
    prepare_text_for_tts,
)


class TestTTSClientTextMessage:
    def test_create(self):
        msg = TTSClientTextMessage(text="Hello world")
        assert msg.type == "Text"
        assert msg.text == "Hello world"


class TestTTSClientVoiceMessage:
    def test_create(self):
        msg = TTSClientVoiceMessage(
            embeddings=[0.1, 0.2, 0.3],
            shape=[1, 3],
        )
        assert msg.type == "Voice"
        assert msg.embeddings == [0.1, 0.2, 0.3]


class TestTTSClientEosMessage:
    def test_create(self):
        msg = TTSClientEosMessage()
        assert msg.type == "Eos"


class TestTTSTextMessage:
    def test_create(self):
        msg = TTSTextMessage(
            type="Text",
            text="hello",
            start_s=0.0,
            stop_s=1.0,
        )
        assert msg.text == "hello"
        assert msg.start_s == 0.0
        assert msg.stop_s == 1.0

    def test_equality(self):
        msg1 = TTSTextMessage(type="Text", text="", start_s=0, stop_s=0)
        msg2 = TTSTextMessage(type="Text", text="", start_s=0, stop_s=0)
        assert msg1 == msg2


class TestTTSAudioMessage:
    def test_create(self):
        msg = TTSAudioMessage(
            type="Audio",
            pcm=[0.0, 0.1, 0.2],
        )
        assert msg.type == "Audio"
        assert len(msg.pcm) == 3


class TestTTSErrorMessage:
    def test_create(self):
        msg = TTSErrorMessage(
            type="Error",
            message="Service unavailable",
        )
        assert msg.type == "Error"
        assert msg.message == "Service unavailable"


class TestTTSReadyMessage:
    def test_create(self):
        msg = TTSReadyMessage(type="Ready")
        assert msg.type == "Ready"


class TestPrepareTextForTTS:
    def test_strip(self):
        assert prepare_text_for_tts("  hello  ") == "hello"

    def test_remove_asterisks(self):
        assert prepare_text_for_tts("hello *world*") == "hello world"

    def test_remove_underscores(self):
        assert prepare_text_for_tts("hello_world") == "helloworld"

    def test_remove_backticks(self):
        assert prepare_text_for_tts("`code` block") == "code block"

    def test_replace_curly_quotes(self):
        assert prepare_text_for_tts('"hello"') == '"hello"'
        assert prepare_text_for_tts('"hello"') == '"hello"'
        assert prepare_text_for_tts("'world'") == "'world'"
        assert prepare_text_for_tts("'world'") == "'world'"

    def test_remove_colon_spaces(self):
        assert prepare_text_for_tts("prefix : content") == "prefix content"

    def test_combined(self):
        result = prepare_text_for_tts('  "hello" : *world*  ')
        assert result == '"hello" world'

    def test_empty_string(self):
        assert prepare_text_for_tts("") == ""

    def test_no_special_chars(self):
        assert prepare_text_for_tts("hello world") == "hello world"


class TestTtsStreamingQuery:
    def test_defaults(self):
        query = TtsStreamingQuery()
        assert query.format == "PcmMessagePack"
        assert query.seed is None
        assert query.temperature is None

    def test_to_url_params(self):
        query = TtsStreamingQuery(
            voice="alloy",
            temperature=0.8,
        )
        params = query.to_url_params()
        assert params.startswith("?")
        assert "voice=alloy" in params
        assert "temperature=0.8" in params
        # None values should not appear
        assert "seed" not in params

    def test_to_url_params_empty(self):
        query = TtsStreamingQuery()
        params = query.to_url_params()
        assert params == "?format=PcmMessagePack"
