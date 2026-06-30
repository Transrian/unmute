"""Tests for llm/system_prompt.py: All instruction types and system prompt generation."""

from unittest.mock import patch

from unmute.llm.system_prompt import (
    CONVERSATION_STARTER_SUGGESTIONS,
    ConstantInstructions,
    QuizShowInstructions,
    SmalltalkInstructions,
    get_default_instructions,
    get_readable_llm_name,
)


class TestGetReadableLlmName:
    def test_simple_model_name(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            name = get_readable_llm_name()
            assert name == "test model"

    def test_model_with_slash(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="creator/model-name"):
            name = get_readable_llm_name()
            assert name == "model name"

    def test_model_with_underscores(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="my_model_v2"):
            name = get_readable_llm_name()
            assert name == "my model v2"

    def test_openrouter_preset(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="@preset/my-preset"):
            name = get_readable_llm_name()
            assert name == "my preset"


class TestGetDefaultInstructions:
    def test_returns_constant_instructions(self):
        instructions = get_default_instructions()
        assert isinstance(instructions, ConstantInstructions)


class TestConstantInstructions:
    def test_default_text(self):
        instructions = ConstantInstructions()
        assert instructions.type == "constant"
        assert "back and forth" in instructions.text

    def test_custom_text(self):
        instructions = ConstantInstructions(text="Always be helpful.")
        assert instructions.text == "Always be helpful."

    def test_make_system_prompt(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = ConstantInstructions()
            prompt = instructions.make_system_prompt()
            assert "test model" in prompt
            assert "Speak English" in prompt

    def test_make_system_prompt_french(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = ConstantInstructions(language="fr")
            prompt = instructions.make_system_prompt()
            assert "Speak French" in prompt
            assert "«" in prompt  # French guillemets


class TestSmalltalkInstructions:
    def test_type(self):
        instructions = SmalltalkInstructions()
        assert instructions.type == "smalltalk"

    def test_make_system_prompt(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = SmalltalkInstructions()
            prompt = instructions.make_system_prompt()
            assert "test model" in prompt
            assert "back and forth" in prompt

    def test_make_system_prompt_has_time(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = SmalltalkInstructions()
            prompt = instructions.make_system_prompt()
            # Should contain a time-like string
            assert "at " in prompt

    def test_make_system_prompt_has_starter(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = SmalltalkInstructions()
            prompt = instructions.make_system_prompt()
            # Should contain one of the conversation starter suggestions
            found = any(s in prompt for s in CONVERSATION_STARTER_SUGGESTIONS)
            assert found


class TestQuizShowInstructions:
    def test_type(self):
        instructions = QuizShowInstructions()
        assert instructions.type == "quiz_show"

    def test_make_system_prompt_has_questions(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = QuizShowInstructions()
            prompt = instructions.make_system_prompt()
            # Should contain numbered questions
            assert "1." in prompt
            assert "2." in prompt

    def test_make_system_prompt_jaopardy_theme(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = QuizShowInstructions()
            prompt = instructions.make_system_prompt()
            assert "quiz show" in prompt.lower()


class TestSystemPromptTemplate:
    def test_common_elements(self):
        """All instruction types should produce prompts with common elements."""
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            for cls in [
                ConstantInstructions,
                SmalltalkInstructions,
                QuizShowInstructions,
            ]:
                instructions = cls()
                prompt = instructions.make_system_prompt()
                assert "BASICS" in prompt
                assert "test model" in prompt
                assert "TRANSCRIPTION ERRORS" in prompt

    def test_french_guillemets_in_template(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = ConstantInstructions()
            prompt = instructions.make_system_prompt()
            assert "«" in prompt  # French guillemets should be in the template
