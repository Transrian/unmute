"""Tests for llm/system_prompt.py: All instruction types and system prompt generation."""

from unittest.mock import patch

from unmute.llm.system_prompt import (
    ANIMALS_EASY,
    ANIMALS_HARD,
    CONVERSATION_STARTER_SUGGESTIONS,
    ConstantInstructions,
    GuessAnimalInstructions,
    NewsInstructions,
    QuizShowInstructions,
    SmalltalkInstructions,
    UnmuteExplanationInstructions,
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


class TestGuessAnimalInstructions:
    def test_type(self):
        instructions = GuessAnimalInstructions()
        assert instructions.type == "guess_animal"

    def test_make_system_prompt_has_animal(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = GuessAnimalInstructions()
            prompt = instructions.make_system_prompt()
            # Should contain an animal from the easy list
            found = any(animal in prompt for animal in ANIMALS_EASY)
            assert found

    def test_make_system_prompt_has_hard_animal(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = GuessAnimalInstructions()
            prompt = instructions.make_system_prompt()
            found = any(animal in prompt for animal in ANIMALS_HARD)
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


class TestNewsInstructions:
    def test_type(self):
        instructions = NewsInstructions()
        assert instructions.type == "news"

    def test_make_system_prompt_fallback(self):
        """When news API fails, it should fall back to smalltalk."""
        with (
            patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"),
            patch("unmute.llm.system_prompt.get_news", return_value=None),
        ):
            instructions = NewsInstructions()
            prompt = instructions.make_system_prompt()
            assert "error" in prompt.lower() or "something else" in prompt.lower()

    def test_make_system_prompt_with_news(self):
        from unmute.llm.newsapi import Article, NewsResponse, Source

        fake_news = NewsResponse(
            status="ok",
            totalResults=1,
            articles=[
                Article(
                    source=Source(id="verge", name="The Verge"),
                    author="Test",
                    title="Test Article",
                    description="Test description",
                    publishedAt="2024-01-01",
                    content="Test content",
                )
            ],
        )

        with (
            patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"),
            patch("unmute.llm.system_prompt.get_news", return_value=fake_news),
        ):
            instructions = NewsInstructions()
            prompt = instructions.make_system_prompt()
            assert "Test Article" in prompt


class TestUnmuteExplanationInstructions:
    def test_type(self):
        instructions = UnmuteExplanationInstructions()
        assert instructions.type == "unmute_explanation"

    def test_make_system_prompt(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = UnmuteExplanationInstructions()
            prompt = instructions.make_system_prompt()
            assert "test model" in prompt
            assert "Kyutai" in prompt
            assert "speech-to-text" in prompt

    def test_make_system_prompt_has_unmute_explanation(self):
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            instructions = UnmuteExplanationInstructions()
            prompt = instructions.make_system_prompt()
            assert "modular AI system" in prompt


class TestSystemPromptTemplate:
    def test_common_elements(self):
        """All instruction types should produce prompts with common elements."""
        with patch("unmute.llm.system_prompt.autoselect_model", return_value="test-model"):
            for cls in [
                ConstantInstructions,
                SmalltalkInstructions,
                GuessAnimalInstructions,
                QuizShowInstructions,
                UnmuteExplanationInstructions,
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
