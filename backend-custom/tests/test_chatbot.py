"""Tests for llm/chatbot.py."""

import pytest

from unmute.llm.chatbot import Chatbot
from unmute.llm.system_prompt import (
    ConstantInstructions,
    GuessAnimalInstructions,
    SmalltalkInstructions,
)


class TestChatbotInitialState:
    def test_has_system_prompt(self):
        chatbot = Chatbot()
        assert len(chatbot.chat_history) >= 1
        assert chatbot.chat_history[0]["role"] == "system"

    def test_initial_conversation_state(self):
        chatbot = Chatbot()
        assert chatbot.conversation_state() == "waiting_for_user"

    def test_initial_instructions_none(self):
        chatbot = Chatbot()
        assert chatbot.get_instructions() is None


class TestConversationState:
    def test_bot_speaking(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "assistant", "content": "Hello"})
        assert chatbot.conversation_state() == "bot_speaking"

    def test_user_speaking(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "user", "content": "Hi there"})
        assert chatbot.conversation_state() == "user_speaking"

    def test_waiting_for_user_empty_user_message(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "user", "content": "   "})
        assert chatbot.conversation_state() == "waiting_for_user"

    def test_waiting_for_user_after_user_turn(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "user", "content": ""})
        assert chatbot.conversation_state() == "waiting_for_user"

    def test_unknown_role_raises(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "invalid", "content": "test"})
        with pytest.raises(RuntimeError, match="Unknown role"):
            chatbot.conversation_state()


class TestAddChatMessageDelta:
    @pytest.mark.asyncio
    async def test_new_message(self):
        chatbot = Chatbot()
        is_new = await chatbot.add_chat_message_delta("Hello", "user")
        assert is_new is True
        assert chatbot.chat_history[-1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_continuation(self):
        chatbot = Chatbot()
        await chatbot.add_chat_message_delta("Hello", "user")
        is_new = await chatbot.add_chat_message_delta(" world", "user")
        assert is_new is False
        assert chatbot.chat_history[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_auto_space(self):
        chatbot = Chatbot()
        await chatbot.add_chat_message_delta("Hello", "user")
        await chatbot.add_chat_message_delta("world", "user")
        assert chatbot.chat_history[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_no_double_space(self):
        chatbot = Chatbot()
        await chatbot.add_chat_message_delta("Hello", "user")
        await chatbot.add_chat_message_delta(" world", "user")
        assert chatbot.chat_history[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_already_has_space(self):
        chatbot = Chatbot()
        await chatbot.add_chat_message_delta("Hello ", "user")
        await chatbot.add_chat_message_delta("world", "user")
        assert chatbot.chat_history[-1]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_switching_roles(self):
        chatbot = Chatbot()
        await chatbot.add_chat_message_delta("Hello", "user")
        is_new = await chatbot.add_chat_message_delta("Hi", "assistant")
        assert is_new is True
        assert len([m for m in chatbot.chat_history if m["role"] != "system"]) == 2

    @pytest.mark.asyncio
    async def test_empty_delta_continuation(self):
        chatbot = Chatbot()
        await chatbot.add_chat_message_delta("Hello", "user")
        is_new = await chatbot.add_chat_message_delta("", "user")
        assert is_new is False

    @pytest.mark.asyncio
    async def test_race_condition_guard(self):
        chatbot = Chatbot()
        # Set up a message
        chatbot.chat_history.append({"role": "assistant", "content": "Hello"})
        # Try to add with wrong generating_message_i
        is_new = await chatbot.add_chat_message_delta(" world", "assistant", generating_message_i=1)
        # Should not add because chat_history has more than generating_message_i entries
        assert is_new is False

    @pytest.mark.asyncio
    async def test_new_message_after_empty(self):
        chatbot = Chatbot()
        # First message creates a new message
        is_new = await chatbot.add_chat_message_delta("", "user")
        assert is_new is True  # New message (empty)
        # Second adds to the empty message, so it's new
        is_new = await chatbot.add_chat_message_delta("hello", "user")
        assert is_new is True  # last_message was empty


class TestPreprocessedMessages:
    def test_with_only_system_prompt(self):
        chatbot = Chatbot()
        messages = chatbot.preprocessed_messages()
        # Should have system + fake user "Hello!"
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_with_real_conversation(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "user", "content": "Hello"})
        chatbot.chat_history.append({"role": "assistant", "content": "Hi"})
        messages = chatbot.preprocessed_messages()
        assert len(messages) >= 3  # system + user + assistant at minimum
        assert messages[0]["role"] == "system"


class TestSetInstructions:
    def test_set_constant_instructions(self):
        chatbot = Chatbot()
        instructions = ConstantInstructions(text="Be nice.")
        chatbot.set_instructions(instructions)
        assert chatbot.get_instructions() is not None
        assert "Be nice." in chatbot.get_system_prompt()

    def test_set_smalltalk_instructions(self):
        chatbot = Chatbot()
        instructions = SmalltalkInstructions()
        chatbot.set_instructions(instructions)
        assert chatbot.get_instructions() is not None
        assert chatbot.get_instructions().type == "smalltalk"

    def test_set_instructions_updates_system_prompt(self):
        chatbot = Chatbot()
        old_prompt = chatbot.get_system_prompt()
        instructions = ConstantInstructions(text="Unique instruction string.")
        chatbot.set_instructions(instructions)
        assert chatbot.get_system_prompt() != old_prompt


class TestLastMessage:
    def test_last_message_exists(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "user", "content": "Hello"})
        chatbot.chat_history.append({"role": "assistant", "content": "Hi there"})
        assert chatbot.last_message("assistant") == "Hi there"

    def test_last_message_none(self):
        chatbot = Chatbot()
        assert chatbot.last_message("user") is None

    def test_last_message_ignores_empty(self):
        chatbot = Chatbot()
        chatbot.chat_history.append({"role": "user", "content": "Hello"})
        chatbot.chat_history.append({"role": "user", "content": "   "})
        assert chatbot.last_message("user") == "Hello"

    def test_last_message_ignores_system(self):
        chatbot = Chatbot()
        result = chatbot.last_message("system")
        # System prompt is never empty, so it should return it
        assert result is not None
