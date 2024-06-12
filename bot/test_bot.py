import pytest
import config
import requests
from bot import command_start_handler
from bot import command_help_handler
from bot import command_info_handler
from bot import command_check_handler
from bot import predict_by_photo
from bot import text_handler
from unittest.mock import MagicMock, patch
from aiogram_tests.test_utils import MagicMockWithAttributes
from aiogram import F
from aiogram.types import Message, PhotoSize
from aiogram.filters import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE


@pytest.mark.asyncio
async def test_command_start_handler():
    requester = MockedBot(MessageHandler(
        command_start_handler,
        Command(commands=["start"])
        )
    )
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == config.START_TEXT


@pytest.mark.asyncio
async def test_command_help_handler():
    requester = MockedBot(MessageHandler(
        command_help_handler,
        Command(commands=["help"])
        )
    )
    calls = await requester.query(MESSAGE.as_object(text="/help"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == config.HELP_TEXT


@pytest.mark.asyncio
async def test_command_info_handler():
    requester = MockedBot(MessageHandler(
        command_info_handler,
        Command(commands=["info"])
        )
    )
    calls = await requester.query(MESSAGE.as_object(text="/info"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == config.INFO_TEXT


@pytest.mark.asyncio
async def test_predict_by_photo():
    requester = MockedBot(MessageHandler(
            predict_by_photo,
            F.photo
        )
    )
    io = MagicMock()
    with patch("bot.predict_by_photo.requests.post") as mocked_post: 
        mocked_post.return_value.text = '{"Vegetable": "Tomato"}'
        message = Message(
            from_user=MagicMockWithAttributes(id=123, full_name="Test User"),
            photo=[PhotoSize(file_id="photo_id", file_unique_id="unique_id")]
        )
        calls = await requester.query(message.as_object())
        # Проверяем, что `bot.download` был вызван с правильными параметрами
        assert calls.download.fetchone().file_id == "photo_id"
        assert calls.download.fetchone().file_unique_id == "unique_id"
        # Проверяем, что ответ отправлен пользователю
        answer_message = calls.send_message.fetchone().text
        assert answer_message == config.PREDICT.format("Tomato") 


@pytest.mark.asyncio
async def test_command_check_handler():
    requester = MockedBot(MessageHandler(
        command_check_handler,
        Command(commands=["check"])
        )
    )
    calls = await requester.query(MESSAGE.as_object(text="/check"))
    r = requests.get(config._APP_ADRESS)
    if r.status_code == 200:
        answer_message = calls.send_message.fetchone().text
        assert answer_message == config.CHECK_STATUS_OK
    else:
        answer_message = calls.send_message.fetchone().text
        assert answer_message == config.CHECK_STATUS_FAIL


@pytest.mark.asyncio
async def test_text_handler():
    request = MockedBot(MessageHandler(text_handler))
    calls = await request.query(message=MESSAGE.as_object(text="Any Text"))
    answer_message = calls.send_messsage.fetchone()
    assert answer_message.text == config.REPLY_ON_TEXT