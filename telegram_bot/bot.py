import config 
import logging
import numpy as np
import time
from aiogram import Dispatcher, Bot, types, executor
from io import BytesIO
from PIL import Image
from skimage import color
from skimage.feature import hog


# логирование
logging.basicConfig(level=logging.INFO)
# бот
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
# диспетчер
dp = Dispatcher(bot)

# хендлер на команду старт
@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Start: {user_id} {user_full_name} {time.asctime()}. Message: {message}')
    await message.answer(config.START_BOT_TEXT)

# хендлер на текстовое сообщение от пользователя    
@dp.message_handler(content_types=["text"])
async def answer_on_text(message: types.Message):
    try:
        user_id = message.from_user.id
        user_full_name = message.from_user.full_name
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Text has been sent. Message: {message["text"]}.')
        await message.reply(config.ANSWER_ON_TEXT)
    except:
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Error in main_handler. Message: {message["text"]}.')
        await message.reply("Something went wrong...")  

# хендлер на фото   
@dp.message_handler(content_types=['photo'])
async def photo_id(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Photo has been sent.')
    try:
        # загрузка фото и классификация изображенного на нем объекта
        io = BytesIO()
        await message.photo[-1].download(destination_file=io)
        image = Image.open(io)
        fd = hog(color.rgb2gray(image), 
            orientations=8, 
            pixels_per_cell=(16,16), 
            cells_per_block=(4, 4), 
            block_norm= 'L2'
        )
        fd = fd.reshape((1, fd.shape[0]))
        prediction = config.MODEL.predict(fd)
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Photo has been processed.')
        await message.answer('Овощ на фото - {}'.format(config.VEG_DICT[int(prediction[0])]))
    except:
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Exception while editing photo.')
        await message.answer('Something went wrong while editing photo.')
    
