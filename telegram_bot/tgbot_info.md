Telegram-бот реализован с помощью библиотек aiogram v2.25.1 + FastAPI.

[Ссылка](https://test-tgbot-service.onrender.com/) на хостинг. 
Для хостинга использовался дополнительный [репозиторий](ttps://github.com/NickolayD/TG_Bot_Classifier).

[Ссылка](https://web.telegram.org/a/#6944300570) на бота в telegram.

Доступный функционал:
- команда /start - выводит краткую информацию о боте и доступных командах;
- команда /rate - позволяет пользователю оценить работу бота по шкале от 0 до 5 (реализована кнопочная клавиатура);
- команда /stat - выводит статистику по работе бота (кол-во уникальных пользователей, рейтинг бота, кол-во сделанных предсказаний и доля правильных);

По умолчанию бот находится в режиме ожидания сообщения от пользователя, которое содержит изображение для классификации.
В ответ бот отправляет сообщение с предсказанием о принадлежности объекта на фото к некоторому классу.
Фото содержит inline-кнопки, с помощью которых пользователь может сказать, правильно ли был проклассифицирован объект.
Ответы пользователей записываются для статистики.

Пример работы telegram-бота представлен в файле tgbot_demonstration.gif