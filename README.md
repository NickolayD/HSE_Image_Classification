# Классификация овощей по фотографиям
Репозиторий для выполнения годового проекта 1-го года магистерской программы НИУ ВШЭ "Машинное обучение и высоконагруженные системы".
## Команда:
  - [Дарьин Николай](https://github.com/NickolayD)
## Основные этапы проекта их описание
Задачей данного проекта является создание сервиса для распознавания изображений различных овощей по фотографии. 

В качестве исходного набора данных использовался датасет, содержащий 15 различных классов овощей. Более подробно узнать о датасете и представленных 
классах можно на сайте [kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/data).

### 1. Разведочный анализ данных (октябрь 2023 - 15 ноября 2023)
На первом этапе был проведен разведочный анализ данных. В нем были рассмотрены основные представители каждого класса и их основные параметры (размер, формат). Проведена проверка наличия в датасете выбросов и объектов, которые не соответствуют заявленным параметрам. Во всех трех наборах данных (обучающая, валидационная и тестовая выборки) отсутствует дисбаланс классов.
Его основные результаты представлены в папке "eda" данного репозитория.

### 2. Рассмотрение различных ML-моделей и разработка web-сервиса модели (15 ноября 2023 - 21 декабря 2023)
На втором этапе в качестве бейслайна для проведения классификации был выбран метод опорных векторов (SVM). Основная идея метода заключается в построении гиперплоскости, разделяющей объекты выборки оптимальным способом. Алгоритм работает в предположении, что чем больше расстояние между разделяющей гиперплоскостью и объектами разделяемых классов, тем меньше будет средняя ошибка классификатора. Для обучения использовались модели из [sklearn.svm](https://scikit-learn.org/stable/api/sklearn.svm.html). Лучшая метрика качества (accuracy), полученная на тестовой выборке с помощью метода опорных векторов, составляет 86 %. Более подробно о рассмотренных алгоритмах и полученных результатах можно узнать, обратившись к папке "ML_model".

Для взаимодействия стороннего пользователя с обученной SVM-моделью был написан web-сервис с помощью библиотеки [FastAPI](https://fastapi.tiangolo.com/).

### 3. Доработка ML-моделей и развитие web-сервиса (21 декабря 2023 - 4 марта 2024)
Проведена аналитика по всем рассмотреннным моделям классического машинного обучения. Модель, с помощью которой были получены максимальные значения метрик качества, была обернута в пайплайн.
Структура и код проекта  были приведены к общепринятой форме (PEP8). Также был развит web-сервис - теперь взаимодействие пользователя с моделью осуществляется через мессенджер [Telegram](https://web.telegram.org) посредством реализованного чат-бота. Бот быт написан с помощью библиотеки [AIOgram](https://docs.aiogram.dev/en/latest/). Для хостинга разработанного web-сервиса использовался облачный сервис [Render](https://render.com/).

### 4. Дальнейшая модернизация и развитие web-сервиса (4 марта 2024 - 31 апреля 2024)
На данном этапе был обновлен телеграм-бот (AIOgram v2 -> AIOgram v3). Также была произведена обертка всего сервиса в Docker-контейнер (папка "application" в данном репозитории), инструкции по запуску которого представлены ниже в разделе "Запуск проекта".

### 5. Разработка DL-моделей (31 апреля 2024 - 2 июня 2024)
На данном этапе были рассмотрены различные DL-модели, которые используемуются в задачах классификации объектов на изображении. Произведена оценка их точности и сравнение полученных результатов с результатами пайплайнов для этой задачи, представленных на платформе kaggle. 
В качестве фреймфорка была выбрана библиотека [PyTorch](https://pytorch.org/).

Были расмотрены 4 модели:
  - полносвязная нейронная сеть (accuracy на тесте 92,1 %);
  - сверточная нейронная сеть (accuracy на тесте 98,2 %);
  - MobileNet_v2 со случайными весами (accuracy на тесте 99,3 %);
  - предобученный MobileNet_v2 (accuracy на тесте 99,3 %).
Более побробно о структуре рассмотренных нейронных сетей, а также процессе их обучения, можно узнать, обратившись к папке "DL_models".

### 6.Дополнительно
В дополнение к рассматриваемой задаче была проведена работа с аналогичным датасетом - [fruit100](https://www.kaggle.com/datasets/marquis03/fruits-100).
Решалась аналогичная задача классификации фруктов на 100 возможных категорий.Наилучшая точность, которую дали рассмотренные модели - 82 %, что 
немного больше, чем макимальная точность в рассмотренных пайплайнах на kaggle.

## Пример работы сервиса
![](https://github.com/NickolayD/HSE_Image_Classification/blob/main/DL_models/Example.gif)

## Запуск проекта
Пререквизиты:
 - Docker;
 - Python 3.8 или выше (для запуска без Docker).

Запуск через Docker:
1. Клонировать репозиторий к себе на локальную машину
   ```
   git clone https://github.com/NickolayD/HSE_Image_Classification.git
   ```
2. Перейти в папку скачанного проекта
   ```
   cd HSE_Image_Classification
   ```
3. Не забудьте создать своего бота с помощью [BotFather](https://web.telegram.org/a/#93372553) и замените значение `BOT_TOKEN` в файле docker-compose.yaml
   на то, которое будет принадлежать вашему новому созданному боту;

3. Запустить контейнеры через docker-compose
   ```
   docker compose up
   ```
   Для запуска проекта в фоновом режиме можно использовать флаг `-d

Запуск без Docker:
1. Создайте виртуальную среду (опционально, но рекомендуется)
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Установите необходимые библиотеки
   ```
   pip install -r ./application/requirements.txt
   pip install -r ./bot/requirements.txt
   ```
3. В файле application/config.py присвоить переменной `TOKEN` значение, соответствующее вашему телеграм боту
4. В файле application/config.py изменить значение переменной `_APP_ADRESS` на `http://0.0.0.0:5001` 
5. Отдельно запустить FastAPI сервис и телеграм бота
   ```
   cd application
   python3 app.py
   ```
   ```
   cd bot
   python3 bot.py
   ```
