import settings
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from preparation import DataPreporation


# Подготовка данных
# - обучение
train = DataPreporation()
train.load_data(settings.path_to_train)
train.convert_to_gray()
train.get_hog_features()
x_train, y_train = train.get_dataset()
# - валидация
validation = DataPreporation()
validation.load_data(settings.path_to_validation)
validation.convert_to_gray()
validation.get_hog_features()
x_val, y_val = validation.get_dataset()
# - тест
test = DataPreporation()
test.load_data(settings.path_to_test)
test.convert_to_gray()
test.get_hog_features()
x_test, y_test = test.get_dataset()
# Инициализация модели
print(len(x_test), len(y_test))
model = svm.SVC(kernel='rbf',
                C=0.1,
                cache_size=500,
                max_iter=1000,
                random_state=42
                )
# Обучение модели
model.fit(x_train, y_train)
# Предсказание и метрики для валидационных данных
y_pred_val = model.predict(x_val)
print("Validation Accuracy: "+str(accuracy_score(y_val, y_pred_val)))
print(classification_report(y_val, y_pred_val))
# Предсказание и метрики для тестовых данных
y_pred_test = model.predict(x_test)
print("Test Accuracy: "+str(accuracy_score(y_test, y_pred_test)))
print(classification_report(y_test, y_pred_test))
