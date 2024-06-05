import DL_classes
import functions as f
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


# ПРЕОБРАЗОВАНИЯ ВХОДНЫХ ДАННЫХ
# преобразование для обучение CNN
transform_cnn = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# преобразование для обучения FCNN
transform_fcnn = T.Compose([
    T.Resize((150, 150)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ФОРМИРОВАНИЕ ДАТАСЕТОВ
train_dataset = DL_classes.MyDataset(
    root=r'C:\Users\Nick\HOG\try1\Vegetable_Images_EDA\train',
    load_to_ram=0,
    transform=transform_fcnn
)
val_dataset = DL_classes.MyDataset(
    root=r'C:\Users\Nick\HOG\try1\Vegetable_Images_EDA\validation',
    load_to_ram=0,
    transform=transform_fcnn
)
test_dataset = DL_classes.MyDataset(
    root=r'C:\Users\Nick\HOG\try1\Vegetable_Images_EDA\test',
    load_to_ram=0,
    transform=transform_fcnn
)


# ФОРМИРОВАНИЕ ДАТАЛОДЕРОВ
train_loader = DataLoader(
    train_dataset, batch_size=32, drop_last=True,
    shuffle=True, pin_memory=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=32, drop_last=True,
    shuffle=False, pin_memory=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=32, drop_last=True,
    shuffle=False, pin_memory=True, num_workers=0
)


# Число эпох обучения
num_epochs = 50
# Расчет на GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Выбор модели
model = DL_classes.ChooseYourModel('imagenet_weights').model
model.to(device)
# Оптимайзер
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Функция потерь
criterion = torch.nn.CrossEntropyLoss()
# Расписание для learning_rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


# ПРОЦЕСС ОБУЧЕНИЯ И ВАЛИДАЦИИ
train_losses, val_losses, train_accuracies, val_accuracies = f.train(
    model, optimizer, scheduler, criterion, train_loader,
    val_loader, num_epochs, save='imagenet_weights',
    device=device
)
# Вывод макс точности на трейне и валидации
print('Макс.точность на train: ', max(train_accuracies))
print('Макс.точность на validation: ', max(val_accuracies))


# ПРОВЕРКА МОДЕЛИ НА ТЕСТЕ
# Инициализация модели
model = DL_classes.ChooseYourModel('imagenet_weights').model
# Загрузка обученной модеил
ckpt = torch.load('imagenet_weights.pt')
model.load_state_dict(ckpt['model'])
# Функция потерь
criterion = torch.nn.CrossEntropyLoss()
# Тестирование
test_acc = f.validation_epoch(model, criterion, test_loader, device=False)[1]
print(f'Test accuracy:\t\t{round(test_acc * 100, 2)} %')
