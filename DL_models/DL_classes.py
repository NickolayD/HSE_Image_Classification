import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MyDataset(Dataset):
    ''' Кастомный класс для работы с исходным датасетом '''
    def __init__(self, root, load_to_ram=True, transform=None):
        super().__init__()
        self.root = root
        self.load_to_ram = load_to_ram
        self.transform = transform
        self.all_files = []
        self.all_labels = []
        self.images = []
        self.classes = sorted(os.listdir(self.root))
        for i, class_name in enumerate(self.classes):
            files = sorted(os.listdir(os.path.join(self.root, class_name)))
            self.all_files += files
            self.all_labels += [i] * len(files)
            if self.load_to_ram:
                self.images += self._load_images(files, i)

    def _load_images(self, image_files, label):
        images = []
        for filename in image_files:
            image = Image.open(os.path.join(self.root,
                                            self.classes[label],
                                            filename)).convert('RGB')
            images += [image]
        return images

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, item):
        label = self.all_labels[item]
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.all_files[item]
            image = Image.open(os.path.join(self.root,
                                            self.classes[label],
                                            filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FCNN(torch.nn.Module):
    ''' Полносвязная нейронная сеть'''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 8384),
            torch.nn.ReLU(),
            torch.nn.Linear(8384, 4192),
            torch.nn.ReLU(),
            torch.nn.Linear(4192, 2096),
            torch.nn.ReLU(),
            torch.nn.Linear(2096, 1048),
            torch.nn.ReLU(),
            torch.nn.Linear(1048, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.net(x)
        return logits


class ConvNet(torch.nn.Module):
    ''' Сверточная нейронная сеть '''
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(  # (3, 224, 224)
            torch.nn.Conv2d(in_channels=3, out_channels=16,
                            kernel_size=3, padding=1),  # (16, 224, 224)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # (16, 112, 112)
            torch.nn.Conv2d(in_channels=16, out_channels=32,
                            kernel_size=3, padding=1),  # (32, 112, 112)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # (32, 56, 56)
            torch.nn.Conv2d(in_channels=32, out_channels=64,
                            kernel_size=3, padding=1),  # (64, 56, 56)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # (64, 28, 28)
            torch.nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, padding=1),  # (128, 14, 14)
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(in_features=128, out_features=15)

    def forward(self, x):  # (B, 3, 224, 224)
        feature_map = self.net(x)  # (B, 128, 14, 14)
        feature_vector = feature_map.mean(dim=(2, 3))  # (B, 128)
        logits = self.classifier(feature_vector)  # (B, 15)
        return logits


class ChooseYourModel:
    def __init__(self, model_name):
        if model_name == 'fcnn':
            self.model = FCNN(3*150*150, 15)
        elif model_name == 'cnn':
            self.model = ConvNet()
        elif model_name == 'imagenet':
            self.model = mobilenet_v2(num_classes=15)
        elif model_name == 'imagenet_weights':
            self.model = mobilenet_v2(
                weights=MobileNet_V2_Weights.IMAGENET1K_V1
            )
            self.model.classifier[1] = torch.nn.Linear(1280, 15)
        else:
            raise Exception('Wrong model name.')
