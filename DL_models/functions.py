import torch


# функция для обучения модели и расчета точности на train
def training_epoch(model, optimizer, criterion, train_loader, device=False):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    for images, labels in train_loader:
        if device:
            images = images.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


# функция для расчета точности на валидации
@torch.no_grad()
def validation_epoch(model, criterion, val_loader, device=False):
    val_loss, val_accuracy = 0.0, 0.0
    model.eval()
    for images, labels in val_loader:
        if device:
            images = images.to(device)
            labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        val_loss += loss.item() * images.shape[0]
        val_accuracy += (logits.argmax(dim=1) == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy


def train(model, optimizer, scheduler, criterion, train_loader,
          val_loader, num_epochs, save=False, device=False):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_validation = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            device=device
        )
        val_loss, val_accuracy = validation_epoch(
            model, criterion, val_loader,
            device=device
        )
        if scheduler is not None:
            scheduler.step()
        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        val_losses += [val_loss]
        val_accuracies += [val_accuracy]
        # сохранение модели
        if save and val_accuracies[-1] > best_validation:
            best_validation = val_accuracies[-1]
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, f'{save}.pt')
    return train_losses, val_losses, train_accuracies, val_accuracies
