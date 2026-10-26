import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(
    root="src/data/train",
    transform=train_transform
)

val_dataset = ImageFolder(
    root="src/data/val",
    transform=val_transform
)

last_dataset = ImageFolder(
    root="src/data/last",
    transform=val_transform
)


class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Primeiro bloco
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Segundo bloco
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(64 * 28 * 28, 1)


    def forward(self, x):

        # Bloco 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Bloco 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x



#INITIALIZE

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
   val_dataset,
    batch_size=64,
    shuffle=False
)

last_loader = DataLoader(
   last_dataset,
    batch_size=2,
    shuffle=False
)

model = CatDogCNN().to(device)
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

start_epoch = checkpoint["epoch"] + 1

print("A continuar da epoch:", start_epoch)
print("Última Train Loss:", checkpoint["train_loss"])
print("Última Val Loss:", checkpoint["val_loss"])

epochs = 100
best_val_loss = float("inf")

#model.eval()

#for images, labels in last_loader:
   # images = images.to(device)
    #labels = labels.float().unsqueeze(1).to(device)

            # 1. Forward
    #logits = model(images)
    #probabilities = torch.sigmoid(logits)
    #predictions = (probabilities >= 0.5).float()
   # print("Probabilidades de cão:", probabilities)
   # print("Predictions:", predictions)

for epoch in range(epochs):
    
    model.train()

    train_loss = 0
    val_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # 1. Forward
        logits = model(images)

        # 2. Loss
        loss = loss_fn(logits, labels)

        # 3. Limpar gradientes anteriores
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Atualizar pesos
        optimizer.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Batch {batch_idx + 1}/{len(train_loader)}"
        )

    average_train_loss = train_loss / len(train_loader)

    model.eval()

    with torch.no_grad():

        correct = 0
        total = 0       

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            # 1. Forward
            logits = model(images)

            # 2. Loss
            loss = loss_fn(logits, labels)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.numel()

            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total * 100

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss

        torch.save({
            "epoch": epoch + start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": average_train_loss,
            "val_loss": average_val_loss
        }, "best_checkpoint.pth")

        print("✅ Novo melhor modelo guardado")

    torch.save({
        "epoch": epoch + start_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": average_train_loss,
        "val_loss": average_val_loss
    }, "checkpoint.pth")


    print(
    f"Epoch {epoch + 1}/{epochs} - "
    f"Train Loss: {average_train_loss:.4f} - "
    f"Val Loss: {average_val_loss:.4f} - "
    f"Val Accuracy: {val_accuracy:.2f}%"
    )