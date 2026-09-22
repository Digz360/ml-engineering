import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.profiler import profile, ProfilerActivity


class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 4)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(4, 1)

    def forward(self, X):
        Z1 = self.layer1(X)
        A1 = self.activation(Z1)
        Z2 = self.layer2(A1)

        return Z2


# REPRODUCIBILITY
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNetwork().to(device)
model = torch.compile(model)

print("Device:", device)

# 2500 PLANETS
X = torch.rand(2500, 2)

# Features:
# X[:, 0] = temperatura (0 = muito fria, 1 = muito quente)
# X[:, 1] = quantidade de água (0 = nenhuma, 1 = muita)

temperature = X[:, 0]
water = X[:, 1]

# REGRA "REAL" DO NOSSO UNIVERSO
habitable = (
    (temperature > 0.3) &
    (temperature < 0.7) &
    (water > 0.4)
)

Y = habitable.float().reshape(-1, 1)

# TRAIN / VALIDATION SPLIT

X_train = X[:2000]
Y_train = Y[:2000]

X_val = X[2000:]
Y_val = Y[2000:]


# DATASETS
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

# DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False
)

#SETUP
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
scaler = torch.amp.GradScaler("cuda")

import time

torch.cuda.synchronize()
start = time.perf_counter()
for epoch in range(100):

    # =====================
    # TRAIN
    # =====================

    model.train()
    train_loss = 0.0


    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
    
        #RESET GRADIENTS
        optimizer.zero_grad()

        #FORWARD
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            predictions = model(X_batch)
            loss = loss_fn(predictions, Y_batch)

        #BACKWARD AND UPDATE
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * X_batch.size(0)

    # MÉDIA DA LOSS DOS BATCHES DESTA EPOCH

    train_loss /= len(train_dataset)

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, Y_batch in val_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)
            predicted_classes = (probabilities >= 0.5).float()

            correct += (predicted_classes == Y_batch).sum().item()
            total += Y_batch.size(0)

            loss = loss_fn(logits, Y_batch)

            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_dataset)
    val_accuracy = correct / total


    if epoch % 10 == 0:
            print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f}"
            f"Val Accuracy = {val_accuracy * 100:.2f}%"
        )
torch.cuda.synchronize()
end = time.perf_counter()

print(f"Training time: {end - start:.4f}s")



