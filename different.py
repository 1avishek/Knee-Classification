import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset automatically
data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=827)

# Convert to tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Dataset + DataLoader
class HousingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(HousingDataset(X_train, y_train), batch_size=64, shuffle=False)
test_loader = DataLoader(HousingDataset(X_test, y_test), batch_size=64, shuffle=False)

# Model definition
torch.manual_seed(41)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')
        nn.init.constant_(self.fc3.bias, 0)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        total_loss += loss.item() * xb.size(0)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Avg loss = {total_loss / len(train_loader.dataset):.6f}")

# Answers
max_w = model.fc2.weight.max().item()
print(f"\nAnswer 1 (max weight in 2nd hidden layer): {max_w:.2f}")
print(f"Answer 2 (final training loss): {total_loss / len(train_loader.dataset):.3f}")

# Evaluate test loss
model.eval()
test_loss = 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        test_loss += loss.item() * xb.size(0)
print(f"Answer 3 (test loss): {test_loss / len(test_loader.dataset):.3f}")
