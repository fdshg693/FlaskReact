import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) データ取得
iris = load_iris()
X, y = iris.data, iris.target

# 2) 訓練／テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3) スケーリング
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4) TensorDataset化
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long))
test_ds  = TensorDataset(
    torch.tensor(X_test,  dtype=torch.float32),
    torch.tensor(y_test,  dtype=torch.long))

# 5) DataLoader化
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

class SimpleNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
criterion = nn.CrossEntropyLoss()      # 分類タスクなので交差エントロピー損失
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9)

# 6) 学習
num_epochs = 20
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:2d}  Loss: {avg_loss:.4f}")

# 7) 評価
model.eval()
correct = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy:.3f}")
# # 8) モデル保存
# torch.save(model.state_dict(), "iris_model.pth")
# # 9) モデル読み込み
# model = SimpleNet()
# model.load_state_dict(torch.load("iris_model.pth"))