import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary

# =============================
# 1. 数据加载
# =============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

# =============================
# 2. 模型定义（更深的 CNN）
# =============================
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
summary(DeepCNN(), (1, 28, 28), device="cpu")

# =============================
# 3. 初始化
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 正则化
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
l1_lambda = 1e-5  # L1 正则强度

# =============================
# 4. 训练循环
# =============================
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # 基础损失
        loss = criterion(output, target)

        # # 加 L1 正则化
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        # loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch}/10], Train Loss: {total_loss/len(train_loader):.6f}")

# =============================
# 5. 测试阶段
# =============================
model.eval()
correct = 0
test_loss = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

test_loss /= len(test_loader)
acc = 100. * correct / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%")
