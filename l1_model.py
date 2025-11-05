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
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

# =============================
# 2. 定义网络结构
# =============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
summary(SimpleCNN(), (1, 28, 28), device="cpu")

# =============================
# 3. 初始化模型、优化器和损失函数
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 正则化
criterion = nn.CrossEntropyLoss()

# L1 正则化系数
l1_lambda = 1e-5

# =============================
# 4. 训练循环
# =============================
for epoch in range(1, 6):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # 普通的交叉熵损失
        loss = criterion(output, target)

        # 加上 L1 正则化项
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        
        # # 加上 L2 正则化项
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + 1e-4 * l2_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.6f}')

# =============================
# 5. 测试
# =============================
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

print(f'Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
