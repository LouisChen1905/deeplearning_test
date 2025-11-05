import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchsummary import summary

# 使用 CPU
device = torch.device("cpu")
print("Using device:", device)

# 定义一个简单 CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = self.pool(self.relu(self.conv3(x)))
        # x = x.view(-1, 32 * 5 * 5)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
summary(Net(), (1, 28, 28), device="cpu")

# 数据集加载（自动下载到 ./data）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# 加载模型
try:
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn_cpu.pth", map_location=device))
    model.eval()
    print("Model loaded from mnist_cnn_cpu.pth")
except FileNotFoundError:
    model = None

if model is None:
    # ------------------------------------------------
    # 如果模型不存在，则训练一个简单的 CNN 模型
    # ------------------------------------------------
    print("Training a new model...")
    model = Net().to(device)
    print(model)

    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    l1_lambda = 1e-5  # L1 正则化系数
    l2_lambda = 1e-4  # L2 正则化系数
    # 训练
    for epoch in range(10):  # 跑 3 轮即可
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 加上 L1 正则化项
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            # 加上 L2 正则化项
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(f"[Epoch {epoch+1}, Step {i+1}] loss: {running_loss / 200:.4f}")
                running_loss = 0.0

    print("Finished Training")

    # save the trained model
    torch.save(model.state_dict(), "mnist_cnn_cpu.pth")
    print("Model saved to mnist_cnn_cpu.pth")

    # 测试准确率
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print(f"Accuracy on 10,000 test images: {100 * correct / total:.2f}%")
    
summary(model, (1, 28, 28))

# ------------------------------------------------
# 6️⃣ 可视化 conv1 特征图
# ------------------------------------------------
example_data, example_target = next(iter(test_loader))
image = example_data[4:5]  # 取一张图像
label = example_target[4].item()

# 仅经过第一层卷积
with torch.no_grad():
    conv1_output = F.relu(model.conv1(image))
    conv2_output = F.relu(model.conv2(conv1_output))

# 显示原图 + 8 个特征图
plt.figure(figsize=(10, 3))
plt.subplot(1, 9, 1)
plt.imshow(image[0, 0], cmap="gray")
plt.title(f"Input\nLabel:{label}")
plt.axis("off")

for i in range(8):
    plt.subplot(1, 9, i + 2)
    plt.imshow(conv1_output[0, i], cmap="gray")
    plt.title(f"F{i+1}")
    plt.axis("off")
    
plt.figure(figsize=(10, 3))
plt.subplot(1, 17, 1)
plt.imshow(image[0, 0], cmap="gray")
plt.title(f"Input\nLabel:{label}")
plt.axis("off")
for i in range(16):
    plt.subplot(1, 17, i + 2)
    plt.imshow(conv2_output[0, i], cmap="gray")
    plt.title(f"F{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
