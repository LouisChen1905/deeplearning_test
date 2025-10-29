import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ 定义一个最小CNN
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)   # 输入1通道，输出8通道
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # 输入8通道，输出16通道

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# -----------------------------
# 2️⃣ 初始化模型与假输入
# -----------------------------
model = SimpleCNN()
model.eval()

# 造一张 28x28 的假灰度图（模拟MNIST）
x = torch.randn(1, 1, 28, 28)

# 只跑到 conv1 输出（第一层特征图）
with torch.no_grad():
    features = F.relu(model.conv1(x))

print("conv1 输出形状：", features.shape)  # [1, 8, 26, 26]

# -----------------------------
# 3️⃣ 可视化卷积结果
# -----------------------------
fig, axes = plt.subplots(1, 8, figsize=(15, 3))
for i in range(8):
    axes[i].imshow(features[0, i].numpy(), cmap='gray')
    axes[i].set_title(f'Feature {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
