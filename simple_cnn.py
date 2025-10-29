import torch
import torch.nn as nn
import torch.optim as optim

# 使用 CPU
device = torch.device("cpu")
print("Running on:", device)

# 1. 定义一个最小 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 14 * 14, 10)  # 假设输入是 28x28
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 4 * 14 * 14)
        x = self.fc1(x)
        return x

# 2. 初始化
model = SimpleCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 生成假输入（100 张 28x28 灰度图像）
x = torch.randn(100, 1, 28, 28).to(device)
y = torch.randn(100, 10).to(device)

# 4. 训练几轮
for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss = {loss.item():.6f}")

# 5. 推理
test = torch.randn(1, 1, 28, 28).to(device)
pred = model(test)
print("Test output shape:", pred.shape)
