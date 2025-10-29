import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个简单的全连接网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. 创建模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 生成一些假数据（y = x1 + x2）
x = torch.randn(100, 2)
y = x.sum(dim=1, keepdim=True)

# 4. 简单训练几轮
for epoch in range(50):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d}, Loss = {loss.item():.6f}")

# 5. 测试
test = torch.tensor([[2.0, 3.0]])
print("Input:", test)
print("Predicted:", model(test).item())
