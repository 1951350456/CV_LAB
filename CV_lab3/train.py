import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

import pandas as pd

# 定义超参数
EPOCH = 5  # 训练迭代次数
BATCH_SIZE = 32  # 训练批次的大小
LR = 0.0005  # 学习率

modelName = "lyz"
optimizer_type = 'Adam'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.manual_seed(1)

print("加载mnist手写数据集")
train_data = torchvision.datasets.MNIST(
    root='./data/',      
    train=True,    
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    transform=torchvision.transforms.ToTensor(), 
    train=False  
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255 
test_y = test_data.targets[:2000]

train_x = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
train_y = train_data.targets[:2000]

class CNN(nn.Module):
    """3个卷积2个全连接、模仿经典的LeNet-5"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 第一层卷积
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 第二层卷积
        self.conv2_drop = nn.Dropout2d()  # 第三层卷积，并使用丢弃法没随机丢弃权重梯度 抑制过拟合
        self.fc1 = nn.Linear(320, 64)  # 第一层全连接层
        self.fc2 = nn.Linear(64, 10)  # 第二层全连接层

    def forward(self, x):  # 向前传播
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 在第一次卷积后加上激活函数、池化层，然后接上第二层卷积
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # 在第二次卷积后加上激活函数、池化层
        x = x.view(-1, 320)  # 将得到的张量展平，变成一维 类似一维数组
        x = F.relu(self.fc1(x))  # 接入第一个全连接层
        x = F.dropout(x, training=self.training)  # 进行随机丢弃法  （训练模式下）
        x = self.fc2(x)  # 接入第二次全连接层
        return x


cnn = CNN()

"""模型训练部分"""
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

# 损失函数
loss_func = nn.CrossEntropyLoss()

print("开始训练, 大约需要5、6分钟")
count = 0
countList = []
trainLossList = []
testLossList = []
accuracyList = []
accuracyList_train = []

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        output = cnn(b_x)                       # 先将数据放到cnn中计算output
        loss = loss_func(output, b_y)           # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()                   # 梯度归零
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新梯度

        if step % 100 == 0:
            test_output = cnn(test_x)
            test_loss = loss_func(test_output, test_y)    
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

            train_output = cnn(train_x)
            pred_train_y = torch.max(train_output, 1)[1].data.numpy()
            accuracy_train = float((pred_train_y == train_y.data.numpy()).astype(int).sum()) / float(train_y.size(0))

            if step % 500 == 0:
                countList.append(count)
                trainLossList.append(round(float(loss.data.numpy()), 2))
                testLossList.append(round(float(test_loss.data.numpy()), 2))
                accuracyList.append(round(accuracy, 2))
                accuracyList_train.append(round(accuracy_train, 2))


                print('count:', count, '| Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test loss: %.4f' % test_loss.data.numpy(),
                 '| test accuracy: %.2f' % accuracy, '| train accuracy: %.2f' % accuracy_train)
                count += 1
torch.save(cnn.state_dict(), modelName + '.pkl')


# 可视化
fig1 = plt.figure()
plt.title('Loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.plot(countList, trainLossList, label="train_Loss")
plt.plot(countList, testLossList, label="test_Loss")
plt.legend(['train_Loss','test_Loss'])
plt.savefig("result_loss.png")

fig2 = plt.figure()
plt.title('accuracy')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.plot(countList, accuracyList_train, label="train accuracy")
plt.plot(countList, accuracyList, label="test accuracy")
plt.legend(['train accuracy','test accuracy'])
plt.savefig("result_accuracy.png")