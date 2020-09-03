import torch
import torch.nn as nn


class Frist_Net(nn.Module):
    def __init__(self):
        super(Frist_Net, self).__init__()
        # 定义CNN卷积层网络
        self.cnn_unit = nn.Sequential(
            # layer1  神经元个数52*5  核大小为5  步长为1  填充为0
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # layer2  神经元个数26*5  核大小为5  步长为1  填充为0
            nn.AvgPool1d(kernel_size=2, stride=2),
            # layer3  神经元个数22*5  核大小为5  步长为1  填充为0
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # layer4  神经元个数11*5  核大小为5  步长为1  填充为0
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        # 部分级一维卷积神经网络
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2, 2),
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(185, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
            # nn.ReLU(),
            nn.Softmax(1)
        )

    # 前向运算 ----- layer5  神经元个数55

    def forward(self, inputs1, inputs2):
        x1 = inputs1.size(0)
        x2 = inputs2.size(0)
        out1 = self.cnn_unit(inputs1)
        out1 = out1.view(x1, 55)
        out2 = self.model(inputs2)
        out2 = out2.view(x2, 130)

        outputs = torch.cat([out1, out2], dim=1)# 将对象级卷积[ , 55]和部分级卷积[ , 130]输出维度拼接成[ , 185]
        # outputs = outputs.view(-1, 185)
        # print(outputs.shape)# 打印Flatten层的维度

        out = self.fc(outputs)
        return out

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.layer1 = nn.Sequential(
#             # layer1  神经元个数52*5  核大小为5  步长为1  填充为0
#             nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             # layer2  神经元个数26*5  核大小为5  步长为1  填充为0
#             nn.AvgPool1d(kernel_size=2, stride=2),
#             # layer3  神经元个数22*5  核大小为5  步长为1  填充为0
#             nn.Conv1d(in_channels=5, out_channels=5, kernel_size=5, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             # layer4  神经元个数11*5  核大小为5  步长为1  填充为0
#             nn.AvgPool1d(kernel_size=2, stride=2),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(1, 5, 5, 1),
#             nn.ReLU(),
#             nn.AvgPool1d(2, 2),
#         )
#         self.fc = nn.Sequential(nn.Linear(135, 20),
#                                 nn.ReLU(),
#                                 nn.Softmax(4)
#         )
#
#     def forward(self, x1, x2):
#         y1 = self.layer1(x1)
#         y2 = self.layer2(x2)
#         y1 = torch.flatten(y1)
#         y2 = torch.flatten(y2)
#         y = torch.cat((y1, y2))
#         output = self.fc(y)
#         return output







