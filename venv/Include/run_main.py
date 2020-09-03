import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import arrhythmia_dataset_dev, arrhythmia_dataset_equal, arrhythmia_dataset_dev_val, arrhythmia_dataset_equal_val
from model import Frist_Net
from sklearn.metrics import confusion_matrix, f1_score


import pandas as pd
import numpy as np


# hyper parameter
batch_size = 128
epoch = 50
lr = 0.005
train_or_load = False

if __name__ == "__main__":
    print(20*'--', "Program parameters", 20*'--')
    # print("Model name: VGG_1D")
    print("Batch_size:%d" % batch_size)
    print("Learning rate:%.3f" % (lr))
    print("num of epoch:%d" % (epoch))
    print("train or load:", train_or_load)

    # --------------------         获取数据集       -----------------------
    training_set_dev = arrhythmia_dataset_dev(train=True)
    train_loder_dev = DataLoader(training_set_dev, batch_size, True, drop_last=False)
    training_set_equal = arrhythmia_dataset_equal(train=True)
    train_loder_equal = DataLoader(training_set_equal, batch_size, True, drop_last=False)

    # val_dev_set = arrhythmia_dataset_dev_val()
    # val_dev_dataloder = DataLoader()

    test_set = arrhythmia_dataset_dev(train=False)
    test_loader_dev = DataLoader(dataset=test_set, batch_size=20, shuffle=False, drop_last=False)
    test_set_equal = arrhythmia_dataset_dev(train=False)
    test_loader_equal = DataLoader(dataset=test_set_equal, batch_size=20, shuffle=False)

    # --------------------         建立模型       -----------------------
    model = Frist_Net()
    # model = Secend_Net()
    # model = Net()
    # print(Net)
    # print(training_set_equal.data.shape)
    gpu = torch.cuda.is_available()
    if gpu:
        model.cuda()
        # model.cuda()
    # --------------------      损失函数和优化器     ---------------------
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fir = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------------------      训练     ---------------------
    if train_or_load:
        for epoch in range(epoch):
            # train
            model.train()
            # model_secend.train()
            train_loss, train_acc, train_num = 0.0, 0.0, 0.0
            for i, data in enumerate(zip(train_loder_dev, train_loder_equal)):
                if gpu:
                    data[0][1] = data[0][1].cuda()
                    data[0][0] = data[0][0].float().unsqueeze(1).cuda()
                    data[1][1] = data[1][1].cuda()
                    data[1][0] = data[1][0].float().unsqueeze(1).cuda()
                    # print(Y1)
                    # print(Y1.shape)
                Y_hat = model(data[0][0], data[1][0])
                # print(Y_hat)
                # loss = loss_fn(Y_hat, Y1)
                loss = loss_fn(Y_hat, data[0][1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                train_acc += (Y_hat.argmax(dim=1) == data[0][1]).float().sum().cpu().data.numpy()
                train_num += data[0][1].shape[0]
            train_acc = train_acc/train_num*100
            train_loss = train_loss/train_num*1000

            #  test
            model.eval()
            test_loss, test_acc, test_num = 0.0, 0.0, 0.0
            for i, data in enumerate(zip(test_loader_dev, test_loader_equal)):
                if gpu:
                    x1 = data[0][0].float().unsqueeze(1).cuda()
                    x2 = data[1][0].float().unsqueeze(1).cuda()
                    y1 = data[0][1].cuda()
                    y2 = data[1][1].cuda()
                y_hat = model(x1, x2)
                loss = loss_fn(y_hat, y1)
                test_loss += loss
                test_num += y1.shape[0]
                test_acc += (y_hat.argmax(dim=1) == y1).float().sum().cpu().data.numpy()
            test_acc = test_acc / test_num * 100
            test_loss = test_loss / test_num * 100
            print("epoch: %d, train_loss: %.4f, train_acc:%.2f, "
                  "test_loss: %.4f, test_acc:%.2f"%(epoch+1,train_loss,train_acc,test_loss,test_acc))

            # print("epoch: %d, train_loss: %.4f, train_acc:%.2f, "
            #       %(epoch+1,train_loss,train_acc))
            torch.save(model.state_dict(), 'model.pkl')

    else:
        model.load_state_dict(torch.load('model.pkl', map_location='cpu'))
        model.eval()
        test_equal = pd.read_csv('./数据集/mdiff_test.csv')
        test_equal['label'] = test_equal['label'].map({0: 0, 1: 1})
        test_data_equal = np.array(test_equal.iloc[:, np.arange(56)])
        test_label_equal = np.array(test_equal['label'])
        test_dev = pd.read_csv('./数据集/diff_test.csv')
        test_dev['label'] = test_dev['label'].map({0: 0, 1: 1})
        test_data_dev = np.array(test_dev.iloc[:, np.arange(56)])
        test_label_dev = np.array(test_dev['label'])
        # 预测
        test_data_dev = torch.from_numpy(test_data_dev)
        test_data_equal = torch.from_numpy(test_data_equal)
        if gpu:
            test_data_dev = test_data_dev.float().unsqueeze(1).cuda()
            test_data_equal = test_data_equal.float().unsqueeze(1).cuda()
        # else:
        #     test_data = test_data.float().unsqueeze(1)
        test_label_prob = model(test_data_dev, test_data_equal)
        test_label_pre = test_label_prob.argmax(dim=1).cpu().data.numpy()

        print('Model confusion_matrix:\n', confusion_matrix(test_label_dev, test_label_pre))
        print('Model f1 score(micro and macro): %.4f, %.4f' % (f1_score(test_label_dev, test_label_pre, average="micro"),
                                                                 f1_score(test_label_dev, test_label_pre, average="macro")))










