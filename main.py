import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np

from datetime import datetime as dt

import utils
import models

from torch.utils.tensorboard import SummaryWriter


def main():
    
    if torch.cuda.is_available(): # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    
    training_size = 100000
    test_size = 10000
    epochs_num = 10000
    hidden_size = 50
    batch_size = 1000

    train_x, train_t = utils.mkDataSet(training_size)
    test_x, test_t = utils.mkDataSet(test_size)

    model = models.Predictor(1, hidden_size, 1)
    model = model.cuda()
    
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y-%m-%d_%H-%M-%S')

    writer = SummaryWriter(log_dir="./logs/" + tstr)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        batch_len_train = int(training_size / batch_size)
        for i in range(batch_len_train):
            step = epoch*batch_len_train + i
            optimizer.zero_grad()

            data, label = utils.mkRandomBatch(train_x, train_t, batch_size)

            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss
            training_accuracy += np.sum(np.abs((output.data - label.data).detach().cpu().numpy()) < 0.1)

            writer.add_scalar("step/running_loss_train", loss, step)

        #test
        test_accuracy = 0.0
        batch_len_test = int(test_size / batch_size)
        for i in range(batch_len_test):
            step = epoch*batch_len_test + i
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]).cuda(), torch.tensor(test_t[offset:offset+batch_size]).cuda()
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.data - label.data).detach().cpu().numpy()) < 0.1)
        
        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))
        writer.add_scalar("epoch/running_loss", running_loss, epoch + 1)
        writer.add_scalar("epoch/training_accuracy", training_accuracy, epoch + 1)
        writer.add_scalar("epoch/test_accuracy", test_accuracy, epoch + 1)

if __name__ == '__main__':
    main()