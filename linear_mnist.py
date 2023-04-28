import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from squib.functions.evaluation import accuracy
from squib.updaters.updater import StanderdUpdater
from squib.trainer.trainer import Trainer



def MLP(output_channel:int):
    class Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)
            
#nn.ReLU(inplace=True)
#nn.Sigmoid()
#nn.Tanh()
#1,1,0 (output_channel*28*28)
    kernel_size_num = 3
    #基本ステータスは2
    exp_stride_num = 1
    #基本ステータスは1
    padding_num = 1
    mlp = nn.Sequential( 
        Flatten(),                                                # (-1, 128*3*3)
        nn.Linear(784, 100),#全結合
        nn.ReLU(inplace=True),
        nn.Linear(100, 10)
    )

    return mlp


def ClassificationUpdater(model, optimizer=None, tag=None) -> StanderdUpdater:
    cel = nn.CrossEntropyLoss()

    def _loss_func(x, t):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        y = model(x)
        loss = cel(y, t)
        accu = accuracy(y, t)
        result = {
            'loss': loss.item(),
            'accuracy': accu
        }
        return loss, result

    upd = StanderdUpdater(loss_func=_loss_func,
                          optimizer=optimizer,
                          tag=tag)

    return upd


def main():
    BATCH_SIZE = 1000
    trainset = MNIST(root='./mnist', train=True,  download=False,
                          transform=ToTensor())
    validationset = MNIST(root='./mnist', train=False, download=False,
                          transform=ToTensor())

    train_loader = DataLoader(trainset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=2)
    validation_loader = DataLoader(validationset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=2)

    #ここを変更する
    model = MLP(output_channel=32)
    # opt = optim.Adadelta(model.parameters(), rho=0.9, eps=1e-06,lr=1.0)
    opt = optim.SGD(model.parameters(), lr=1e-3)
    device = torch.device('cuda:0')
    model.to(device)

    train_updater = ClassificationUpdater(model, tag='tr', optimizer=opt)
    validation_updater = ClassificationUpdater(model, tag='vl')

    #ここを変更する
    trainer = Trainer(loader=train_loader,
                      updater=train_updater,
                      device=device,
                      save_to='./result_nn')
    trainer.log_report(keys=['tr/loss', 'vl/loss', 'tr/accuracy', 'vl/accuracy'],
                       plots={
                           'loss.png': ['tr/loss', 'vl/loss'],
                           'accuracy.png': ['tr/accuracy', 'vl/accuracy'],
    },
        trigger=(1, 'epoch'))

    trainer.add_evaluation(loader=validation_loader,
                           updater=validation_updater,
                           trigger=(1, 'epoch'))

    trainer.save_model(path='models/models_{epoch}.pth',
                       model=model,
                       trigger=(1, 'epoch'))
    trainer.save_trainer(path='trainer.pth',
                         models={'model': model, 'opt': opt},
                         trigger=(1, 'epoch'))

    trainer.run((100, 'epoch'))


if __name__ == "__main__":
   # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
