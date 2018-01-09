import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lapsrn import Net, L1CharbonnierLoss, HighFrequencyLoss, MixedLoss
from dataset import DatasetFromFolder

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--ckEvery", type=int, default=1., help="save checkpoint every nth iteration, Default: 1")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=1e-5")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpu", type=int, default=0, help="Use nth GPU (for cuda mode)")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.1, type=float, help="Momentum, Default: 0.1")
parser.add_argument("--dataset", default="", type=str, help="path to learning dataset (default: none)")
parser.add_argument("--hfs_loss_weight", type=int, default=.1, help="High frequency loss weight (default: n=0.1)")

def main():
    global opt, model
    opt = parser.parse_args()
    print opt

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.set_device(opt.gpu)
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromFolder(opt.dataset, opt.batchSize)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

    print("===> Building model")
    model = Net()
    criterion = MixedLoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        if epoch % opt.ckEvery == 0:
            save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by momentum every step epochs"""
    lr = opt.lr * (opt.momentum ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print "epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, label_x2, label_x4, label_x8 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()
            label_x8 = label_x8.cuda()

        HR_2x, HR_4x, HR_8x = model(input)

        loss_cb_x2, loss_hfs_x2 = criterion(HR_2x, label_x2)
        loss_cb_x4, loss_hfs_x4 = criterion(HR_4x, label_x4)
        loss_cb_x8, loss_hfs_x8 = criterion(HR_8x, label_x8)
        loss_cb = loss_cb_x2 + loss_cb_x4 + loss_cb_x8
        loss_hfs = opt.hfs_loss_weight * (loss_hfs_x2 + loss_hfs_x4 + loss_hfs_x8)

        loss_x2 = loss_cb_x2 + opt.hfs_loss_weight * loss_hfs_x2
        loss_x4 = loss_cb_x4 + opt.hfs_loss_weight * loss_hfs_x4
        loss_x8 = loss_cb_x8 + opt.hfs_loss_weight * loss_hfs_x8

        optimizer.zero_grad()

        loss_x2.backward(retain_variables=True)
        loss_x4.backward(retain_variables=True)
        loss_x8.backward()

        optimizer.step()

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Charbonnier Loss: {:.3f}, HFS Loss: {:.3f}".format( \
                    epoch, iteration, len(training_data_loader), loss_cb.data[0], loss_hfs.data[0]))

def save_checkpoint(model, epoch):
    model_folder = "model_train/"
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
