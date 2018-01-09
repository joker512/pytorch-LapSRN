import argparse, os
import torch
import random
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lapsrn_wgan import Net, L1CharbonnierLoss, HighFrequencyLoss, MixedLoss
from dataset import DatasetFromFolder
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--ckEvery", type=float, default=1., help="save checkpoint every nth iteration, Default: 1")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=1e-5")
parser.add_argument("--step", type=int, default=5, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=5")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpu", type=int, default=0, help="Use nth GPU (for cuda mode)")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.1, type=float, help="Momentum, Default: 0.1")
parser.add_argument("--dataset", default="", type=str, help="path to learning dataset (default: none)")
parser.add_argument("--hfs_loss_weight", type=int, default=.1, help="High frequency loss weight (default: n=0.1)")
parser.add_argument("--gen_loss_weight", type=int, default=10., help="Generator loss weight (default: n=10)")
parser.add_argument("--train_disc_iter", type=int, default=2, help="Train discriminator each nth iteration (default: n=2)")

writer = SummaryWriter()
WRITE_DATA_ITER = 100

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
        if os.path.isfile(opt.resume + "_gen.pth"):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint_gen = torch.load(opt.resume + "_gen.pth")
            model.netg.load_state_dict(checkpoint_gen["model"].state_dict())
            if os.path.isfile(opt.resume + "_disc.pth"):
                checkpoint_disc = torch.load(opt.resume + "_disc.pth")
                model.netd.load_state_dict(checkpoint_disc["model"].state_dict())
            if not opt.start_epoch:
                opt.start_epoch = checkpoint_gen["epoch"] + 1
            print("=> start epoch {}".format(opt.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        if epoch % math.ceil(opt.ckEvery) == 0:
            save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by momentum every step epochs"""
    lr = opt.lr * (opt.momentum ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print "epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, label_x2, label_x4, label_x8 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)

        if iteration % WRITE_DATA_ITER == 0:
            input_grid = vutils.make_grid(input.data[0], normalize=True)
            label_x2_grid = vutils.make_grid(label_x2.data[0], normalize=True)
            label_x4_grid = vutils.make_grid(label_x4.data[0], normalize=True)
            label_x8_grid = vutils.make_grid(label_x8.data[0], normalize=True)

            writer.add_image('input', input_grid, iteration)
            writer.add_image('2x label', label_x2_grid, iteration)
            writer.add_image('4x label', label_x4_grid, iteration)
            writer.add_image('8x label', label_x8_grid, iteration)

        if opt.cuda:
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()
            label_x8 = label_x8.cuda()

        disc_fake, HR_2x, HR_4x, HR_8x = model(input)
        disc_real = model.netd(label_x8)

        loss_cb_x2, loss_hfs_x2 = criterion(HR_2x, label_x2)
        loss_cb_x4, loss_hfs_x4 = criterion(HR_4x, label_x4)
        loss_cb_x8, loss_hfs_x8 = criterion(HR_8x, label_x8)
        loss_cb = loss_cb_x2 + loss_cb_x4 + loss_cb_x8
        loss_hfs = opt.hfs_loss_weight * (loss_hfs_x2 + loss_hfs_x4 + loss_hfs_x8)

        loss_x2 = loss_cb_x2 + opt.hfs_loss_weight * loss_hfs_x2
        loss_x4 = loss_cb_x4 + opt.hfs_loss_weight * loss_hfs_x4
        loss_x8 = loss_cb_x8 + opt.hfs_loss_weight * loss_hfs_x8

        loss_disc = ((disc_real - 1.) ** 2 + (disc_fake - 0.) ** 2) / 2.
        loss_gen = opt.gen_loss_weight * ((disc_fake - 1.) ** 2)

        optimizer.zero_grad()
        loss_x2.backward(retain_graph=True)
        loss_x4.backward(retain_graph=True)
        loss_x8.backward(retain_graph=True)

        learn_disc = iteration % opt.train_disc_iter == 0
        for p in model.netd.parameters():
            p.requires_grad = False
        loss_gen.backward(retain_graph=learn_disc)
        for p in model.netd.parameters():
            p.requires_grad = True

        if learn_disc:
            for p in model.netg.parameters():
                p.requires_grad = False
            loss_disc.backward()
            for p in model.netg.parameters():
                p.requires_grad = True

        optimizer.step()

        if iteration % WRITE_DATA_ITER == 0:
            writer.add_scalar('Charbonnier loss', loss_cb.data[0], iteration)
            writer.add_scalar('HighFrequency loss', loss_hfs.data[0], iteration)
            writer.add_scalar('Discriminator loss', loss_disc.data[0], iteration)
            writer.add_scalar('Generator loss', loss_gen.data[0], iteration)

            HR_8x = HR_8x.cpu()
            HR_8x = vutils.make_grid(HR_8x.data[0], normalize=True)
            HR_4x = HR_4x.cpu()
            HR_4x = vutils.make_grid(HR_4x.data[0], normalize=True)
            HR_2x = HR_2x.cpu()
            HR_2x = vutils.make_grid(HR_2x.data[0], normalize=True)

            writer.add_image('8x result', HR_8x, iteration)
            writer.add_image('4x result', HR_4x, iteration)
            writer.add_image('2x result', HR_2x, iteration)

            print("===> Epoch[{}]({}/{}): CharbLoss (2x+4x+8x): {:.3f}, HighFreqLoss (2x+4x+8x): {:.3f}, DiscLoss: {:.3f}, GenLoss: {:.3f}".format( \
                    epoch, iteration, len(training_data_loader), loss_cb.data[0], loss_hfs.data[0], \
                    loss_disc.data[0], loss_gen.data[0]))

            if iteration == int(len(training_data_loader) * opt.ckEvery) and not opt.ckEvery.is_integer() :
                save_checkpoint(model, epoch, iteration)

def save_checkpoint(model, epoch, iteration=0):
    model_folder = "model_train/"
    if iteration == 0:
        model_out_path = model_folder + "epoch_{}".format(epoch)
    else:
        model_out_path = model_folder + "epoch_{}_iter_{}".format(epoch, iteration)
    state_gen = {"epoch": epoch ,"model": model.netg}
    state_disc = {"epoch": epoch ,"model": model.netd}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state_gen, model_out_path + "_gen.pth")
    torch.save(state_disc, model_out_path + "_disc.pth")
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
