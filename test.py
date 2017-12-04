import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from dataset import convert_to_numpy
import os

parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--gpu", type=int, default=0, help="Use nth GPU (for cuda mode)")
parser.add_argument("--model", default="model/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--datapath", default="Set", type=str, help="path to the folder with images")
parser.add_argument("--resultpath", default="Result", type=str, help="path to the folder with test results")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border, :]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def load_img(name):
    gt_image = Image.open(name)
    gt_image = gt_image.convert('RGB')
    gt_image = gt_image.crop((0, 0) + tuple(ti - ti % opt.scale for ti in gt_image.size))
    low_image = gt_image.resize(tuple(ti // opt.scale for ti in gt_image.size), Image.BICUBIC)
    baseline_image = low_image.resize(gt_image.size, Image.BICUBIC)
    return convert_to_numpy(gt_image, False), convert_to_numpy(low_image, True), convert_to_numpy(baseline_image, False)


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if cuda:
    torch.cuda.set_device(opt.gpu)

model = torch.load(opt.model)["model"]

if cuda:
    model = model.cuda()
else:
    model = model.cpu()

for image_name in os.listdir(opt.datapath):
    gt_image, input_img_th, baseline_img = load_img(os.path.join(opt.datapath, image_name))

    psnr_bicubic = PSNR(baseline_img, gt_image, shave_border=opt.scale)
    input_img_th = Variable(torch.from_numpy(input_img_th).float()).view(1, input_img_th.shape[0], input_img_th.shape[1], input_img_th.shape[2])

    if cuda:
        input_img_th = input_img_th.cuda()

    start_time = time.time()
    HR_2x, HR_4x = model(input_img_th)
    elapsed_time = time.time() - start_time

    HR_2x = HR_2x.cpu()
    HR_4x = HR_4x.cpu()

    output_img_th = HR_2x.data[0].numpy().astype(np.float32)
    output_img_th[output_img_th<0] = 0
    output_img_th[output_img_th>255.] = 255.

    output_img = np.transpose(output_img_th, (1, 2, 0))
    psnr_predicted = PSNR(output_img, gt_image, shave_border=opt.scale)
    prefix = os.path.join(opt.resultpath, os.path.splitext(image_name)[0])

    print("Image_name", image_name);
    print("Scale",opt.scale)
    print("PSNR_predicted", psnr_predicted)
    print("PSNR_bicubic", psnr_bicubic)
    print("It takes {}s for processing".format(elapsed_time))
    print

    fig = plt.figure()
    ax = plt.subplot("131")
    gt_pil = Image.fromarray(gt_image.astype(np.uint8), 'RGB')
    gt_pil.save(prefix + '_gt.png')
    ax.imshow(gt_pil)
    ax.set_title("GT")

    ax = plt.subplot("132")
    bl_pil = Image.fromarray(baseline_img.astype(np.uint8), 'RGB')
    bl_pil.save(prefix + '_bl.png')
    ax.imshow(bl_pil)
    ax.set_title("Input(Bicubic)")

    ax = plt.subplot("133")
    out_pil = Image.fromarray(output_img.astype(np.uint8), 'RGB')
    out_pil.save(prefix + '_out.png')
    ax.imshow(out_pil)
    ax.set_title("Output(LapSRN)")

    fig.savefig(prefix + '.png', dpi=200)
