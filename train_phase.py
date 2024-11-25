import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from models import mvsnet
import numpy as np
import time

from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
import wandb
from tensorboardX import SummaryWriter

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--bayesian_mode', action='store_true', help='bayesian head')
parser.add_argument('--prob_vol_mode', action='store_true', help='prob vol guide')

parser.add_argument('--dataset', default='dtu_yao_phase', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=32, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="20,24,28:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=384, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--kl_weight', type=float, default=1, help='kl loss weight')
parser.add_argument('--nll_weight', type=float, default=1, help='nll loss weight')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=200, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')

parser.add_argument('--wandb', action='store_true', help='refine output')
parser.add_argument('--project', default='mvstpu', help='wandb project')
parser.add_argument('--entity', default='juneberm',  help='wandb entity')
parser.add_argument('--experiment_name', type=str, help='test name')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth)
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 3, args.numdepth)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False)

# model, optimizer
model = MVSNet(depth_dim=args.numdepth, bayesian_mode=args.bayesian_mode)
if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()

if args.bayesian_mode:
    model_loss = NLLKLLoss
elif args.prob_vol_mode:
    model_loss = mvsnet_loss_KL
else:
    model_loss = mvsnet_loss

consist_loss = None
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckptb 
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            scalar_outputs, image_outputs = train_sample(sample, kl_weight= args.kl_weight, nll_weight=args.nll_weight, detailed_summary=False)

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
                logging = {
                    'Tr Loss': round(scalar_outputs['loss'], 4),
                    'Tr abs Loss': round(scalar_outputs["abs_uph_error"], 4),
                    'Tr rmse Loss': round(scalar_outputs["rmse_uph_error"], 4),
                }

                if args.bayesian_mode:
                    logging['Tr NLL loss'] = round(scalar_outputs['nll_loss'], 4)
                
                if args.prob_vol_mode:
                    logging['Tr KL loss'] = round(scalar_outputs['kl_loss'], 4)

                if args.wandb:
                    wandb.log(logging)

            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), scalar_outputs["loss"],
                                                                                     time.time() - start_time))
            del scalar_outputs, image_outputs

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>2}.ckpt".format(args.logdir, epoch_idx))

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            scalar_outputs, image_outputs = test_sample(sample, kl_weight= args.kl_weight, nll_weight=args.nll_weight, detailed_summary=False)

            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)

                logging = {
                    'Val Loss': round(scalar_outputs['loss'], 4),
                    'Val abs Loss': round(scalar_outputs["abs_uph_error"], 4),
                    'Val rmse Loss': round(scalar_outputs["rmse_uph_error"], 4),
                }

                if args.bayesian_mode:
                    logging['Val NLL loss'] = round(scalar_outputs['nll_loss'], 4)

                if args.prob_vol_mode:
                    logging['Val KL loss'] = round(scalar_outputs['kl_loss'], 4)

                if args.wandb:
                    wandb.log(logging)

            avg_test_scalars.update(scalar_outputs)
            print('Epoch {}/{}, Iter {}/{}, test abs loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), round(scalar_outputs["abs_uph_error"], 4),
                                                                                     time.time() - start_time))
            del scalar_outputs, image_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        scalar_outputs, image_outputs = test_sample(sample, detailed_summary=False)
        nll_loss, kl_loss = scalar_outputs["nll_loss"], scalar_outputs["kl_loss"]
        loss = nll_loss + kl_loss
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, kl_weight, nll_weight, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_value = sample_cuda["depth_values"]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    prob_volume = outputs["prob_volume"]

    loss, l1_loss, nll_loss, kl_loss = 0, 0, 0, 0
    if args.bayesian_mode:
        sigma = outputs["sigma"]
        nll_loss, kl_loss = model_loss(depth_est, sigma, depth_gt, mask, prob_volume, depth_value) ## NLLKLLoss
        loss = nll_loss * nll_weight + kl_loss * kl_weight
    elif args.prob_vol_mode:
        l1_loss, kl_loss = model_loss(depth_est, depth_gt, mask, prob_volume, depth_value)
        loss = l1_loss + kl_loss * kl_weight
    else:
        loss = model_loss(depth_est, depth_gt, mask)

    loss.backward()
    optimizer.step()

    #### scaled disp
    # depth_est, depth_gt = 1./ depth_est, 1./ depth_gt

    #### disp
    # depth_min, depth_max = 1. / depth_value[:, -1], 1. / depth_value[:, 0]
    # depth_est, depth_gt = disp_to_depth(depth_est, depth_min, depth_max)[1], disp_to_depth(depth_gt, depth_min, depth_max)[1]

    scalar_outputs = {'loss': loss}
    if args.bayesian_mode:
        scalar_outputs["nll_loss"] = nll_loss
        disp_min, disp_max = depth_value[:, 0], depth_value[:, -1]
        depth_est = depth_est * (disp_max - disp_min) + disp_min
    elif args.prob_vol_mode:
        scalar_outputs["l1_loss"] = l1_loss
        scalar_outputs["kl_loss"] = kl_loss

    image_outputs = {"depth_est": depth_est * mask, "depth_gt": depth_gt * mask,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}

    image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
    scalar_outputs["abs_uph_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["rmse_uph_error"] = RMSEDepthError_metrics(depth_est, depth_gt, mask > 0.5)

    return tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, kl_weight, nll_weight, detailed_summary=False):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_value = sample_cuda["depth_values"]

    with torch.no_grad():
        outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        depth_est = outputs["depth"]
        prob_volume = outputs["prob_volume"]

        loss, nll_loss, kl_loss = 0, 0, 0
        if args.bayesian_mode:
            sigma = outputs["sigma"]
            nll_loss, kl_loss = model_loss(depth_est, sigma, depth_gt, mask, prob_volume, depth_value) ## NLLKLLoss
            loss = nll_loss * nll_weight + kl_loss * kl_weight
        elif args.prob_vol_mode:
            l1_loss, kl_loss = model_loss(depth_est, depth_gt, mask, prob_volume, depth_value)
            loss = l1_loss + kl_loss * kl_weight
        else:
            loss = model_loss(depth_est, depth_gt, mask)

        scalar_outputs = {'loss': loss}
        if args.bayesian_mode:
            scalar_outputs["nll_loss"] = nll_loss
        elif args.prob_vol_mode:
            scalar_outputs["l1_loss"] = l1_loss
            scalar_outputs["kl_loss"] = kl_loss

        image_outputs = {"depth_est": depth_est * mask, "depth_gt": depth_gt * mask,
                        "ref_img": sample["imgs"][:, 0],
                        "mask": sample["mask"]}
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

        scalar_outputs["abs_uph_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["rmse_uph_error"] = RMSEDepthError_metrics(depth_est, depth_gt, mask > 0.5)

    return tensor2float(scalar_outputs), image_outputs


def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=False)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    if args.mode == "train":

        # CFG = {
        #     "epochs" : args.epochs,
        #     "batch_size" : args.batch_size,
        #     "optimizer" : args.optimizer,
        #     "scheduler" : args.scheduler,
        #     "criterion" : args.criterions,
        # }
        if args.wandb:
            wandb.init(
                project=args.project, entity=args.entity, name=args.experiment_name,
            )
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
