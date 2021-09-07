# train LA dual classifier(PDC-Net)
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_dc import VNet_DC
from dataloaders import utils
from utils import ramps, losses
from utils import losses_param_discrepancy
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='dual_classifier', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')

parser.add_argument('--param_weight', type=float, default=0.1, help='param_weight')
parser.add_argument('--param_rampup', type=float, default=40.0, help='param_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_param_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.param_weight * ramps.sigmoid_rampup(epoch, args.param_rampup)


if __name__ == "__main__":
    ## make logger file
    if os.path.exists(snapshot_path):
        shutil.rmtree(snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = VNet_DC(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model_1 = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer_encoder = optim.SGD([{"params": model_1.block_one.parameters()},
                                   {"params": model_1.block_one_dw.parameters()},
                                   {"params": model_1.block_two.parameters()},
                                   {"params": model_1.block_two_dw.parameters()},
                                   {"params": model_1.block_three.parameters()},
                                   {"params": model_1.block_three_dw.parameters()},
                                   {"params": model_1.block_four.parameters()},
                                   {"params": model_1.block_four_dw.parameters()},
                                   {"params": model_1.block_five.parameters()},
                                   {"params": model_1.block_five_up.parameters()},
                                   {"params": model_1.block_six.parameters()},
                                   {"params": model_1.block_six_up.parameters()},
                                   {"params": model_1.block_seven.parameters()},
                                   {"params": model_1.block_seven_up.parameters()},
                                   {"params": model_1.block_eight.parameters()},
                                   {"params": model_1.block_eight_up.parameters()},
                                   {"params": model_1.block_nine.parameters()}],
                                  lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_classifier = optim.SGD([{"params": model_1.out_conv1.parameters()},
                                      {"params": model_1.out_conv2.parameters()}],
                                     lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    model_1.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            ## calculate the loss
            outputs_1, outputs_2 = model_1(volume_batch)
            # task loss -------------------------------------------------------------------------------------
            loss_seg_1 = F.cross_entropy(outputs_1[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft_1 = F.softmax(outputs_1, dim=1)
            loss_seg_dice_1 = losses.dice_loss(outputs_soft_1[:labeled_bs, 1, :, :, :],
                                               label_batch[:labeled_bs] == 1)

            loss_seg_2 = F.cross_entropy(outputs_2[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft_2 = F.softmax(outputs_2, dim=1)
            loss_seg_dice_2 = losses.dice_loss(outputs_soft_2[:labeled_bs, 1, :, :, :],
                                               label_batch[:labeled_bs] == 1)

            loss = (0.5 * (loss_seg_1 + loss_seg_dice_1) + 0.5 * (loss_seg_2 + loss_seg_dice_2))

            # param loss ------------------------------------------------------------------------------------
            param_dist = losses_param_discrepancy.cosdistance_loss_in(model_1.out_conv1, model_1.out_conv2)
            param_weight = get_current_param_weight(iter_num // 150)
            param_loss = param_weight * param_dist

            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            (loss + param_loss).backward()
            optimizer_encoder.step()
            optimizer_classifier.step()

            # consistency loss ------------------------------------------------------------------------------
            outputs_1, outputs_2 = model_1(volume_batch)
            consistency_dist = consistency_criterion(outputs_1, outputs_2)
            consistency_dist = torch.mean(consistency_dist)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss = consistency_weight * consistency_dist

            optimizer_encoder.zero_grad()
            consistency_loss.backward()
            optimizer_encoder.step()

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg_1', loss_seg_1, iter_num)
            writer.add_scalar('loss/loss_seg_dice_1', loss_seg_dice_1, iter_num)
            writer.add_scalar('loss/loss_seg_2', loss_seg_2, iter_num)
            writer.add_scalar('loss/loss_seg_dice_2', loss_seg_dice_2, iter_num)

            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)

            writer.add_scalar('train/param_weight', param_weight, iter_num)
            writer.add_scalar('train/param_dist', param_dist, iter_num)
            writer.add_scalar('train/param_loss', param_loss, iter_num)

            logging.info('iteration %d : loss: %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(outputs_soft_1[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_1', grid_image, iter_num)

                image = torch.max(outputs_soft_2[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_2', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                image = torch.max(outputs_soft_1[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_1', grid_image, iter_num)

                image = torch.max(outputs_soft_2[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_2', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer_encoder.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_classifier.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 1000 == 0:
                save_mode_path_1 = os.path.join(snapshot_path, 'model_1_iter_' + str(iter_num) + '.pth')
                torch.save(model_1.state_dict(), save_mode_path_1)
                logging.info("save model_1 to {}".format(save_mode_path_1))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    save_mode_path_1 = os.path.join(snapshot_path, 'model_1_iter_' + str(max_iterations) + '.pth')
    torch.save(model_1.state_dict(), save_mode_path_1)
    logging.info("save model to {}".format(save_mode_path_1))

    writer.close()
