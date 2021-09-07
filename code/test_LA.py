import os
import argparse
import shutil
import torch
from networks.vnet import VNet
from networks.vnet_dc import VNet_DC

from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/',
                    help='Name of Experiment')
parser.add_argument('--model', type=str, default='dual_classifier', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--test_paper_model', type=bool, default=False, help='use the model in paper or not')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/" + FLAGS.model + "/"
test_save_path = "../model/prediction/" + FLAGS.model + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNet_DC(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    if FLAGS.test_paper_model:
        save_mode_path = os.path.join("../model/PDC_8976/PDC-Net_8976.pth")
    else:
        save_mode_path = os.path.join(snapshot_path, 'model_1_iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=False, test_save_path=test_save_path) # save result True / False
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(6000)
    print("--------test model: {}--------".format(FLAGS.model))
    print("dice:", metric[0])
    print("jaccard:", metric[1])
    print("asd:", metric[3])
    print("95hd:", metric[2])
