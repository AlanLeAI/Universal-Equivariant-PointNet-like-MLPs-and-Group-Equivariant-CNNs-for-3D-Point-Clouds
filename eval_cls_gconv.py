import argparse
import os
import sys
import numpy as np 
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from model.group_pointconv import GroupPointConvDensityClsSsg as GroupPointConvDensityClsSsg
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
import provider
from pathlib import Path
from tqdm import tqdm
import math
from random import randint
import random


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('GPointConv')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--gpu', type=str, default='cpu', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--model_name', default='gpointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device =  args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./roration_eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints_rotation/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = 'data/modelnet40_normal_resampled'


    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    logger.info("The number of test data is: %d", len(TEST_DATASET))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    num_class = 40
    classifier = GroupPointConvDensityClsSsg(num_classes=num_class).to(device)

    if args.checkpoint is not None:
        print('Load CheckPoint from {}'.format(args.checkpoint))
        logger.info('Load CheckPoint')
        # Load
        checkpoint = torch.load(args.checkpoint)
        classifier.load_state_dict(checkpoint['model_state_dict'])
#         classifier.load_state_dict(torch.load(args.checkpoint).module.state_dict())
        count_parameters(classifier)

    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)

    '''EVAL'''

    logger.info('Start evaluating...')
    print('Start evaluating...')

    accuracy = test_original(classifier, testDataLoader)
    print('Test Accuracy: %f' % accuracy)
    logger.info('Test original:'+str(accuracy))

    acc_Rz = test_rotation_oz(classifier, testDataLoader)
    print('Random Oz Accuracy: %f' % acc_Rz)
    logger.info('Random Oz Accuracy:'+str(acc_Rz))

    acc_Rx = test_rotation_ox(classifier, testDataLoader)
    print('Random Ox Accuracy: %f' % acc_Rx)
    logger.info('Random Ox Accuracy:'+str(acc_Rx))

    acc_Ry = test_rotation_oy(classifier, testDataLoader)
    print('Random Oy Accuracy: %f' % acc_Ry)
    logger.info('Random Oy Accuracy:'+str(acc_Ry))
    # test_rotation_group(classifier,testDataLoader,split_group=12)
    print('Average Rx_Ry_Rz:',(acc_Rz+acc_Rx+acc_Ry)/3)

    acc_so3 = test_rotation_so3(classifier, testDataLoader)
    logger.info("average acc with so3 rotation:" + str(acc_so3))
    print("average acc with so3 rotation:" ,acc_so3)

    logger.info('End of evaluation...')


#--------------------------------------------------------#
def test_original(model, loader):
    device = torch.device('cpu')
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def test_rotation_group(model, loader, split_group = None ,name=None):
    device = torch.device('cpu')

    G24 = torch.from_numpy(np.array([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],

        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [1, 0, 0], [0, -1, 0]],

        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],

        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],

        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

        [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    ])).float()

    if split_group != None:
        r_group = G24[split_group]

    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points[:,:,:3] = torch.matmul(points[:,:,:3], r_group) #rotate-sample
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    print("test_rotation_group:", accuracy)
    return accuracy

def test_random_angel(model, loader, coordinates = "Rx" , phase ="custom"):
    device = torch.device('cpu')
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]

        random.seed(j)

        if phase == "custom":
            r = random.Random(j)
            alpha = r.choice([randint(0, 30),randint(60, 120),randint(150, 180)])
            rotation_angle = alpha*np.pi / 180.

        elif phase == "random":
            alpha = randint(0, 180)
            rotation_angle = alpha*np.pi / 180.

        points[:,:,:3] = rotate_point_cloud_by_angle(points[:,:,:3], coordinates, rotation_angle) #rotate-sample

        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    print("random angel acc:", accuracy)
    return accuracy



def test_rotation_oz(model, loader):
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.data.numpy()
        points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])

        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def test_rotation_ox(model, loader):
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.data.numpy()
        points[:, :, :3] = provider.rotate_point_cloud_x(points[:, :, :3])

        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def test_rotation_oy(model, loader):
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.data.numpy()
        points[:, :, :3] = provider.rotate_point_cloud_y(points[:, :, :3])

        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

if __name__ == '__main__':
    args = parse_args()
    main(args)
