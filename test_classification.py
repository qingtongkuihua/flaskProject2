"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def load_txt_point_cloud(filename):
    points_data = np.loadtxt(fname=filename, delimiter=",", dtype=np.float32)

    print('shape: ', points_data.shape)

    # pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])
    points = np.asarray(points_data)
    print(type(points))
    # 过滤速度较小的点

    # new_points = np.delete(filtered_points, [3, 4], axis=1)

    return points


def test(model, filename, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    testpoint = load_txt_point_cloud(filename)
    testpoint = np.expand_dims(testpoint, axis=0)
    testpoint = torch.from_numpy(testpoint)
    testpoint = testpoint.transpose(2, 1)
    print(type(testpoint))
    print(testpoint.shape)
    pred, _ = classifier(testpoint)
    pred_choice = pred.data.max(1)[1]
    print("自己测试的用力")
    print(pred_choice)

    instance_acc = 0
    class_acc = 0
    return pred_choice

    # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
    # for j, (points, target) in tqdm(enumerate(loader), total=10):
    #     if not args.use_cpu:
    #         points, target = points.cuda(), target.cuda()
    #
    #     points = points.transpose(2, 1)
    #     print(type(points))
    #     print(points.shape)
    #     # vote_pool = torch.zeros(target.size()[0], num_class).cuda()
    #     vote_pool = torch.zeros(target.size()[0], num_class)
    #
    #     for _ in range(vote_num):
    #         pred, _ = classifier(points)
    #         vote_pool += pred
    #     pred = vote_pool / vote_num
    #     pred_choice = pred.data.max(1)[1]
    #     print("+++++++++++++++++++++++++++++")
    #     print(pred_choice)
    #     print("+++++++++++++++++++++++++++++")
    #
    #     for cat in np.unique(target.cpu()):
    #         classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
    #         class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
    #         class_acc[cat, 1] += 1
    #     correct = pred_choice.eq(target.long().data).cpu().sum()
    #     mean_correct.append(correct.item() / float(points.size()[0]))
    #
    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # class_acc = np.mean(class_acc[:, 2])
    # instance_acc = np.mean(mean_correct)
    # return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/dataSet/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    print("=============")
    print(test_dataset)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
    print("=======================")
    print(testDataLoader)

    '''MODEL LOADING'''
    num_class = args.num_category
    print(num_class)
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print(model_name)
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        print("zhixinfg")
        classifier = classifier.cuda()

    print("========================")

    device = torch.device('cpu')
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
