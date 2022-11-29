# -*- utf-8 -*-
import os
import shutil
import json
import argparse
import numpy as np
import cv2
import copy

from tools.dataset.parse_helper import parse_predictions, parse_gt


parser = argparse.ArgumentParser(description='Process vehicle keypoints: 18 points.', allow_abbrev=False)
parser.add_argument('-p', '--pred-file', type=str, help='path to prediction json file')
parser.add_argument('-t', '--thresh', type=float, default=0.5, help='predict score thresh')
parser.add_argument('-g', '--gt-file', type=str, nargs='+', default=None, help='path to gt annotaion file, coco style')
parser.add_argument('--root-dir', type=str, help='path to json files root dir')
parser.add_argument('--only-gt', action='store_true', help='only visualize preds with gt')
parser.add_argument('--verbose', action='store_true', help='vis with predict score')
parser.add_argument('-v', '--vis_ratio', type=float, default=0, help='visualization ratio')
parser.add_argument('--save_dir', type=str, default=None, help='path to save visualized images')
args = parser.parse_args()
 
'''
cmd examples:
python vis_pred_results.py -p ~/disk1/projects/pytorch-keypoints/experiments/vehicle/hourglass/results/test_pose_exp9.3_boxcars-test_with-filter/results.json -t 0.7 -v 0.1 --only-gt
python vis_pred_results.py -p ~/disk1/projects/pytorch-keypoints/experiments/vehicle/hourglass/results/test_pose_exp9.3_boxcars-test_with-filter/results.json -t 0.7 -v 0.01
'''
# args.pred_file = '/mnt/disk1/data_for_linjiaojiao/projects/pytorch-keypoints/experiments/vehicle/hourglass/results/test_pose_exp0_boxcars/results.json'

if 'boxcars' in args.pred_file:
    args.root_dir = '/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k_crop/images'
    args.gt_file = ['/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k_crop/annotations/keypoints/vehicle_keypoints_18points_val_v1.json',
                    '/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k_crop/annotations/keypoints/vehicle_keypoints_18points_train_v1.json']
else:
    args.root_dir = '/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/images'
    args.gt_file = ['/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/keypoints/vehicle_keypoints_18points_val_v1.json',
                    '/mnt/disk1/data_for_linjiaojiao/datasets/UA_DETRAC_crop_fps5/annotations/keypoints/vehicle_keypoints_18points_train_v1.json']


'''
results (list[dict]): Testing results containing the following
items:
- preds (np.ndarray[N,K,3]): The first two dimensions are \
    coordinates, score is the third dimension of the array.
- boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
    scale[1],area, score]
- image_paths (list[str]): For example, ['data/coco/val2017\
    /000000393226.jpg']
- heatmap (np.ndarray[N, K, H, W]): model output heatmap
- bbox_id (list(int)).

example:
[{
    "preds": [[[94.25, 54.25001525878906, 0.8612086176872253], ...]], 
    "boxes": [[128.0, 128.0, 1.5999999046325684, 1.5999999046325684, 102399.984375, 1.0]],
    "image_paths": ["/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k/images/uvoz1/skoda/fabia/combi/0/0_0.png"], 
    "bbox_ids": [0], 
    "output_heatmap": null
}]

'''


def visualize_predictions(pred_dict, gt_dict, root_dir=None):

    vis_cnt = 0
    res_ix = 0
    for rel_path, pred_kpt in pred_dict.items():

        res_ix += 1

        if root_dir is not None:
            image_path = os.path.join(root_dir, rel_path)
        else:
            image_path = rel_path

        gt_kpt = gt_dict.get(rel_path, None)
        if args.only_gt and gt_kpt is None:
            continue
        
        if np.random.rand() <= args.vis_ratio:
            # visualize
            img = cv2.imread(image_path)
            ori_h, ori_w, _ = img.shape
            vis_scale_h = 400 / ori_h
            vis_scale_w = 400 / ori_w
            img = cv2.resize(img, (400, 400))

            if args.verbose:
                verbose_img = copy.deepcopy(img)
            valid_pred_kpt_cnt = 0
            for kp_idx in range(len(pred_kpt)):
                '''
                pred_kpt: (np.ndarray[K,3]): The first two dimensions are coordinates, score is the third dimension of the array.
                '''
                # plot pred keypoints
                point = pred_kpt[kp_idx]
                label = kp_idx + 1
                point_x = int(float(point[0]) * vis_scale_w)
                point_y = int(float(point[1]) * vis_scale_h)
                point_score = float(point[2])

                if args.verbose and point_score >= min(0.1, args.thresh):
                    # vis with all predicted points (thresh > 0.1)
                    if point_score >= args.thresh:
                        vis_str = '{}'.format(label)
                        color = (0, 255, 0)
                    else: 
                        vis_str = '{}: {:.01f}'.format(label, point_score)
                        color = (14, 173, 238)
                    cv2.circle(verbose_img, (point_x + 2, point_y + 2), 3, color, -1, cv2.LINE_AA)
                    cv2.putText(verbose_img, vis_str, (point_x + 2, point_y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

                # if kp_idx == 0:
                #     cv2.putText(img, 'green: pred', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                if point_score >= args.thresh:
                    cv2.circle(img, (point_x, point_y), 5, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.putText(img, str(label), (point_x + 2, point_y + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    
                    valid_pred_kpt_cnt += 1

                # plot gt keypoints
                if gt_kpt is not None:
                    '''
                    (Pdb) gt_kpt.shape
                    (18, 3)
                    '''
                    gt_point = gt_kpt[kp_idx]
                    gt_point_x = int(float(gt_point[0]) * vis_scale_w)
                    gt_point_y = int(float(gt_point[1]) * vis_scale_h)
                    cv2.circle(img, (gt_point_x, gt_point_y), 3, (0, 0, 255), -1, cv2.LINE_AA)
                    # if kp_idx == 0:
                    #     cv2.putText(img, 'red: gt', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(img, '{}'.format(label), (gt_point_x - 2, gt_point_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            save_to = os.path.join(args.save_dir, '{}points_'.format(valid_pred_kpt_cnt) + image_path.replace('/', '_'))
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            cv2.imwrite(save_to, img)
            print('image saved: {}'.format(save_to))

            if args.verbose:
                # vis with more information
                save_to = os.path.join(args.save_dir, '{}points_'.format(valid_pred_kpt_cnt) + image_path.replace('/', '_') + '_verbose.jpg')
                cv2.imwrite(save_to, verbose_img)

            vis_cnt += 1
        
        if res_ix % 1000 == 0:
            print('[{}/{}] processed, vis cnt: {}'.format(res_ix, len(pred_dict.keys()), vis_cnt))
        
    print('{} results been processed, vis cnt: {}'.format(len(pred_dict.keys()), vis_cnt))


if __name__ == '__main__':
    np.random.seed(4436)

    gt_dict = parse_gt(args.gt_file)
    pred_dict = parse_predictions(args.pred_file, root_dir=args.root_dir)

    if args.save_dir is None:
        if args.verbose:
            args.save_dir = os.path.join(os.path.dirname(args.pred_file), 'pred_keypoints_vis_thresh{}_verbose'.format(args.thresh))
        else:
            args.save_dir = os.path.join(os.path.dirname(args.pred_file), 'pred_keypoints_vis_thresh{}'.format(args.thresh))
        if args.only_gt:
            args.save_dir = args.save_dir + '_with-gt'

    if args.vis_ratio > 0 and os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    visualize_predictions(pred_dict, gt_dict, root_dir=args.root_dir)

    print('Done.')
