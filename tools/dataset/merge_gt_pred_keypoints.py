# -*- utf-8 -*-
import os
import json
import argparse
import numpy as np
import mmcv

from tools.dataset.parse_helper import parse_predictions, parse_gt


parser = argparse.ArgumentParser(description='union gt&pred vehicle keypoints: 18 points into one list file.', allow_abbrev=False)
parser.add_argument('-p', '--pred-file', type=str, help='path to prediction json file')
parser.add_argument('-g', '--gt-file', type=str, nargs='+', default=None, help='path to gt annotaion file, coco style')
parser.add_argument('--root-dir', type=str, help='path to json files root dir')
args = parser.parse_args()
 
# args.root_dir = '/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k_crop/images'
# args.pred_file = '/mnt/disk1/data_for_linjiaojiao/projects/pytorch-keypoints/experiments/vehicle/hourglass/results/'\
#                     'test_pose_exp9.3_boxcars-test_with-filter/results.json'
# args.gt_file = ['/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k_crop/annotations/vehicle_keypoints_18points_train_v1.json', 
#                 '/mnt/disk1/data_for_linjiaojiao/datasets/BoxCars21k_crop/annotations/vehicle_keypoints_18points_val_v1.json']



def merge_gt_pred(gt_dict, pred_file, root_dir=None):
    '''
    pred_file format:
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
    -------------
    gt_dict: {
        relative_path: gt_kpt
    }
    > gt_kpt: (np.ndarray[K,3]) The first two dimensions are \
        coordinates, vislibility[v] is the third dimension of the array.
        v=0: not labeled, v=1: labeled but not visible, and v=2: labeled and visible
    '''
    with open(pred_file, 'r') as f:
        results = json.load(f)

    prefix, ext = os.path.splitext(args.pred_file)
    out_file = '{}_merge_gt{}'.format(prefix, ext)
    
    pred_cnt, gt_cnt =0, 0
    for res_ix, res in enumerate(results):
        batch_size = len(res['preds'])
        for b_idx in range(batch_size):
            image_path = res['image_paths'][b_idx]
            if root_dir is not None:
                rel_path = image_path.replace(root_dir, '')
                while rel_path[0] == '/':
                    rel_path = rel_path[1:]
            else:
                rel_path = image_path

            if rel_path in gt_dict.keys():
                # replace with gt keypoints
                gt_cnt += 1
                gt_keypoints = gt_dict[rel_path]
                res['preds'][b_idx] = gt_keypoints
            else:
                pred_cnt += 1

            if res_ix % 100 == 0:
                print('[{}/{}] processed, gt_cnt: {} pred_cnt: {}'.format(res_ix + 1, len(results), gt_cnt, pred_cnt))

    mmcv.dump(results, out_file, indent=4)
    print('{} objects are processed, gt_cnt: {} pred_cnt: {}'.format(len(results), gt_cnt, pred_cnt))
    print('file saved: {}'.format(out_file))


if __name__ == '__main__':
    np.random.seed(4436)

    gt_dict = parse_gt(args.gt_file)
    # pred_dict = parse_predictions(args.pred_file, root_dir=args.root_dir)

    merge_gt_pred(gt_dict, args.pred_file, root_dir=args.root_dir)

    print('Done.')
