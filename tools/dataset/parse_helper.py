# -*- utf-8 -*-
import json
import numpy as np
import os


def read_json(json_path):
    with open(json_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    
    return json_data


def parse_gt(gt_files):
    '''
    
    coco object keypoints format:
    {
        "info": {
                    "description":"Vehicle keypoints dataset, 18 points",
                    "version":"1.0","year":2022,
                    "contributor":"Lin, Jiaojiao",
                    "date_created":"2022-11-10"
                },
        "licenses": [],
        "images": [image],
        "annotations": [annotation],
        "categories": [category]
    }
    
    > More specifically:
    images: [
        {
            "license": -1,
            "file_name": "fit0/audi/a4/mk1/0/0_0.png",
            "height": 132,                  # image height
            "width": 175,                   # image width
            "id": 0                         # image identity id, important to associate with annotations
        },
            ...
    ]
    ---
    annotations: [
        {
            "keypoints": [x1,y1,v1,x2,y2,v2...],      # keypoints name, 18 * 3: [x, y, visible]
            "num_keypoints": 18,
            "id": int,                          # annotation identity id
            "image_id": int,                    # associate to image id
            "category_id": 0,                   # associate to category id
            "bbox": [x,y,width,height]          # bounding boxes, default as whole image
            "area": float                       # bounding box area
        }
    ]
    ---
    categories: [
        {
            "supercategory": "vehicle",
            "id": 0,
            "name": "vehicle",
            "keypoints": [
                            "front_lpr_left", "front_light_left", "A_pillar_left", "front_roof_left", 
                            "rear_roof_left", "rear_light_left", "rear_wheel_left", "front_wheel_left",
                            "front_lpr_right", "front_light_right", "A_pillar_right", "front_roof_right",
                            "rear_roof_right", "rear_light_right", "rear_wheel_right", "front_wheel_right",
                            "rear_lpr_left", "rear_lpr_right"],
            "skeleton": []
        }
    ]
    
    -------------
    return: dict
    {
        rel_path: gt_kpt
    }
    > gt_kpt: (np.ndarray[K,3]) The first two dimensions are \
        coordinates, vislibility[v] is the third dimension of the array.
        v=0: not labeled, v=1: labeled but not visible, and v=2: labeled and visible
    '''
    gt_dict = dict()

    if gt_files is None:
        return {}
    
    if not isinstance(gt_files, list):
        gt_files = [gt_files]

    for cur_file in gt_files:
        assert os.path.exists(cur_file), 'file not exists: {}'.format(cur_file)
        gts = read_json(cur_file)
        gt_images = gts['images']
        gt_annos = gts['annotations']
        
        img_info_dict = dict()
        for img in gt_images:
            file_name = img['file_name']
            image_id = img['id']
            img_info_dict[image_id] = file_name

        for ann in gt_annos:

            image_id = ann['image_id']
            file_name = img_info_dict[image_id]

            num_keypoints = ann['num_keypoints']
            keypoints = ann['keypoints']
            arr_keypoints = np.array(keypoints)
            arr_keypoints = arr_keypoints.reshape(num_keypoints, 3)

            gt_dict[file_name] = arr_keypoints
            
    return gt_dict


def parse_predictions(pred_file, root_dir=None):
    
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
    -------------
    return: dict
    {
        rel_path: pred_kpt
    }
    > pred_kpt: (np.ndarray[K,3]) The first two dimensions are \
        coordinates, score is the third dimension of the array.

    '''
    with open(pred_file, 'r', encoding='utf8') as fp:
        results = json.load(fp)

    pred_dict = dict()
    for res_ix, res in enumerate(results):
        batch_size = len(res['preds'])
        for b_idx in range(batch_size):
            pred_kpt = res['preds'][b_idx]
            image_path = res['image_paths'][b_idx]
            if root_dir is not None:
                rel_path = image_path.replace(root_dir, '')
                while rel_path[0] == '/':
                    rel_path = rel_path[1:]
            else:
                rel_path = image_path
            pred_dict[rel_path] = pred_kpt
    return pred_dict
