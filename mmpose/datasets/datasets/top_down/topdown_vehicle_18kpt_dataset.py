# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict
import copy
import cv2
import time

import json_tricks as json
import numpy as np
from mmcv import Config, deprecated_api_warning
from xtcocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset
from mmpose.utils.misc_helper import try_decode


class ScalableCOCO(COCO):
    def __init__(self, annotation_file=None, image_folder=None, test_index=None, ann_data=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (list(str)): location of annotation files
        :param image_folder (list(str)): location to the folder that hosts images.
        :return:

        Very similar to original COCO API, except for supporting multiple annotation_files
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.test_index = test_index
        
        self.anno_file = annotation_file
        if not isinstance(self.anno_file, list):
            self.anno_file = [self.anno_file]
        self.image_folder = image_folder
        if not isinstance(self.anno_file, list):
            self.image_folder = [self.image_folder]
        
        if not annotation_file == None:

            print('loading annotations into memory...')
            tic = time.time()
            # https://github.com/cocodataset/cocoapi/pull/453/
            if ann_data == None:
                dataset_list = []
                for cur_prefix, cur_anno_file in zip(self.image_folder, self.anno_file):
                    with open(cur_anno_file, 'r') as f:
                        cur_dataset = json.load(f)
                        # merge img_prefix
                        if 'images' in cur_dataset:
                            for img in cur_dataset['images']:
                                rel_path = img['file_name']
                                img['file_name'] = osp.join(cur_prefix, rel_path)
                    dataset_list.append(cur_dataset)

                dataset = self.extendDataset(dataset_list)
            else:
                dataset = ann_data
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
        
        if 'annotations' in self.dataset:
            for i in range(len(self.dataset['annotations'])):
                if self.test_index is not None:
                    keypoints = np.array(self.dataset['annotations'][i]['keypoints']).reshape([-1, 3])
                    keypoints = keypoints[self.test_index, :]
                    self.dataset['annotations'][i]['keypoints'] = keypoints.reshape([-1]).tolist()
                if 'iscrowd' not in self.dataset['annotations'][i]:
                    self.dataset['annotations'][i]['iscrowd'] = False

    def extendDataset(self, dataset_list):
        """
        merge multiple coco-style dataset
        should be careful with some identical ids:
            > `images.id` (image identity)
            > `annotations.id` (annotation identity)
            > `annotations.image_id` (annotation associate to image identity)
        for sanity, check multiple datasets share the same category info:
            > `categories`  (simply compare str)
        """

        if len(dataset_list) == 1:
            return dict(dataset_list[0])

        max_image_id = 0
        max_anno_id = 0

        dataset = None
        for cur_dataset in dataset_list:

            '''
            (Pdb) cur_dataset.keys()
            odict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
            '''
            if dataset is not None:
                assert len(dataset['categories']) == len(cur_dataset['categories']), \
                    'mismatched categories in multiple datasets: {} vs. {}'.format(dataset['categories'], cur_dataset['categories'])
                for cat1, cat2 in zip(dataset['categories'], cur_dataset['categories']):
                    assert cat1 == cat2, \
                    'mismatched categories in multiple datasets: {} vs. {}'.format(dataset['categories'], cur_dataset['categories'])

            image_id_mapping = dict()
            if 'images' in cur_dataset:
                '''
                (Pdb) cur_dataset['images'][0].keys()
                odict_keys(['license', 'file_name', 'height', 'width', 'id'])
                '''
                for img in cur_dataset['images']:
                    if img['id'] > max_image_id:
                        max_image_id = img['id']
                    else:
                        max_image_id += 1
                        # assign a new image id exceeding the max id to avoid overlapping
                        image_id_mapping[img['id']] = max_image_id
                        img['id'] = max_image_id

            if 'annotations' in cur_dataset:
                '''
                (Pdb) cur_dataset['annotations'][0].keys()
                odict_keys(['keypoints', 'num_keypoints', 'id', 'image_id', 'category_id', 'bbox', 'area'])
                '''
                for anno in cur_dataset['annotations']:
                    ori_image_id = anno['image_id']
                    if ori_image_id in image_id_mapping.keys():
                        new_image_id = image_id_mapping[ori_image_id]
                        anno['image_id'] = new_image_id
                    if anno['id'] > max_anno_id:
                        max_anno_id = anno['id']
                    else:
                        max_anno_id += 1
                        # assign a new image id exceeding the max id to avoid overlapping
                        anno['id'] = max_anno_id

            import pdb; pdb.set_trace()
            # extend dataset
            if dataset is None:
                dataset = cur_dataset
            else:
                dataset['images'].extend(cur_dataset['images'])
                dataset['annotations'].extend(cur_dataset['annotations'])
        return dict(dataset)

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        import pdb; pdb.set_trace()
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats


@DATASETS.register_module()
class TopDownVehicleDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Vehicle keypoints dataset for top-down pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Vehicle keypoint indexes::

        0: "front_lpr_left",
        1: "front_light_left",
        2: "A_pillar_left",
        3: "front_roof_left",
        4: "rear_roof_left",
        5: "rear_light_left",
        6: "rear_wheel_left",
        7: "front_wheel_left",
        8: "front_lpr_right",
        9: "front_light_right",
        10: "A_pillar_right",
        11: "front_roof_right",
        12: "rear_roof_right",
        13: "rear_light_right",
        14: "rear_wheel_right",
        15: "front_wheel_right",
        16: "rear_lpr_left",
        17: "rear_lpr_right"

    Args:
        ann_file list((str)): Path to the annotation file.
        img_prefix list((str)): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/coco.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
            coco_style=True)
        
        self.ann_info['front_vehicle_ids'] = self.parsed_dataset_info.front_vehicle_ids
        self.ann_info['mid_vehicle_ids'] = self.parsed_dataset_info.mid_vehicle_ids
        self.ann_info['rear_vehicle_ids'] = self.parsed_dataset_info.rear_vehicle_ids

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.db = self._get_db()

        print(f'parsed done from annotation file: {self.ann_file } \n=> num_images: {self.num_images}, load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_vehicle_keypoint_annotations()
        else:
            # use bbox from detection
            # gt_db = self._load_coco_person_detection_results()
            raise NotImplementedError('Not support for test mode')
        return gt_db

    def _load_vehicle_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_vehicle_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_vehicle_keypoint_annotation_kernel(self, img_id):
        """load annotation from vehicle dataset using COCOAPI.

        Note:
            bbox:[x1, y1, w, h]

        Args:
            img_id: coco image id

        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            image_file = osp.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
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
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = defaultdict(list)

        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts[image_id].append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                if kwargs.get('rle_score', False):
                    pose_score = n_p['keypoints'][:, 2]
                    n_p['score'] = float(box_score + np.mean(pose_score) +
                                         np.max(pose_score))
                else:
                    kpt_score = 0
                    valid_num = 0
                    for n_jt in range(0, num_joints):
                        t_s = n_p['keypoints'][n_jt][2]
                        if t_s > vis_thr:
                            kpt_score = kpt_score + t_s
                            valid_num = valid_num + 1
                    if valid_num != 0:
                        kpt_score = kpt_score / valid_num
                    # rescoring
                    n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_coco_keypoint_results(valid_kpts, res_file)

        # do evaluation only if the ground truth keypoint annotations exist
        if 'annotations' in self.coco.dataset:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)

            if tmp_folder is not None:
                tmp_folder.cleanup()
        else:
            warnings.warn(f'Due to the absence of ground truth keypoint'
                          f'annotations, the quantitative evaluation can not'
                          f'be conducted. The prediction results have been'
                          f'saved at: {osp.abspath(res_file)}')
            name_value = {}

        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', self.sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts


@DATASETS.register_module()
class TopDownSimpleImageDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Vehicle keypoints dataset for top-down pose estimation.
    simple datasets, input: img_list

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        ann_file (str): Path to the image list file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=True):
        assert test_mode, 'TopDownSimpleImageDataset can only be used in test mode'

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
            coco_style=False)

        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')
    
    @staticmethod
    def parse_list(list_file, root_dir=None):
        """
        To parse list files defined in hybrid dataset format:
        first line:
            # path height width
        other lines:
            00001_c001_1.jpg 240 350
            ...
        :param path:
        :return:
        """
        assert osp.exists(list_file), 'file not exists:{}'.format(list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()

        items = []
        keys = None
        for ln in lines:
            if ln.startswith('#'):
                ln = ln.replace('#', '')
                keys = ln.strip().split()
                continue
            content = ln.strip().split()
            assert len(content) == len(keys), 'invalid content:{}, while keys:{}'.format(content, keys)

            item = dict()
            for k, v in zip(keys, content):
                item[k] = try_decode(v)
            
            if root_dir is not None:
                if 'path' in item.keys():
                    path = osp.join(root_dir, item['path'])
                    item['path'] = path
            items.append(item)

        return items


    def _get_db(self):
        """Load images as the test dataset."""

        num_joints = self.ann_info['num_joints']

        items = self.parse_list(self.ann_file)

        kpt_db = []
        img_id = 0
        for item in items:
            image_file = osp.join(self.img_prefix, item['path'])

            if 'height' in item.keys() and 'width' in item.keys():
                height = item['height']
                width = item['width']
            else:
                img = cv2.imread(image_file)
                height, width, _ = img.shape

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'rotation': 0,
                'bbox': [0, 0, width, height],
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': img_id
            })
            img_id = img_id + 1

            # # debug
            # break

        self.num_images = len(kpt_db)
        return kpt_db
    
    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""
        warnings.warn('Due to the absence of ground truth keypoint annotations, '\
                      'the quantitative evaluation can not be conducted.')
        name_value = {}

        return name_value
    
    # def _write_coco_keypoint_results(self, keypoints, res_file):
    #     """Write results into a json file."""

    #     results = []

    #     for img_kpts in keypoints:
    #         if len(img_kpts) == 0:
    #             continue

    #         _key_points = np.array(
    #             [img_kpt['keypoints'] for img_kpt in img_kpts])
    #         key_points = _key_points.reshape(-1,
    #                                          self.ann_info['num_joints'] * 3)

    #         result = [{
    #             'image_id': img_kpt['image_id'],
    #             'keypoints': key_point.tolist(),
    #             'score': float(img_kpt['score']),
    #             'center': img_kpt['center'].tolist(),
    #             'scale': img_kpt['scale'].tolist()
    #         } for img_kpt, key_point in zip(img_kpts, key_points)]

    #         results.extend(result)

    #     with open(res_file, 'w') as f:
    #         json.dump(results, f, sort_keys=True, indent=4)


    # def results2json(self, results, res_folder=None, **kwargs):
    #     """The pose prediction results will be
    #     saved in ``${res_folder}/result_keypoints.json``.

    #     Note:
    #         - batch_size: N
    #         - num_keypoints: K
    #         - heatmap height: H
    #         - heatmap width: W

    #     Args:
    #         results (list[dict]): Testing results containing the following
    #             items:

    #             - preds (np.ndarray[N,K,3]): The first two dimensions are \
    #                 coordinates, score is the third dimension of the array.
    #             - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
    #                 scale[1],area, score]
    #             - image_paths (list[str]): For example, ['data/coco/val2017\
    #                 /000000393226.jpg']
    #             - heatmap (np.ndarray[N, K, H, W]): model output heatmap
    #             - bbox_id (list(int)).
    #         res_folder (str, optional): The folder to save the testing
    #             results. If not specified, a temp folder will be created.
    #             Default: None.
    #         metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

    #     """
        
    #     if res_folder is not None:
    #         tmp_folder = None
    #         res_file = osp.join(res_folder, 'result_keypoints.json')
    #     else:
    #         tmp_folder = tempfile.TemporaryDirectory()
    #         res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

    #     kpts = defaultdict(list)

    #     for result in results:
    #         preds = result['preds']
    #         boxes = result['boxes']
    #         image_paths = result['image_paths']
    #         bbox_ids = result['bbox_ids']

    #         batch_size = len(image_paths)
    #         for i in range(batch_size):
    #             kpts[image_id].append({
    #                 'keypoints': preds[i],
    #                 'center': boxes[i][0:2],
    #                 'scale': boxes[i][2:4],
    #                 'area': boxes[i][4],
    #                 'score': boxes[i][5],
    #                 'bbox_id': bbox_ids[i]
    #             })

    #     # rescoring and oks nms
    #     num_joints = self.ann_info['num_joints']
    #     vis_thr = self.vis_thr
    #     oks_thr = self.oks_thr
    #     valid_kpts = []
    #     for image_id in kpts.keys():
    #         img_kpts = kpts[image_id]
    #         for n_p in img_kpts:
    #             box_score = n_p['score']
    #             if kwargs.get('rle_score', False):
    #                 pose_score = n_p['keypoints'][:, 2]
    #                 n_p['score'] = float(box_score + np.mean(pose_score) +
    #                                      np.max(pose_score))
    #             else:
    #                 kpt_score = 0
    #                 valid_num = 0
    #                 for n_jt in range(0, num_joints):
    #                     t_s = n_p['keypoints'][n_jt][2]
    #                     if t_s > vis_thr:
    #                         kpt_score = kpt_score + t_s
    #                         valid_num = valid_num + 1
    #                 if valid_num != 0:
    #                     kpt_score = kpt_score / valid_num
    #                 # rescoring
    #                 n_p['score'] = kpt_score * box_score

    #         if self.use_nms:
    #             nms = soft_oks_nms if self.soft_nms else oks_nms
    #             keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
    #             valid_kpts.append([img_kpts[_keep] for _keep in keep])
    #         else:
    #             valid_kpts.append(img_kpts)

    #     self._write_coco_keypoint_results(valid_kpts, res_file)

    #     return res_file
