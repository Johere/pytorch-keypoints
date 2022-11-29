import os.path as osp
import numpy as np
import cv2
from PIL import Image
import torch

from mmcv.utils import TORCH_VERSION
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook
from mmpose.core.evaluation.top_down_eval import _get_max_preds


'''
turorial for writing a custom hook

@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

Configure file:
custom_imports = dict(imports=['mmpose.core.utils.my_hook'], allow_failed_imports=False)
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='VERY_LOW')
]
```
'''
@HOOKS.register_module()
class TensorboardImageKeyPointsLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True,
                 vis_cnt=4,
                 rgb_mode=True,
                 tensor_factor=1/255.,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 target_type='heatmap',
                 vis_heatmap=False):
        """
        target_type: choices can be 'heatmap' or 'point'
            > heatmap: FloatTensor, [N, C, H, W]
            > point: NotImplement
        """
        super(TensorboardImageKeyPointsLoggerHook, self).__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir
        self.vis_cnt = vis_cnt
        self.rgb_mode = rgb_mode
        self.tensor_factor = tensor_factor
        self.mean = mean
        self.std = std

        self.target_type = target_type
        self.vis_heatmap = vis_heatmap
        if self.target_type is not 'heatmap':
            assert not self.vis_heatmap, 'unable to vis heatmaps, because target type is not `heatmap`'
        # for heatmap parser
        kernel_size = 5
        self.pool = torch.nn.MaxPool2d(kernel_size, 1, kernel_size // 2)

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def before_train_epoch(self, runner):
        self.dataloader_iterator = iter(runner.data_loader)
    
    @master_only
    def before_train_iter(self, runner):
        self.current_data_batch = next(self.dataloader_iterator)
    
    def parse_keypoints(self, kpt_target, point_scale):
        """
        Args:
            kpt_target (tensor[K, H, W]): keypoint target heatmaps

        (Pdb) img_metas.keys()
        dict_keys(['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'bbox_id'])
        """
        # parse keypoints
        if self.target_type == 'heatmap':
            kpt_target = np.array(kpt_target.unsqueeze(0))
            """
            Returns:
                tuple: A tuple containing keypoint predictions and scores.

                - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
                - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
            
            (Pdb) preds.shape
            (1, 18, 2)

            """
            preds, maxvals = _get_max_preds(kpt_target)
            points = preds[0]
            points[..., 0] = points[..., 0] * point_scale[0]
            points[..., 1] = points[..., 1] * point_scale[1]

            return points
            
        elif self.target_type == 'point':
            raise NotImplementedError
        else:
            raise ValueError('unknown keypoints target type: {}'.format(self.target_type))
    
    def get_vis_heatmap(self, heatmap, vis_height=-1, vis_width=-1):
        """
        visualize heatmaps for keypoints
        :param heatmap:  [K, H, W]
        :return:
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        heatmap_num = heatmap.shape[0]

        feat = None
        for index in range(0, heatmap_num):
            if feat is None:
                feat = heatmap[index]
            else:
                feat = feat + heatmap[index]

        vis_heatmap = feat / np.max(feat)
        vis_heatmap = np.uint8(255 * vis_heatmap)
        vis_heatmap = cv2.applyColorMap(vis_heatmap, cv2.COLORMAP_JET)

        if vis_height > 0 and vis_width > 0:
            if vis_heatmap.shape[0] != vis_height or vis_heatmap.shape[1] != vis_width:
                vis_heatmap = cv2.resize(vis_heatmap, (vis_height, vis_width))

        return vis_heatmap

    def get_vis_data(self, tensor, points=None, vis_width=-1, vis_height=-1):
        """
        (Pdb) preds.shape
        (18, 2)
        """

        # inverse normalization
        if (self.mean is not None) and (self.std is not None):
            assert tensor.shape[0] == len(self.mean) == len(self.std),\
                'invalid mean and std: mean={}, std={}'.format(self.mean, self.std)
            for c in range(len(self.mean)):
                tensor[c] = tensor[c] * self.std[c]
                tensor[c] = tensor[c] + self.mean[c]
        tensor = tensor / self.tensor_factor
        torch.clamp(tensor, 0, 255)

        # tensor -> opencv image, CHW -> HWC
        tensor = tensor.permute(1, 2, 0).data.cpu()
        img = np.array(tensor, dtype=np.uint8)

        if self.rgb_mode:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # plot keypoint target on image
        if points is not None:
            for pt_ix, pt in enumerate(points):
                label = pt_ix + 1
                xx = int(pt[0])
                yy = int(pt[1])
                if xx < 0  or yy < 0:
                    continue
                cv2.circle(img, (xx + 2, yy + 2), 3, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(img, str(label), (xx + 2, yy + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if vis_height > 0 and vis_width > 0:
            if img.shape[0] != vis_height or img.shape[1] != vis_width:
                img = cv2.resize(img, (vis_height, vis_width))

        return img

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = f'{var}/{runner.mode}'
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)

        '''
        (Pdb) self.current_data_batch.keys()
        dict_keys(['img', 'target', 'target_weight', 'img_metas'])

        (Pdb) self.current_data_batch['img'].shape
        torch.Size([32, 3, 256, 256])

        (Pdb) self.current_data_batch['target'].shape
        torch.Size([32, 18, 64, 64])

        (Pdb) len(self.current_data_batch['img_metas'].data[0])
        32
        (Pdb) img_metas_batch.data[0][0].keys()
        dict_keys(['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs', 'bbox_id'])


        '''
        img_batch = self.current_data_batch['img']
        target_batch = self.current_data_batch['target']
        # img_metas_batch = self.current_data_batch['img_metas'].data[0]
        batch_size = len(img_batch)
        selected_indices = np.random.choice(np.arange(batch_size), min(self.vis_cnt, batch_size), replace=False)

        vis_height = 400
        vis_width = 400
        if self.vis_heatmap:
            rst = np.zeros((vis_height, vis_width * len(selected_indices) * 2, 3))
        else:
            rst = np.zeros((vis_height, vis_width * len(selected_indices), 3))
        for pos, batch_ix in enumerate(selected_indices):

            _, img_h, img_w = img_batch[batch_ix].shape
            _, hm_h, hm_w = target_batch[batch_ix].shape
            point_scale = [img_h / hm_h, img_w / hm_w]
            
            # vis_one_from_batch
            points = self.parse_keypoints(target_batch[batch_ix], point_scale=point_scale)
            vis_img = self.get_vis_data(img_batch[batch_ix], points)
            vis_heatmap = self.get_vis_heatmap(heatmap=target_batch[batch_ix], vis_height=img_h, vis_width=img_w)

            if self.vis_heatmap:
                left = vis_width * 2 * pos
                right = left + vis_width
                rst[:, left: right, :] = cv2.resize(vis_img, (vis_height, vis_width))
                rst[:, right: right + vis_width, :] = cv2.resize(vis_heatmap, (vis_height, vis_width))
            else:
                left = vis_width * pos
                right = left + vis_width
                rst[:, left: right, :] = cv2.resize(vis_img, (vis_height, vis_width))

        # add image to tensorboard
        rst = rst.clip(0, 255).astype(np.uint8, copy=False)
        # prepare for tensorboard [RGB, CHW]
        rst = cv2.cvtColor(rst, cv2.COLOR_BGR2RGB)
        rst = rst.transpose((2, 0, 1))  # HWC -> CHW
        self.writer.add_image('train_images', rst, runner.iter)

    @master_only
    def after_run(self, runner):
        self.writer.close()