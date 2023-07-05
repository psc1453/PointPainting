import argparse
import glob
import os

import math
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar_painted_zte.yaml',
                        help='specify the config for demo')
    parser.add_argument('--root_dir', type=str, default=Path(__file__).resolve().parent.parent,
                        help='root dir of the project.(detector dir')

    parser.add_argument('--data_path', type=str, default='../data/zte/training/painted_lidar/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str,
                        default='../output/kitti_models/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = args.root_dir

    return args, cfg


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def trans_xyx(original_xyz):
    calib_file = "/home/psc/Documents/Study/Research/ZTE/LiDAR-Fusion/PointPainting/detector/data/zte/cali_2.txt"
    with open(calib_file) as f:
        lines = f.readlines()
    obj = lines[2].strip().split(' ')[1:]
    cam2velo = np.array(obj, dtype=np.float32).reshape(4, 4)
    invR = cam2velo[:3, :3].transpose()
    invT = -invR.dot(cam2velo[:3, 3])
    center = np.array(original_xyz)
    cam_coords = invR.dot(center.transpose()) + invT
    return cam_coords[0], cam_coords[1], cam_coords[2]


def get_rox_y(original_rotation_y):
    original_rotation_y = (-math.pi / 2 - original_rotation_y) % (2 * math.pi)
    if original_rotation_y > math.pi:
        original_rotation_y -= 2 * math.pi
    elif original_rotation_y < -math.pi:
        original_rotation_y += 2 * math.pi
    return original_rotation_y


def get_alpha(original_xyz):
    x, y, z = original_xyz
    return math.atan(x / z)


if __name__ == '__main__':
    args, cfg = parse_config()
    log_file = 'detection_result/log.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    inference_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=inference_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    class_names = cfg.CLASS_NAMES
    for i in range(len(class_names)):
        try:
            os.remove('detection_result/result_%d.txt' % (i + 1))
        except FileNotFoundError:
            print("No previous result, continue.")


    for i, data_dict in tqdm(enumerate(inference_dataset), total=len(inference_dataset), desc="Detection Progress: "):
        batch_dict = inference_dataset.collate_batch([data_dict])
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            pred_result = pred_dicts[0]

            labels = pred_result['pred_labels'].cpu()
            scores = pred_result['pred_scores'].cpu()
            boxes = pred_result['pred_boxes'].cpu()

            for box_index in range(scores.shape[0]):
                frame = i + 3000
                obj_type = labels[box_index]
                confidence = scores[box_index]
                h, w, l, x, y, z, rot_y = boxes[box_index][4], boxes[box_index][5], boxes[box_index][3], \
                    boxes[box_index][0], boxes[box_index][1], boxes[box_index][2], boxes[box_index][6]

                rot_y = get_rox_y(rot_y)
                x, y, z = trans_xyx((x, y, z))
                alpha = get_alpha((x, y, z))

                with open('detection_result/result_%d.txt' % obj_type, 'a+') as f:
                    print('%d,%d,-1,-1,-1,-1,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f'
                          % (frame,
                             obj_type,
                             confidence,
                             h, w, l, x, y, z, rot_y,
                             alpha), file=f)
