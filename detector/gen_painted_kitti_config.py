import sys
import yaml
from pathlib import Path
from easydict import EasyDict
from pcdet.datasets.painted_kitti.painted_kitti_dataset import create_kitti_infos

if __name__ == '__main__':
    if sys.argv.__len__() > 1 and sys.argv[1].endswith('.yaml'):
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[1])))
        ROOT_DIR = Path(__file__).resolve().parent.resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )