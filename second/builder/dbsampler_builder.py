import pickle

import second.core.preprocess as prep
from second.builder import preprocess_builder
from second.core.preprocess import DataBasePreprocessor
from second.core.sample_ops import DataBaseSamplerV2


def build(sampler_config):
    cfg = sampler_config
    groups = list(cfg.sample_groups)
    prepors = [
        preprocess_builder.build_db_preprocess(c)
        for c in cfg.database_prep_steps
    ]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object  # [0, 0]
    groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.database_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)  # dict
        # db_infos.keys()
        # Out[10]: dict_keys(['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Tram', 'Misc', 'Person_sitting'])
        # db_infos["Car"] is a list of dict
        # len(db_infos["Car"])
        # Out[11]: 14357
        # db_infos["Car"][0]
        # Out[12]:
        # {'name': 'Car',
        #  'path': 'gt_database/3_Car_0.bin',  # gt_database is cropped point cloud for target object
        #  'image_idx': 3,
        #  'gt_idx': 0,
        #  'box3d_lidar': array([13.51070263, -0.98177998, -0.9094899, 1.73000002, 4.1500001,
        #                        1.57000005, 1.62]),
        #  'num_points_in_gt': 674,
        #  'difficulty': 0,
        #  'group_id': 1}
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(db_infos, groups, db_prepor, rate, grot_range)
    return sampler
