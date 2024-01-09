import h5py
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    semantic = f['semantic'][:]
    model_id = f['model_id'][:]

    return data, label, semantic, model_id


class shapenet_TEST_dataset(Dataset):
    def __init__(self, config):
        if config["complementme"]:

            filename = os.path.join(config["base_dir"], "generated_datasplits_complementme", config["middle_name"],
                                    "generated_datasplits_complementme",
                                    config["category"] + '_' + str(config["num_source"])
                                    + '_' + config["mode"] + ".h5")
        else:
            filename = os.path.join(config["base_dir"], "generated_datasplits", config["middle_name"],
                                    "generated_datasplits", config["category"] + '_' + str(config["num_source"])
                                    + '_' + config["mode"] + ".h5")
#            filename = os.path.join(config["base_dir"], "generated_datasplits", config["middle_name"],
#                                    "datasplits_randRT", config["category"] + '_' + str(config["num_source"])
#                                    + '_' + config["mode"] + '_' + "randRT" + ".h5")

        self.dis_mat_path = os.path.join(config["base_dir"], "dis_mat", config["category"])

        all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
        self.target_points = all_target_points
        self.target_labels = all_target_labels
        self.target_semantics = all_target_semantics
        self.target_ids = all_target_model_id

        self.n_samples = all_target_points.shape[0]
        self.random_rot = config["random_rot"]

        print("Number of targets: " + str(self.n_samples))

    def __getitem__(self, index):
        # occlusion handling
        points = self.target_points[index]  # size 2048 x 3
        # note that ids and labels  are only used in visualization and retrieval
        ids = self.target_ids[index]  # 1
        labels = self.target_labels[index]  # 2048   view label, from which view
        semantics = self.target_semantics[index]  # 2048  part segementation
        ##  randomly generate occ points
#        choose_one_occ = np.random.rand()
#        if choose_one_occ < 0.3:
#            points_occ, points_occ_mask = generate_occ_point_ball(points, ids, save_pth=self.dis_mat_path)
#        elif choose_one_occ < 0.6:
#            points_occ, points_occ_mask = generate_occ_point_random(points)
#        elif choose_one_occ < 0.9:
#            points_occ, points_occ_mask = generate_occ_point_slice(points)
#        else:
#            points_occ, points_occ_mask = generate_occ_point_part(points, semantics)
        # focalization
#        ori_point_occ = points_occ
#        points_occ_mean = np.mean(points_occ, axis=0, keepdims=True)
#        points_occ = points_occ - points_occ_mean
        
#        return points, ids, labels[points_occ_mask], semantics[points_occ_mask], points_occ, points_occ_mask, ori_point_occ
        return points
        
    def __len__(self):
        return self.n_samples