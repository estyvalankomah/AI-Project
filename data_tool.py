import pandas as pd
import numpy as np
from scipy.misc import imread
import config
from tool import img_path_to_GEI
import logging
import os
logger = logging.getLogger("data")


def load_training_validation_data(train_dir=None, val_dir=None):
    age_group = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    train_dir_path = "%s\\GEI_IDList_train.csv" % config.Project.gait_dataset
    test_dir_path = "%s\\GEI_IDList_test.csv" % config.Project.gait_dataset
    train_files = pd.read_csv (train_dir_path, dtype = str)
    train_id = train_files.ID
    test_files = pd.read_csv (test_dir_path, dtype = str)
    test_id = test_files.ID
    if train_dir is None:
        train_dir = "train_dir"
    if val_dir is None:
        val_dir = "test_dir"

    training_x = []
    training_y = []

    validation_x = []
    validation_y = []

    for group in age_group:
        for id in train_id:
            img_path = "%s\%s\%s\%s.png" % (config.Project.gait_dataset, group, train_dir, id)
            if not os.path.isfile(img_path):
                continue
            else:
                img_dir = []
                im = imread(img_path)
                img_dir.append(im)
                data = img_path_to_GEI(img_dir)

                if len(data.shape) > 0:
                    training_x.append(data)
                    training_y.append(group)
                else:
                    logger.warning("fail to extract %s " % img_path)

        for id in test_id:
            img_path = "%s\%s\%s\%s.png" % (config.Project.gait_dataset, group, val_dir, id)
            if not os.path.isfile(img_path):
                continue
            else:
                img_dir = []
                im = imread(img_path)
                img_dir.append(im)
                data = img_path_to_GEI(img_dir)

                if len(data.shape) > 0:
                    validation_x.append(data)
                    validation_y.append(group)
                else:
                    logger.warning("fail to extract %s " % img_path)

            
    return training_x, training_y, validation_x, validation_y


