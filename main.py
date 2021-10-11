import logging
import numpy as np
import pickle
import config
import os
import data_tool
from sklearn.metrics import accuracy_score
from data_tool import load_training_validation_data
from model.models import RandomForestClassification
from feature.hog import flatten


logger = logging.getLogger("main")

level = logging.INFO
log_filename = '%s.log' % __file__
format = '%(asctime)-12s[%(levelname)s] %(message)s'
datefmt ='%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=level,
                    format=format,
                    filename=log_filename,
                    datefmt=datefmt)

if __name__ == '__main__':
    train_dir = "train_dir"
    val_dir = "test_dir"

    training_x, training_y, validation_x, validation_y = load_training_validation_data(train_dir=train_dir, val_dir=val_dir)
    training_feature_x = [flatten(x) for x in training_x]
    validation_feature_x = [flatten(x) for x in validation_x]
    
    logger.info("start")
    model = RandomForestClassification()
    model.fit(x_train=training_feature_x, y_train=training_y)
    predict_file = "%s\model\predict.pickle" % config.Project.project_path
    pickle.dump(model, open(predict_file, 'wb'))
    predict_y = model.predict(validation_feature_x)
    logger.info("done")