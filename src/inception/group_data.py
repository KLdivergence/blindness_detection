import os
import os.path as op

import pandas as pd
from tqdm import tqdm


def group_images(path, label_csv):
    """Group images to subdirectories based on their labels
    
    Parameters:
        path (str): Path to the image folder
        label_csv (str): Path to the csv file containing image labels
    """

    labels = load_labels(label_csv)
    images = os.scandir(path)
    file_count = len(next(os.walk(path))[2])

    for image in tqdm(images, total=file_count):
        try:
            label = labels[image.name.split(".")[0]]
        except KeyError:
            continue
        
        des_path = op.join(op.dirname(image.path), str(label))
        if not op.isdir(des_path):
            os.makedirs(des_path)
        os.rename(image.path, op.join(des_path, image.name))
        

def load_labels(csv_file):
    """Load image labels from csv file
    
    Parameters:
        csv_file (str): path to the csv file

    Returns:
        dict: keys are the image names and values are the labels
    """

    label_df = pd.read_csv(csv_file)
    cols = label_df.columns
    label_dict = dict(zip(label_df[cols[0]], label_df[cols[1]]))

    return label_dict


def create_val()

if __name__ == "__main__":
    group_images("../../data/train_images", "../../data/train.csv")
