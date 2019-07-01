import os.path as op

from tensorflow.keras.preprocessing import 

class LoadData:
    """Load images from data folder"""

    def __init__(self):
        """Initialize required class variables"""

        datapath = op.join(op.pardir, "data")
        self.train_images_path = op.join(datapath, "train_images")
        self.test_images_path = op.join(datapath, "test_images")
        self.train_labels_csv = op.join(datapath, "train.csv")

    def train(self, batch_size: int):
        """Generator for training set"""

        pass

    def test(self, batch_size: int):
        """Generator for testing set"""

        pass
