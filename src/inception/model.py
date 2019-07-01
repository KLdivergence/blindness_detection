from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


def create_model(lr=0.001):
    """Create compiled keras model

    Parameters:
        lr (float): learning rate

    Returns:
        keras.Model: a instance of keras.Model compiled with
            categorical_crossentropy loss and adam optimizer
    """

    optimizer = Adam(lr=lr)
    model = InceptionV3(include_top=True, weights=None, classes=5)
    model.compile(loss=categorical_crossentropy, optimizer=optimizer)
    return model
