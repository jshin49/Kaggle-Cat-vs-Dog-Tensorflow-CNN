class SimpleConfig(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Hyperparameters
    lr = 0.00001        # Learning Rate
    l2 = 0.01           # L2 Loss Coefficient
    dropout = 0.4       # Dropout Rate
    batch_size = 16     # SGD Batch Size
    epochs = 30         # Number of Training Epochs
    threshold = 0.6     # Threshold for accurate classification

    # Data Processing
    image_size = 64    # resize image to image_size*image_size
    channels = 3        # Channel Size

    valid_size = 0.1

    # Saver
    model_name = 'simple'
    ckpt_path = 'ckpt/' + model_name


class DeepConfig(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Hyperparameters
    lr = 0.00001        # Learning Rate
    l2 = 0.01           # L2 Loss Coefficient
    dropout = 0.4       # Dropout Rate
    batch_size = 16     # SGD Batch Size
    epochs = 30         # Number of Training Epochs
    threshold = 0.6     # Threshold for accurate classification

    # Data Processing
    image_size = 64     # resize image to image_size*image_size
    channels = 3        # Channel Size

    valid_size = 0.1

    # Saver
    model_name = 'deep'
    ckpt_path = 'ckpt/' + model_name
