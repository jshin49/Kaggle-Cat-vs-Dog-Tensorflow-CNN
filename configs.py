class SimpleConfig(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Hyperparameters
    lr = 0.001          # Learning Rate
    l2 = 0.001          # L2 Loss Coefficient
    dropout = 0.5       # Dropout Rate
    batch_size = 16     # SGD Batch Size
    epochs = 15         # Number of Training Epochs
    threshold = 0.5     # Threshold for accurate classification

    # Data Processing
    train_dir = './data/train'
    test_dir = './data/test'
    image_dir = './arrays/'
    image_size = 64     # resize image to image_size*image_size
    channels = 3        # Channel Size

    valid_size = 0.1

    # Saver
    model_name = 'model-lr{}-img{}'.format(lr, image_size)
    ckpt_path = 'ckpt/' + model_name


class DeepConfig(object):
    """
    Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # Hyperparameters
    lr = 0.001          # Learning Rate
    l2 = 0.0001         # L2 Loss Coefficient
    dropout = 0.5       # Dropout Rate
    batch_size = 32     # SGD Batch Size
    epochs = 30         # Number of Training Epochs
    threshold = 0.5     # Threshold for accurate classification

    # Data Processing
    train_dir = './data/train'
    test_dir = './data/test'
    image_dir = './arrays/'
    image_size = 224    # resize image to image_size*image_size
    channels = 3        # Channel Size
    valid_size = 1000
    train_size = 24000

    # Saver
    model_name = 'model-lr{}-img{}'.format(lr, image_size)
    ckpt_path = 'ckpt/' + model_name
