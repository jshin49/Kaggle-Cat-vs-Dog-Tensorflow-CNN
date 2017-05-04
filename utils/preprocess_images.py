import cv2
import numpy as np


def process_image(img, img_size, pixel_depth):
    '''
        Maintain aspect ratio of image while resizing
    '''
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    if (img.shape[0] >= img.shape[1]):  # height is greater than width
        resizeto = (img_size, int(
            round(img_size * (float(img.shape[1]) / img.shape[0]))))
    else:
        resizeto = (
            int(round(img_size * (float(img.shape[0]) / img.shape[1]))), img_size)

    img = cv2.resize(img, (resizeto[1], resizeto[
        0]), interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(
        img, 0, img_size - img.shape[0], 0, img_size - img.shape[1], cv2.BORDER_CONSTANT, 0)

    img = normalize_image(img, pixel_depth)

    return img[:, :, ::-1]  # turn into rgb format


def normalize_image(image, pixel_depth):
    image_data = np.array(image, dtype=np.float32)
    image_data[:, :, 0] = (image_data[:, :, 0].astype(
        float) - pixel_depth / 2) / pixel_depth
    image_data[:, :, 1] = (image_data[:, :, 1].astype(
        float) - pixel_depth / 2) / pixel_depth
    image_data[:, :, 2] = (image_data[:, :, 2].astype(
        float) - pixel_depth / 2) / pixel_depth

    return image_data
