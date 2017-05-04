# Kaggle-Cat-vs-Dog-Tensorflow-CNN

TensorFlow with tf.layers Implementation of Kaggle Cat vs Dog Challenge

Preparations
1. ```mkdir ckpt```
2. ```mkdir graphs```
3. ```mkdir batches```
4. ```mkdir npys```
5. unzip data in ```./data```
6. ```pip install -r requirements.txt```
7. Install OpenCV 2.4.13 for Python

Steps
1. ```python utils/prepare_data.py```
2. To train Deep CNN Model, run: ```python train_cnn.py DEEP```
3. Else, run: ```python train_cnn.py SIMPLE```