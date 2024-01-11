from tqdm import tqdm
import cv2
import os
from imgaug import augmenters as iaa

image_size = 224
labels = ['COVID', 'Lung_OpacityCroped', 'Normal_croped']
x_train = []  # training images.
y_train = []  # training labels.
x_test = []  # testing images.

def load_data():
    for label in labels:
        trainPath = os.path.join(r'./', label)
        for file in tqdm(os.listdir(trainPath)):
            if (file.endswith('.png')):
                image = cv2.imread(os.path.join(trainPath, file), 0)  # load images in gray.
                image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
                image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # produce a pseudocolored image.
                image = cv2.resize(image, (image_size, image_size))  # resize images into 150*150.
                x_train.append(image)
                y_train.append(labels.index(label))
    testPath = os.path.join(r'./', 'Viral_Pneumonia')
    for file in tqdm(os.listdir(testPath)):
        image = cv2.imread(os.path.join(testPath, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
    X_augmented = augment_data(x_test)
    X_all = x_train + x_test + X_augmented
    Y_all = y_train + [3] * 1345 * 3
    # Y_all = y_train + [3] * 300
    return X_all,Y_all

def augment_data(x_values):
    RESIZE_DIM=224
    X_values_augmented = []
    for x in x_values:
        images_aug = seq1().augment_images(x.reshape(1,RESIZE_DIM,RESIZE_DIM,3))
        X_values_augmented.append(images_aug.reshape(RESIZE_DIM,RESIZE_DIM,3))
        images_aug = seq2().augment_images(x.reshape(1,RESIZE_DIM,RESIZE_DIM,3))
        X_values_augmented.append(images_aug.reshape(RESIZE_DIM,RESIZE_DIM,3))
    # X_values_augmented = np.asarray( X_values_augmented )
    return (X_values_augmented)

def seq1():
    seq = iaa.OneOf([
        iaa.Affine(rotate=(-15, 15)),  # 旋转-15到15度
        iaa.Affine(scale=(0.5, 2)),  # 随机缩放图片0.5~2倍
        iaa.GaussianBlur(sigma=(0, 0.5))  # 添加高斯噪声
    ])
    return seq
def seq2():
    seq = iaa.OneOf([
        iaa.Fliplr(),  # 水平翻转图像
        iaa.Flipud(),  # 垂直翻转图像
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # 平移图像
    ])
    return seq


