from abc import ABCMeta, abstractmethod
import numpy as np
import os, cv2, random


class DataSet(metaclass=ABCMeta):
    @abstractmethod
    def train_data(self):
        pass

    @abstractmethod
    def test_data(self):
        pass


def normalize(x):
    x = x.astype('float32').transpose((2, 0, 1)) / 127.5 - 1.0
    return x.astype('float32')


def denormalize(x):
    x = (x + 1.0) * 127.5

    x = np.round(x * 255)
    x = np.clip(x, 0, 255)

    x = x.transpose((1, 2, 0))
    return x.astype(np.uint8)


class ImageDataSet(DataSet):
    def __init__(self, data_dir, test_count=32):
        super(DataSet).__init__()
        self.data_dir = data_dir

        file_list = []
        for root, dirs, files in os.walk(data_dir):
            for file_name in files:
                _, ext = os.path.splitext(file_name)
                if ext in ['.jpg', '.png', 'jpeg', '.bmp']:
                    file = os.path.join(root, file_name)
                    file_list.append(file)

        self.train_list = file_list[:-test_count]
        self.test_list = file_list[-test_count:]

    @staticmethod
    def load_image(file):
        im = cv2.imread(file)
        return normalize(im)

    def train_data(self):
        random.shuffle(self.train_list)
        for file in self.train_list:
            yield self.load_image(file)

    def test_data(self):
        for file in self.test_list:
            yield self.load_image(file)


if __name__ == '__main__':
    dataset = ImageDataSet("/home/killf/dataset/ffhq/thumbnails128x128")
    for img in dataset.train_data():
        print(img.shape)
        exit()
