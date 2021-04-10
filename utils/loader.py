from pathlib import Path

import tensorflow as tf
from tensorflow.data import Dataset


def preprocess_image(image, width=224, height=224):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [width, height])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


class Bunch:
    def __init__(self, dirname):
        self.root = Path(dirname)

    @property
    def class_names(self):
        return {file.name: k
                for k, file in enumerate(self.root.iterdir())}

    def __call__(self, match='*.jpg'):
        return {file.as_posix(): self.class_names[file.parent.name]
                for file in self.root.rglob(match)}


class PathDataset:
    def __init__(self, data_dir, match='*.jpg'):
        self.bunch = Bunch(data_dir)
        self.paths = Dataset.from_tensor_slices(list(self.bunch(match).keys()))
        self.labels = Dataset.from_tensor_slices(
            list(self.bunch(match).values()))

    @property
    def images(self):
        return self.paths.map(load_and_preprocess_image)

    def __call__(self):
        return Dataset.zip((self.images, self.labels))
