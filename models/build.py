from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import ResNet152


class ModelA(Model):
    def __init__(self, base):
        super().__init__()
        # self.base = ResNet152(include_top=False, weights='imagenet')
        self.base = base
        # 冻结基网络
        self.base.trainable = False  # 凍結權重
        self.net = Sequential()
        for layer in [GlobalAvgPool2D(), Dense(512, activation='relu'), Dense(1, activation='sigmoid')]:
            self.net.add(layer)

    def call(self, inputs):
        xs = self.base(inputs)
        xs = self.net(xs)
        return xs


class ModelB(Model):
    def __init__(self, base):
        super().__init__()
        # self.base = ResNet152(include_top=False, weights='imagenet')
        self.base = base
        # 冻结基网络
        self.base.trainable = False  # 凍結權重
        self.net = Sequential()
        for layer in [Flatten(), Dense(512, activation='relu'), Dense(1, activation='sigmoid')]:
            self.net.add(layer)

    def call(self, inputs):
        xs = self.base(inputs)
        xs = self.net(xs)
        return xs


class ModelC(Model):
    def __init__(self, base):
        super().__init__()
        # self.base = ResNet152(include_top=False, weights='imagenet')
        self.base = base
        # 解凍有節點層
        unfreeze = ['conv5_block3_1_conv', 'conv5_block3_1_bn', 'conv5_block3_2_conv',
                    'conv5_block3_2_bn', 'conv5_block3_3_conv', 'conv5_block3_3_bn']
        for layer in self.base.layers:
            if layer.name in unfreeze:
                layer.trainable = True  # 解凍
            else:
                layer.trainable = False  # 其他凍結權重

        self.net = Sequential()
        for layer in [Flatten(), Dense(512, activation='relu'), Dense(1, activation='sigmoid')]:
            self.net.add(layer)

    def call(self, inputs):
        xs = self.base(inputs)
        xs = self.net(xs)
        return xs


def set_resnet(model_class, optimizer,
               loss='binary_crossentropy',
               metrics=['accuracy']):
    # 建立基网络
    base = ResNet152(include_top=False, weights='imagenet')
    net = model_class(base)
    net.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    return net
