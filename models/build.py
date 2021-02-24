from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D


def create_modelA(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.inputs,
                  outputs=prediction)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def create_modelB(base_model):
    # ---- 建立分類模型 ---- #
    model = Sequential()
    model.add(base_model)    # 將模型做為一層
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(1, activation='sigmoid'))
    #for layer in base_model.layers:
    #layer.trainable = False
    base_model.trainable = False     # 凍結權重
    return model


def create_modelC(base_model):
    # ---- 建立分類模型 ---- #
    model = Sequential()
    model.add(base_model)    # 將模型做為一層
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(1, activation='sigmoid'))
    #for layer in base_model.layers:
    #layer.trainable = False
    return model
