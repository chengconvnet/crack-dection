from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint


def create_history(save_path_mode, model, epochs, batch_size, train_generator, validation_generator):
    # 模型保存格式默认是saved_model,可以自己定义更改原有类来保存hdf5
    ckpt = ModelCheckpoint(save_path_mode, monitor='val_loss', verbose=1,
                           save_best_only=True, save_weights_only=True)
    history = model.fit(x=train_generator,
                        batch_size=train_generator.n // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // batch_size,
                        callbacks=[ckpt])
    return history


def model_fit_save(model_name, model, model_save_dir, *args):
    save_path_mode = f'{model_save_dir}/{model_name}'
    save_path_mode += '-{epoch:02d}-loss{loss:.2f}-acc{accuracy:.2g}.h5'
    history = create_history(save_path_mode, model, *args)
    return history


def write_csv_result(save_dir, model_name, accs, val_accs, losses, val_losses):
    with open(f"{save_dir}/{model_name}_train_acc.csv", "w") as f:
        [f.write(str(acc)+'\n') for acc in accs]
    with open(f"{save_dir}/{model_name}_train_loss.csv", "w") as f:
        [f.write(str(loss)+'\n') for loss in losses]
    with open(f"{save_dir}/{model_name}_val_acc.csv", "w") as f:
        [f.write(str(acc)+'\n') for acc in val_accs]
    with open(f"{save_dir}/{model_name}_val_loss.csv", "w") as f:
        [f.write(str(loss)+'\n') for loss in val_losses]


def plot_history(history, save_dir, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    write_csv_result(save_dir, model_name, acc, val_acc, loss, val_loss)
    plt.plot(acc, 'b', label='Training acc')
    plt.plot(val_acc, 'r--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(
        f'{save_dir}/{model_name}_training_validation_accuracy.png')
    plt.figure()

    plt.plot(loss, 'b', label='Training loss')
    plt.plot(val_loss, 'r--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(
        f'{save_dir}/{model_name}_training_validation_loss.png')
    plt.show()
