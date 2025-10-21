import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.random import set_seed
import itertools
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
set_seed(42)

data = pd.read_csv('nn_data.csv')
train = data[data['split'] == 'train']
test = data[data['split'] == 'test']

x_train = train[['x0', 'x1']]
x_test  = test[['x0', 'x1']]
y_train = train['y']
y_test  = test['y']

max_lrs = [0.01, 0.05]
warmup_ratios = [0.1, 0.2]
layer_sizes = [32, 64]
batch_sizes = [32, 64]
val_splits = [0.1, 0.2, 0.3]

total_epochs = 50

best_acc = 0.0
best_config = None

for max_lr, warmup_ratio, layer_size, batch_size, val_split in itertools.product(max_lrs, warmup_ratios, layer_sizes, batch_sizes, val_splits):
    print(f"\n===== Testing config =====")
    print(f"max_lr={max_lr}, warmup_ratio={warmup_ratio}, layer_size={layer_size}, batch_size={batch_size}, val_split={val_split}")

    warmup_epochs = round(total_epochs * warmup_ratio)

    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            return max_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max_lr * 0.5 * (1 + np.cos(np.pi * progress))

    custom_lr = keras.callbacks.LearningRateScheduler(lr_schedule)

    model = keras.models.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(layer_size, activation="relu"),
        keras.layers.Dense(layer_size, activation="relu"),
        keras.layers.Dense(layer_size, activation="relu"),
        keras.layers.Dense(layer_size, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy"]
    )

    checkpoint_path = f"best_model_{max_lr}_{warmup_ratio}_{layer_size}_{batch_size}_{val_split}.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=0
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=total_epochs,
        validation_split=val_split,
        callbacks=[custom_lr, checkpoint],
        verbose=0
    )

    model.load_weights(checkpoint_path)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test accuracy: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        best_config = (max_lr, warmup_ratio, layer_size, batch_size, val_split)
        print("New best found!")

print("\n===== BEST CONFIGURATION =====")
print(f"Accuracy: {best_acc:.4f}")
print(f"max_lr={best_config[0]}, warmup_ratio={best_config[1]}, layer_size={best_config[2]}, batch_size={best_config[3]}, val_split={best_config[4]}")
