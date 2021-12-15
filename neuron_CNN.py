import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import convolutional_peak_detection as pd


# the data, split to test and train sets
data_stream, Index, Class, sample_rate = pd.load_training_dataset("training.mat")
filtered_data_stream = pd.wavelet_filter_datastream(data_stream)

# get waveform windows for indexes
index_waveforms = []
for x in range(len(Index)):
    windowmin = Index[x]-25
    windowmax = Index[x]+25
    index_waveforms.append(filtered_data_stream[windowmin:windowmax])

# Decrement class numbers to start from 0
Class[:] = [Class_ - 1 for Class_ in Class]

# split 70/30 index and waveforms 85% train (including 15% for validation) 15% test
split_len = int(0.85*len(Index))
train_class = Class[:split_len]
train_waveforms = index_waveforms[:split_len]
test_class = Class[split_len:]
test_waveforms = index_waveforms[split_len:]

# correct dimensions for keras to shape (50,1)
train_waveforms = np.expand_dims(train_waveforms, -1)
#train_waveforms = np.expand_dims(train_waveforms, -1)
test_waveforms = np.expand_dims(test_waveforms, -1)
#test_waveforms = np.expand_dims(test_waveforms, -1)

print("train waveform shape: ", train_waveforms.shape)
print(train_waveforms.shape[0], " train samples")
print(test_waveforms.shape[0], " test samples")


######################################################################################
#################################### - CNN - #########################################

# Model / data parameters
num_classes = 5
input_shape = (50,1)

# convert to binary class matrices
train_class = keras.utils.to_categorical(train_class, num_classes)
test_class = keras.utils.to_categorical(test_class, num_classes)

# create the CNN model topology
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=4, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# display the final model topology
model.summary()

# decide batches and epoch number
batch_size = 10
epochs = 15

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train the model
model.fit(train_waveforms, train_class, batch_size=batch_size, epochs=epochs, validation_split=0.177)

# evaluate the model - display results
score = model.evaluate(test_waveforms, test_class, batch_size =1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
