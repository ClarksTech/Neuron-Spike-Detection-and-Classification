import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, layers
import convolutional_peak_detection as pd
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
import performance_metrics as pm


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
test_waveforms = np.expand_dims(test_waveforms, -1)


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
        layers.Conv1D(32, padding="same", kernel_size=7, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Conv1D(64, padding="same", kernel_size=5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        layers.Conv1D(128, padding="same", kernel_size=3, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        layers.Conv1D(256, padding="same", kernel_size=1, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# display the final model topology
model.summary()

# decide batches and epoch number
batch_size = 128
epochs = 100

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# train the model
history = model.fit(train_waveforms, train_class, batch_size=batch_size, epochs=epochs, validation_split=0.177, callbacks=[es])

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# evaluate the model - display results
score = model.evaluate(test_waveforms, test_class, batch_size =1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# save the models weights for use later
model.save_weights('Training_data_CNN.h5')

# make predictions for test set
training_test_predictions = model.predict(test_waveforms)
training_test_predictions = np.argmax(training_test_predictions, axis=1)
# increment class numbers to start from 1 again
training_test_predictions[:] = [pred + 1 for pred in training_test_predictions]
test_class = Class[split_len:]
test_class[:] = [Class + 1 for Class in test_class]
correct_count = 0
for x in range(len(test_class)):
    if test_class[x] == training_test_predictions[x]:
        correct_count = correct_count + 1
    
performance = correct_count/len(test_class)
print("CNN Predicts the correct class of neuron ", performance, "%")

########################################################################
###################### - Performance Metrics - #########################

# get true positive, true negative, false positive, and false negative classifications for each class
tp, tn, fp, fn = pm.get_confusion_matrix_params(test_class, training_test_predictions, 5)
print("True Positives = ", tp)
print("True Negative = ", tn)
print("False Positives = ", fp)
print("False Negatives = ", fn)

# Sum all classes for an overall metric
TP = sum(tp)
TN = sum(tn)
FP = sum(fp)
FN = sum(fn)

# evaluate overall classification precision
precision = TP/(TP+FP)
print("Overall Precision = ", precision)

# evaluate overall classification recall
recall = TP/(TP+FN)
print("Overall Recall = ", recall)

# evaluate overall classification accuracy
accuracy = (TP+TN)/(TP+FP+FN+TN)
print("Overall Accuracy = ", accuracy)