######################################################################################
############################### - Import Libraries - #################################
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import convolutional_peak_detection as pd
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
import performance_metrics as pm


######################################################################################
################### - Data Extraction, Proccessing, Splitting - ######################
def ideal_data_preperation(data_file):

    # the data, split to test and train sets
    data_stream, Index, Class, sample_rate = pd.load_training_dataset(data_file)
    filtered_data_stream = pd.wavelet_filter_datastream(data_stream)

    # get waveform windows for indexes
    index_waveforms = []
    for x in range(len(Index)):
        windowmin = Index[x]-25
        windowmax = Index[x]+25
        index_waveforms.append(filtered_data_stream[windowmin:windowmax])

    # split 70/30 index and waveforms 85% train (including 15% for validation) 15% test
    split_len = int(0.85*len(Index))
    train_class = Class[:split_len]
    train_waveforms = index_waveforms[:split_len]
    test_class = Class[split_len:]
    test_waveforms = index_waveforms[split_len:]

    # correct dimensions for keras to shape (50,1)
    train_waveforms = np.expand_dims(train_waveforms, -1)
    test_waveforms = np.expand_dims(test_waveforms, -1)

    # verify correct shapes - set shape display == 1
    shape_display = 0 
    if shape_display == 1:
        print("train waveform shape: ", train_waveforms.shape)
        print(train_waveforms.shape[0], " train samples")
        print(test_waveforms.shape[0], " test samples")

    # return the test and train waveforms and classes
    return train_waveforms, train_class, test_waveforms, test_class


######################################################################################
#################################### - CNN - #########################################
# batch 128 - epoch early stop 100 ~50
def CNN_classifier(train_waveforms, train_class, test_waveforms, test_class, batch_size, epochs):

    # decrement classes to start count from 0
    train_class[:] = [Class - 1 for Class in train_class]
    test_class[:] = [Class - 1 for Class in test_class]
    temp_test_class = test_class

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
    test_class_predictions = model.predict(test_waveforms)
    test_class_predictions = np.argmax(test_class_predictions, axis=1)

    # increment class numbers to start from 1 again
    test_class_predictions[:] = [pred + 1 for pred in test_class_predictions]
    temp_test_class[:] = [Class + 1 for Class in temp_test_class]
    test_class = temp_test_class

    # return the models predictions for test data
    return test_class_predictions

######################################################################################
############################# - Performance Metrics - ################################


test_CNN_performance = 1
if test_CNN_performance == 1:

    # prepare the data for the CNN
    training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100)

    # get true positive, true negative, false positive, and false negative classifications for each class
    tp, tn, fp, fn = pm.get_confusion_matrix_params(test_class, test_class_predictions, 5)
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

    # evaluate F1-Score
    f1 = 2*((precision*recall)/(precision+recall))
    print("F1 - Score = ", f1)

