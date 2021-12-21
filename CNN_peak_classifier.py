######################################################################################
############################### - Import Libraries - #################################
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import convolutional_peak_detection as pd
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
import performance_metrics as pm
import CNN_simulated_annealing as sa


######################################################################################
################ - ideal Data Extraction, Proccessing, Splitting - ###################
def ideal_data_preperation(data_file):

    # the data, split to test and train sets
    data_stream, Index, Class, sample_rate = pd.load_training_dataset(data_file)    # load the ideal training data
    filtered_data_stream = pd.wavelet_filter_datastream(data_stream)                # filter the waveform

    # get waveform windows for indexes for ideal indexes
    index_waveforms = []
    # extract window for every known ideal index
    for x in range(len(Index)):
        windowmin = Index[x]-25                                             # 25 points before index
        windowmax = Index[x]+25                                             # 25 points after index
        index_waveforms.append(filtered_data_stream[windowmin:windowmax])   # add waveform window to list

    # split 70/30 index and waveforms 85% train (including 15% for validation) 15% test
    split_len = int(0.85*len(Index))                # define list split point as 85%
    train_class = Class[:split_len]                 # store first 85% in training class list
    train_waveforms = index_waveforms[:split_len]   # store first 85% in training waveforms list
    test_class = Class[split_len:]                  # store last 15% in test class list
    test_waveforms = index_waveforms[split_len:]    # store last 15% in test waveform list

    # correct dimensions for keras to shape (50,1)
    train_waveforms = np.expand_dims(train_waveforms, -1)   # add another dimension to training waveform list as keras requires 2 dimensions
    test_waveforms = np.expand_dims(test_waveforms, -1)     # add another dimension to test waveform list as keras requires 2 dimensions

    # verify correct shapes - set shape display == 1
    shape_display = 0 
    if shape_display == 1:
        print("train waveform shape: ", train_waveforms.shape)  # print the shape of training waveforms to verify dimensionality
        print(train_waveforms.shape[0], " train samples")       # print the number of training samples
        print(test_waveforms.shape[0], " test samples")         # print the number of test samples

    # return the test and train waveforms and classes
    return train_waveforms, train_class, test_waveforms, test_class

######################################################################################
############## - Non-ideal Data Extraction, Proccessing, Splitting - #################
def non_ideal_data_preperation(data_file):

    # the data, split to test and train sets
    data_stream, Index, Class, sample_rate = pd.load_training_dataset(data_file)    # load the ideal training data
    filtered_data_stream = pd.wavelet_filter_datastream(data_stream)                # filter the waveform

    # detect the peaks in the resulting datastream - store peak index and waveform in variables
    peak_start_index, peak_found_waveform, peak_maxima_index = pd.convolution_peak_detection(filtered_data_stream, 0.42, 50)

    # print length of known indexes
    print("known number of peaks: ", len(Index))

    # print length of found indexes
    print("Detected number of peaks: ", len(peak_maxima_index))

    # get the correct and incorrect peak index lists
    incorrect_peak_index, correct_peak_index, correct_predict_maxima_index, correct_predict_maxima_class = pm.get_peak_detection_correct_and_incorrect_index(Index, Class, peak_maxima_index)

    # get waveform windows for indexes for ideal indexes
    correct_predicted_index_waveforms = []
    # extract window for every known ideal index
    for x in range(len(correct_predict_maxima_index)):
        windowmin = correct_predict_maxima_index[x]-25                                             # 25 points before index
        windowmax = correct_predict_maxima_index[x]+25                                             # 25 points after index
        correct_predicted_index_waveforms.append(filtered_data_stream[windowmin:windowmax])   # add waveform window to list

    # split 70/30 index and waveforms 85% train (including 15% for validation) 15% test
    split_len = int(0.85*len(Index))                # define list split point as 85%
    train_class = correct_predict_maxima_class[:split_len]                 # store first 85% in training class list
    train_waveforms = correct_predicted_index_waveforms[:split_len]   # store first 85% in training waveforms list
    test_class = correct_predict_maxima_class[split_len:]                  # store last 15% in test class list
    test_waveforms = correct_predicted_index_waveforms[split_len:]    # store last 15% in test waveform list

    # correct dimensions for keras to shape (50,1)
    train_waveforms = np.expand_dims(train_waveforms, -1)   # add another dimension to training waveform list as keras requires 2 dimensions
    test_waveforms = np.expand_dims(test_waveforms, -1)     # add another dimension to test waveform list as keras requires 2 dimensions

    # verify correct shapes - set shape display == 1
    shape_display = 0 
    if shape_display == 1:
        print("train waveform shape: ", train_waveforms.shape)  # print the shape of training waveforms to verify dimensionality
        print(train_waveforms.shape[0], " train samples")       # print the number of training samples
        print(test_waveforms.shape[0], " test samples")         # print the number of test samples

    # return the test and train waveforms and classes
    return train_waveforms, train_class, test_waveforms, test_class


######################################################################################
#################################### - CNN - #########################################
# batch 128 - epoch early stop 100 ~50
def CNN_classifier(train_waveforms, train_class, test_waveforms, test_class, batch_size, epochs):

    # decrement classes to start count from 0 as data manipulation later simplified
    train_class[:] = [Class - 1 for Class in train_class]   # decrement every value in training class by 1
    test_class[:] = [Class - 1 for Class in test_class]     # decrement every value in test class by 1
    temp_test_class = test_class                            # temporary variable to hold initial class to restore later

    # Model / data parameters for keras
    num_classes = 5         # 5 classes of neuron spike
    input_shape = (50,1)    # data had been prossesed into 2D shape (50,1)

    # convert to binary class matrices of length number of classes to match output of CNN, where position of 1 represents class known as 'one-hot-encoding'
    train_class = keras.utils.to_categorical(train_class, num_classes)  # to_catagorical converts train class number to binary class matrices
    test_class = keras.utils.to_categorical(test_class, num_classes)    # to_catagorical tonverts test class numbers to binary class matrices

    # create the CNN model topology using keras
    model = keras.Sequential(                                                       # CNN model is sequential so define as such
        [
            # Define CNN model layers - all convolutions use relu activation as litrature largely regards as the best
            # padding for convolution has also been set to same - meaning when the window being convolved falls outside 
            # origional dimensions, the excess is set to the same as the last value within dimension
            keras.Input(shape=input_shape),                                         # set the CNN input to (50,1) as defined earlier
            layers.Conv1D(32, padding="same", kernel_size=3, activation="relu"),    # first layer is a convolutional layer - kernal size 3 is with fewer fiters to capture large initial features
            layers.MaxPooling1D(pool_size=2),                                       # max pooling of window size 2 used to half dimensions, keeping trainable paramaters down
            layers.Dropout(0.2),                                                    # dropout layer helps prevent overfitting by randomly dropping 20% of the nodes to the next layer
            layers.Conv1D(64, padding="same", kernel_size=3, activation="relu"),    # second convolution layer increases number of filters to capture more features
            layers.MaxPooling1D(pool_size=2),                                       # max pooling of window size 2 used to half dimensions, keeping trainable paramaters down
            layers.Dropout(0.35),                                                   # dropout layer helps prevent overfitting by randomly dropping 0.35% of the nodes to the next layer
            layers.Conv1D(128, padding="same", kernel_size=3, activation="relu"),   # third convolution layer further increases number of filters capturing small features
            layers.Conv1D(64, padding="same", kernel_size=1, activation="relu"),    # kernal size of 1 is used to decrease the number of features from 128 to 64 acting as channel-wide pooling for dimensionality reduction
            layers.Flatten(),                                                       # flatten converts the multi dimension output of convolution to a single dimension array
            layers.Dense(num_classes, activation="softmax"),                        # dense converts the dimensions of the output equal to that of the number of classes, soft max uses probabalistic values to determine which class the it is most likely belonging to
        ]
    )

    # display the final model topology
    model.summary()

    # compile the model
    # loss function set to categorical crossentropy as this uses 'one-hot-encoding' to calculate crossentropy between label and prediction
    # optimizer set to adam as an adaptive learning rate to best optimise the learning rate depending on stochastic gradient descent
    # metrics set to show precision and recall for each epoch as this is in-line with performance metric calculations
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["Precision", "Recall"])

    # simple early stopping based on the loss value - patientce set to 10 meaning 10 iterations must break to cause stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # train the model - validation split set to 0.177 as this is 15% of whole dataset - callbacks enabled for early stopping preventing overfitting
    history = model.fit(train_waveforms, train_class, batch_size=batch_size, epochs=epochs, validation_split=0.177, callbacks=[es])

    # plot training history and loss to observe if overfitting occured (lines will begine to diverge as ovrfitting occurs)
    pyplot.plot(history.history['loss'], label='train')     # plot training loss
    pyplot.plot(history.history['val_loss'], label='test')  # plot validation loss
    pyplot.legend()
    pyplot.show()

    # evaluate the model - display results
    score = model.evaluate(test_waveforms, test_class, batch_size =1)   # evaluate the model on test data to obtain final performance marks on new data
    print("Test loss:", score[0])           # print the test loss
    print("Test accuracy:", score[1])       # print the test accuracy

    # save the models weights for use later
    model.save_weights('Training_data_CNN.h5')

    # make predictions for test set
    test_class_predictions = model.predict(test_waveforms)              # create predictions for class of each test waveform
    test_class_predictions = np.argmax(test_class_predictions, axis=1)  # convert back from 'one-hot-encoding' by taking the largest value in the binary class matrices

    # increment class numbers to start from 1 again
    test_class_predictions[:] = [pred + 1 for pred in test_class_predictions]   # increment the predicted classes so class count starts from 1 not zero
    temp_test_class[:] = [Class + 1 for Class in temp_test_class]               # increment the temporary rest class so classes count from 1 not 0
    test_class = temp_test_class                                                # assign temp class back to test_class variable for later use

    # return the models predictions for test data
    return test_class_predictions

######################################################################################
########################## - Test Performance Metrics - ##############################

# set to 1 to assess performance of CNN on classifying peaks from training dataset
test_CNN_performance = 1
if test_CNN_performance == 1:

    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    #training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # prepare the non-ideal data for the CNN - this tests classification performance on correctly identified peak indexes at peak maxima
    training_waveforms, training_class, test_waveforms, test_class = non_ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100)

    # get true positive, true negative, false positive, and false negative classifications for each class
    tp, tn, fp, fn = pm.get_confusion_matrix_params(test_class, test_class_predictions, 5)
    print("True Positives = ", tp)      # print true positive array
    print("True Negative = ", tn)       # print true negative array
    print("False Positives = ", fp)     # print false positive array
    print("False Negatives = ", fn)     # print false negative array

    # Print the performance metrics Precision, Recall, and F1-Score for each class (closer to 1 is better)
    f1_score = []   # container for later comparison
    precision = []  # container for later comparison
    recall = []     # container for later comparison
    for i in range(5):          # for each class
        class_number = i+1      # class number + 1 as count starts at 0 but class index starts at 1
        Precision, Recall, F1_score = pm.gen_performance_metrics(tp[i], tn[i], fp[i], fn[i])                                        # generate the performance metrics
        f1_score.append(F1_score)         # store for comparison
        precision.append(Precision)       # store for comparison
        recall.append(Recall)             # store for comparison
        print("Class ", class_number, " Perfromance Metrics: Precision=", Precision, " Recall=", Recall, " F1-score=", F1_score)    # print performance etrics for class

######################################################################################
############################ - Call SA Optimisation - ################################

    # supply SA with non-optimised hyper parameters
    supply = [32, 64, 128, 64, 3, 3, 3, 1, 0.2, 0.35]   # CNN hyper parameters supplied to the SA for optimisation
    demand = [1, 1, 1, 1, 1]                            # the solution demand matrix (matrix of F1-score for each classificatio class where 1 = perfect)

    # set SA parameters
    alpha = 0.80    # decrement temperature by 20% after each complete iteration cycle
    iterations = 5  # perfore 5 itterations at each temperature

    # run simmulated annealing
    final_solution, cost, cost_values, f1_scores = sa.anneal(supply,demand,alpha,iterations) # get a final optimised solution from simulated annealing

    # print the demand, final solution, its cost, and f1-scores
    print("Demand: ", demand)
    print("Final Solution: ", final_solution)
    print("Final Cost: ", cost)
    print("Final F1-Scores: ", f1_scores)

    # plot the history of simulated annealing cost values for visual inspection
    pyplot.title("Error Function in Simmulated Annealing")
    pyplot.plot(cost_values)
    pyplot.grid(True)        
    pyplot.show()

######################################################################################
###################### - Optimised CNN Performance metrics - #########################

    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    #training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # prepare the non-ideal data for the CNN - this tests classification performance on correctly identified peak indexes at peak maxima
    training_waveforms, training_class, test_waveforms, test_class = non_ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = sa.CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100, optimisation_params=final_solution )

    # get true positive, true negative, false positive, and false negative classifications for each class
    sa_tp, sa_tn, sa_fp, sa_fn = pm.get_confusion_matrix_params(test_class, test_class_predictions, 5)
    print("True Positives = ", sa_tp)      # print true positive array
    print("True Negative = ", sa_tn)       # print true negative array
    print("False Positives = ", sa_fp)     # print false positive array
    print("False Negatives = ", sa_fn)     # print false negative array

    # Print the performance metrics Precision, Recall, and F1-Score for each class (closer to 1 is better)
    sa_f1_score = []    # container for comparison
    sa_precision = []   # container for comparison
    sa_recall = []      # container for comparison
    for i in range(5):          # for each class
        class_number = i+1      # class number + 1 as count starts at 0 but class index starts at 1
        sa_Precision, sa_Recall, sa_F1_score = pm.gen_performance_metrics(sa_tp[i], sa_tn[i], sa_fp[i], sa_fn[i])                       # generate the performance metrics
        sa_f1_score.append(sa_F1_score)         # store for comparison
        sa_precision.append(sa_Precision)       # store for comparison
        sa_recall.append(sa_Recall)             # store for comparison
        print("Class ", class_number, " Perfromance Metrics: Precision=", sa_Precision, " Recall=", sa_Recall, " F1-score=", sa_F1_score)        # print performance etrics for class


######################################################################################
########## - Display improvement of performance metrics by optimisation - ############

for i in range(5):
    # create list of class performance metrics pre and post optimisation
    non_optimised = [precision[i], recall[i], f1_score[i]]
    optimised = [sa_precision[i], sa_recall[i], sa_f1_score[i]]

    # create list of metric titles for the comparison
    metric = ["Precision", "Recall", "F1-score"]

    # set the number of indicies on bar graph and their width
    indices = np.arange(3)  # 5 performance metric indicies
    width = 0.20            # width of 0.2

    # Plot the bars for the non-optimised CNN performance metrics
    pyplot.bar(indices, non_optimised, width=width)

    # Offsetting by width plot the bars for the optimised CNN performance metrics
    pyplot.bar(indices + width, optimised, width=width)

    # add the metric titles
    pyplot.xticks(ticks=indices, labels=metric)

    # add bar chart labels
    pyplot.xlabel("Performance Metric")     # x axis is performance metric
    pyplot.ylabel("Performance")      # y axis is performance in range 0 to 1 (1 being best)
    Class = i+1
    pyplot.title("Class %i Performance metrics before and after SA" %Class)   # title the bar graph so the class it is 

    pyplot.show()

