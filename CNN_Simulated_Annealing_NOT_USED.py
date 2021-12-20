######################################################################################
############################### - Import Libraries - #################################
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import convolutional_peak_detection as pd
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
import performance_metrics as pm


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
#################################### - CNN - #########################################
# batch 128 - epoch early stop 100 ~50
def CNN_classifier(train_waveforms, train_class, test_waveforms, test_class, batch_size, epochs, optimisation_params):

    # optimiastion_params = [filter_size_1, filter_size_2, filter_size_3, filter_size_4, kernal_size_1, kernal_size_2, kernal_size_3, kernal_size_4, dropout_1, dropout_2]
    # initial = [32, 64, 128, 32, 7, 5, 3, 1, 0.2, 0.2]

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
            layers.Conv1D(optimisation_params[0], padding="same", kernel_size=optimisation_params[4], activation="relu"),    # first layer is a convolutional layer - kernal size 7 is large with fewer fiters to capture large initial features
            layers.MaxPooling1D(pool_size=2),                                       # max pooling of window size 2 used to half dimensions, keeping trainable paramaters down
            layers.Dropout(optimisation_params[8]),                                                    # dropout layer helps prevent overfitting by randomly dropping 20% of the nodes to the next layer
            layers.Conv1D(optimisation_params[1], padding="same", kernel_size=optimisation_params[5], activation="relu"),    # second convolution layer reduces the kernal size to focus on medium sized feature - as such number of filters increased
            layers.MaxPooling1D(pool_size=2),                                       # max pooling of window size 2 used to half dimensions, keeping trainable paramaters down
            layers.Dropout(optimisation_params[9]),                                                    # dropout layer helps prevent overfitting by randomly dropping 20% of the nodes to the next layer
            layers.Conv1D(optimisation_params[2], padding="same", kernel_size=optimisation_params[6], activation="relu"),   # third convolution layer further reduces kernal size and increases number of filters - as such focuses on small features seperating classes
            layers.Conv1D(optimisation_params[3], padding="same", kernel_size=optimisation_params[7], activation="relu"),    # kernal size of 1 is used to decrease the number of features from 128 to 32 acting as channel-wide pooling for dimensionality reduction
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
    #pyplot.plot(history.history['loss'], label='train')     # plot training loss
    #pyplot.plot(history.history['val_loss'], label='test')  # plot validation loss
    #pyplot.legend()
    #pyplot.show()

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
test_CNN_performance = 0
if test_CNN_performance == 1:

    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100, optimisation_params=[32, 64, 128, 32, 7, 5, 3, 1, 0.2, 0.2])

    # get true positive, true negative, false positive, and false negative classifications for each class
    tp, tn, fp, fn = pm.get_confusion_matrix_params(test_class, test_class_predictions, 5)
    print("True Positives = ", tp)      # print true positive array
    print("True Negative = ", tn)       # print true negative array
    print("False Positives = ", fp)     # print false positive array
    print("False Negatives = ", fn)     # print false negative array

    # Print the performance metrics Precision, Recall, and F1-Score for each class (closer to 1 is better)
    for i in range(5):          # for each class
        class_number = i+1      # class number + 1 as count starts at 0 but class index starts at 1
        precision, recall, F1_score = pm.gen_performance_metrics(tp[i], tn[i], fp[i], fn[i])                                        # generate the performance metrics
        print("Class ", class_number, " Perfromance Metrics: Precision=", precision, " Recall=", recall, " F1-score=", F1_score)    # print performance etrics for class



#acceptance probability function
def acceptance_probability(old_cost, new_cost, T):
    a = math.exp((old_cost-new_cost)/T)
    return a

#annealing function
def anneal(solution, target, alpha, iterations, var):
    old_cost = cost(solution, target)
    cost_values = list()
    cost_values.append(old_cost)
    T = 1.0
    T_min = 0.000001
    break_flag = 0
    while T > T_min:
        i = 1
        while i <= iterations:
            print("Iteration : " + str(i) + " cost : " + str(old_cost))
            print(solution)
            new_solution = neighbour(solution, var)
            new_cost = cost(new_solution, target)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random.random():
                solution = new_solution
                old_cost = new_cost
            i += 1
            cost_values.append(old_cost)
            if new_cost < 0.025:
                break_flag = 1
                break
        if break_flag == 1:
            break
        T = T*alpha
    return solution, old_cost, cost_values

#cost function
def cost(supply, demand):
    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100, optimisation_params=supply)

    # get true positive, true negative, false positive, and false negative classifications for each class
    tp, tn, fp, fn = pm.get_confusion_matrix_params(test_class, test_class_predictions, 5)

    # Print the performance metrics Precision, Recall, and F1-Score for each class (closer to 1 is better)
    f1_scores = []
    for i in range(5):                      # for each class
        precision, recall, F1_score = pm.gen_performance_metrics(tp[i], tn[i], fp[i], fn[i])   
        f1_scores.append(F1_score)          # generate the performance metrics
        
    delta = np.subtract(demand, f1_scores)
    delta2 = np.square(delta)
    ave = np.average(delta2)
    dcost = math.sqrt(ave)

    return dcost

#neighbour function
def neighbour(solution, d):
    # select param to be altered
    delta = random.randint(0,9)
    new_solution = solution

    # num filter selection
    if delta <= 3:
        possible_filters = [8, 16, 32, 64, 128, 256, 512]
        index = possible_filters.index(solution[delta])
        less_than = 0 + index
        more_than = 6 - index
        if less_than == 0:
            direction = 1
        if more_than == 0:
            direction = 0
        if less_than != 0 and more_than != 0:
            direction = random.randint(0,1)
        if direction == 0:
            new_solution[delta] = possible_filters[index - 1]
        else:
            new_solution[delta] = possible_filters[index + 1]

    # kernal size selection
    if delta > 3 and delta < 7:
        possible_kernals = [3, 5, 7]
        index = possible_kernals.index(solution[delta])
        less_than = 0 + index
        more_than = 2 - index
        if less_than == 0:
            direction = 1
        if more_than == 0:
            direction = 0
        if less_than != 0 and more_than != 0:
            direction = random.randint(0,1)
        if direction == 0:
            new_solution[delta] = possible_kernals[index - 1]
        else:
            new_solution[delta] = possible_kernals[index + 1]
        
    # dropout selection
    if delta > 8:
        less_than = solution[delta] - 0.2
        more_than = 0.5 - solution[delta]
        if less_than == 0:
            direction = 1
        if more_than == 0:
            direction = 0
        if less_than != 0 and more_than != 0:
            direction = random.randint(0,1)
        if direction == 0:
            new_solution[delta] = solution[delta] - 0.01
        else:
            new_solution[delta] = solution[delta] + 0.01

    return new_solution

supply = [64, 64, 64, 32, 5, 5, 5, 1, 0.2, 0.35]
demand = [1, 1, 1, 1, 1]


print("Supply")
print(supply)

print("Demand")
print(demand)

rms = cost(supply,demand)
print("RMS = " + str(rms))

alpha = 0.95
iterations = 200
var = 0.01

final_solution, cost, cost_values = anneal(supply,demand,alpha,iterations,var)

print("Demand: ", demand)
print("Final Solution: ", final_solution)
print("Final Cost", cost)

plt.subplot(232)
plt.title("Error Function in Simmulated Annealing")
plt.plot(cost_values)
plt.grid(True)

#plt.subplot(234)
#plt.title("Initial Supply Matrix")
#plt.imshow(supply, cmap='hot')

#plt.subplot(235)
#plt.title("Optimised Supply Matrix")
#plt.imshow(final_solution, cmap='hot')

#plt.subplot(236)
#plt.title("Demand Matrix")
#plt.imshow(demand, cmap='hot')
        
plt.show()