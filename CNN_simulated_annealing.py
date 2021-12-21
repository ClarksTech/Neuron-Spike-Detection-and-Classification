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
############## - Non-ideal Data Extraction, Proccessing, Splitting - #################
def non_ideal_data_preperation(data_file):

    # the data, split to test and train sets
    data_stream, Index, Class, sample_rate = pd.load_training_dataset(data_file)    # load the ideal training data
    filtered_data_stream = pd.wavelet_filter_datastream(data_stream)                # filter the waveform

    # detect the peaks in the resulting datastream - store peak index and waveform in variables 0.4896, 0.42
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
############################# - Optimisation CNN - ###################################
# batch 128 - epoch early stop 100 ~50
def CNN_classifier(train_waveforms, train_class, test_waveforms, test_class, batch_size, epochs, optimisation_params):

    # optimiastion_params = [filter_size_1, filter_size_2, filter_size_3, filter_size_4, kernal_size_1, kernal_size_2, kernal_size_3, kernal_size_4, dropout_1, dropout_2]

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
            keras.Input(shape=input_shape),                                                                                     # set the CNN input to (50,1) as defined earlier
            layers.Conv1D(optimisation_params[0], padding="same", kernel_size=optimisation_params[4], activation="relu"),       # first layer is a convolutional layer - kernal size is large with fewer fiters to capture large initial features
            layers.MaxPooling1D(pool_size=2),                                                                                   # max pooling of window size 2 used to half dimensions, keeping trainable paramaters down
            layers.Dropout(optimisation_params[8]),                                                                             # dropout layer helps prevent overfitting by randomly dropping 20% of the nodes to the next layer
            layers.Conv1D(optimisation_params[1], padding="same", kernel_size=optimisation_params[5], activation="relu"),       # second convolution layer reduces the kernal size to focus on medium sized feature - as such number of filters increased
            layers.MaxPooling1D(pool_size=2),                                                                                   # max pooling of window size 2 used to half dimensions, keeping trainable paramaters down
            layers.Dropout(optimisation_params[9]),                                                                             # dropout layer helps prevent overfitting by randomly dropping nodes to the next layer
            layers.Conv1D(optimisation_params[2], padding="same", kernel_size=optimisation_params[6], activation="relu"),       # third convolution layer further reduces kernal size and increases number of filters - as such focuses on small features seperating classes
            layers.Conv1D(optimisation_params[3], padding="same", kernel_size=optimisation_params[7], activation="relu"),       # kernal size of 1 is fixed to decrease the number of features acting as channel-wide pooling for dimensionality reduction
            layers.Flatten(),                                                                                                   # flatten converts the multi dimension output of convolution to a single dimension array
            layers.Dense(num_classes, activation="softmax"),                                                                    # dense converts the dimensions of the output equal to that of the number of classes, soft max uses probabalistic values to determine which class the it is most likely belonging to
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

    # evaluate the model - display results
    score = model.evaluate(test_waveforms, test_class, batch_size =1)   # evaluate the model on test data to obtain final performance marks on new data
    print("Test loss:", score[0])           # print the test loss
    print("Test accuracy:", score[1])       # print the test accuracy

    # save the models weights for use later
    model.save_weights('Optimised_Training_data_CNN.h5')

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
############################# - Simulated Annealing - ################################

######################################################################################
#################### - Acceptance probability of new solution - ######################
def acceptance_probability(old_cost, new_cost, T):
    a = math.exp((old_cost-new_cost)/T)             # function of the difference between last cost and new cost with temperature - higher temperature accepts more leniently
    return a

######################################################################################
############################### - Annealing Loop - ###################################
def anneal(solution, target, alpha, iterations):
    old_cost, f1_scores = cost(solution, target)    # calculate cost of surrent solution
    cost_values = list()                            # container to store previous costs
    cost_values.append(old_cost)                    # add current solution cost to container
    T = 1.0             # temperature set to 1 to start (high)                           
    T_min = 0.000001    # minimum temperature to aneal to if no solution found
    break_flag = 0      # break flag set to 0 used to break out of anneal when solution found

    # loop through simulated anneal whilst temperature above the minimum
    while T > T_min:    
        i = 1           # initialise iteration counter

        # loop through all iterations at the temperature
        while i <= iterations:
            print("Iteration : " + str(i) + " cost : " + str(old_cost))     # print the iteration number and cost for visual progress checking
            print(solution)                                                 # print the solution being used for visual inspection of neighbour production
            new_solution = neighbour(solution)             # produce new neighbour solution
            new_cost, f1_scores = cost(new_solution, target)    # generate the new cost of the new neighbour
            ap = acceptance_probability(old_cost, new_cost, T)  # depending on the new cost and temperature calculate accpetance probability

            # if acceptance probablitity greater than a random number - accept the new solution
            if ap > random.random():
                solution = new_solution     # set current solution as new solution
                old_cost = new_cost         # set the old cost as the new cost

            i += 1                          # increase the iteration counter
            cost_values.append(old_cost)    # store the old cost in history container

            # check if the new accepted solution meets termination criteria
            if new_cost < 0.025:    # termination criteria set to cost less than 0.025
                break_flag = 1      # if meets set the break flag
                break               # break out of current temperature loop
        if break_flag == 1:
            break                   # break out of anneal loop

        T = T*alpha     # decrement the temperature by alpha

    # return the solution, thye solutions cost, history of costs, and the solutions F1-scores
    return solution, old_cost, cost_values, f1_scores

######################################################################################
################################ - Cost Function - ###################################
def cost(supply, demand):
    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    #training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # prepare the non-ideal data for the CNN - this tests classification performance on correctly identified peak indexes at peak maxima
    training_waveforms, training_class, test_waveforms, test_class = non_ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100, optimisation_params=supply)

    # get true positive, true negative, false positive, and false negative classifications for each class
    tp, tn, fp, fn = pm.get_confusion_matrix_params(test_class, test_class_predictions, 5)

    # get the F1-scores for each class
    f1_scores = []          # container to store f1-scores of each class
    for i in range(5):      # for each class
        precision, recall, F1_score = pm.gen_performance_metrics(tp[i], tn[i], fp[i], fn[i])   # get f1-score
        f1_scores.append(F1_score)          # store all f1-scores in list
        
    # cost is the RMS error between the demanded F1-scores and the current solution generated F1-scores
    delta = np.subtract(demand, f1_scores)  # get difference between demand and solution f1-score by subtraction
    delta2 = np.square(delta)               # square the differences to negate negatives
    ave = np.average(delta2)                # get the average of all such that a single value is obtained for cost
    dcost = math.sqrt(ave)                  # square root to obrain RMS error

    # return the RMS error cost and the F1-scores for all classes
    return dcost, f1_scores

######################################################################################
############################# - Neighbour Direction - ################################
def neighbour_direction( possible_filter_num, current_filter_index):
    # decide direction of neighbour & change
    max_up = (len(possible_filter_num)-1) - current_filter_index    # maximum neighbours above current hyperparameter position
    max_down = 0 + current_filter_index                             # maximum neighbours below current hyperparameter position

    # if no neighbours above, must go to neighbour below
    if max_up == 0:
        direction = 0    # change the new solution to the hyperparameter one index below (neighbour below value)

    # if no neighbours below, must go to neighbour above
    elif max_down == 0:
        direction = 1    # change the new solution to the hyperparameter one index above (neighbour above index)

    # if neighbours in both direction exist - randomly select direction
    if max_down != 0 and max_up != 0:
        direction = random.randint(0,1) # randomly select 0 or 1 

    # return the dirction of neighbour to select
    return direction


######################################################################################
######################## - Neighbour Generation Function - ###########################
def neighbour(solution):
    # randomly select the parameter to be change, generating a neighbour of current solution
    change = 7
    # positions 7 and 8 in the CNN hyper parameters are constant so do not accept their positions
    while change == 7 or change == 8:   # while the position selected is invalid
        change = random.randint(0,9)    # select a new random position
    print("change = ", change)          # print selected hyperparameter to be changed for visual confirmation
    new_solution = solution             # coppy current solution to new solution container for manipulation into neighbour
    
    # if hyperparapeter 0 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 0:

        # current hyperparapeter value
        possible_filter_num = [32, 64]          # possible values for hyperparameter 0 to obey architecture laws
        current_filter_num = solution[change]   # current hyperparameter value
        current_filter_index = possible_filter_num.index(current_filter_num)    # current position of hyperparameter in list of possible

        # decide direction of neighbour & change
        max_up = (len(possible_filter_num)-1) - current_filter_index    # maximum neighbours above current hyperparameter position
        max_down = 0 + current_filter_index                             # maximum neighbours below current hyperparameter position
        # if no neighbours above, must go to neighbour below
        if max_up == 0:
            new_solution[change] = possible_filter_num[current_filter_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # if no neighbours below, must go to neighbour above
        elif max_down == 0:
            new_solution[change] = possible_filter_num[current_filter_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)

        # check new solution meets design constraints - if not change required values so it does
        filter_number = [32, 64, 128, 256, 512] # all possible filter numbers
        # if the 2nd convolution layer is now not more than 32 filters larger - fix
        if (solution[change + 1] - new_solution[change]) < 32:
            new_solution[change + 1] = filter_number[filter_number.index(solution[change + 1]) + 1] # increment convolution filter count to next index so satisfies condition
        # if the 3rd convolution layer is now not more than 32 filters larger than 2nd - fix
        if (solution[change + 2] - new_solution[change + 1]) < 32:
            new_solution[change + 2] = filter_number[filter_number.index(solution[change + 2]) + 1] # increment convolution filter count to next index so satisfies condition
        # if the 4th convolution layer is now not more than 32 filters smaller than 3rd - fix
        if (new_solution[change + 2] - solution[change + 3]) < 32:
            new_solution[change + 3] = filter_number[filter_number.index(solution[change + 3]) - 1] # decrement convolution filter count to next index so satisfies condition

    # if hyperparapeter 1 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 1:

        # current hyperparapeter value
        possible_filter_num = [64, 128, 256]    # possible values for hyperparameter 1 to obey architecture laws
        current_filter_num = solution[change]   # current hyperparameter value
        current_filter_index = possible_filter_num.index(current_filter_num)    # current position of hyperparameter in list of possible

        # get direction of neighbour
        direction = neighbour_direction(possible_filter_num, current_filter_index)

        # get new neighbour in that direction
        if direction == 0:
            new_solution[change] = possible_filter_num[current_filter_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # otherwise if direction is 1 select neighbour above
        else:
            new_solution[change] = possible_filter_num[current_filter_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)

        # check new solution meets design constraints - if not change required values so it does
        filter_number = [32, 64, 128, 256, 512] # all possible filter numbers
        # if 1st convolution layer is now more than 32 less than current - fix 
        if solution[change] - solution[change - 1] < 32:
            new_solution[change - 1] = filter_number[filter_number.index(solution[change]) - 1] # decrement convolution filter count to next index so satisfies condition
        # if the 3rd convolution layer is now not more than 32 filters larger than 2nd - fix
        if (solution[change + 1] - new_solution[change]) < 32:
            new_solution[change + 1] = filter_number[filter_number.index(solution[change + 1]) + 1] # increment convolution filter count to next index so satisfies condition
        # if the 4th convolution layer is now not more than 32 filters smaller than 3rd - fix
        if (new_solution[change + 1] - solution[change + 2]) < 32:
            new_solution[change + 2] = filter_number[filter_number.index(solution[change + 2]) - 1] # decrement convolution filter count to next index so satisfies condition

    # if hyperparapeter 2 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 2:

        # current hyperparapeter value
        possible_filter_num = [128, 256, 512]   # possible values for hyperparameter 2 to obey architecture laws
        current_filter_num = solution[change]   # current hyperparameter value
        current_filter_index = possible_filter_num.index(current_filter_num)    # current position of hyperparameter in list of possible

        # get direction of neighbour
        direction = neighbour_direction(possible_filter_num, current_filter_index)

        # get new neighbour in that direction
        if direction == 0:
            new_solution[change] = possible_filter_num[current_filter_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # otherwise if direction is 1 select neighbour above
        else:
            new_solution[change] = possible_filter_num[current_filter_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)

        # check new solution meets design constraints - if not change required values so it does
        filter_number = [32, 64, 128, 256, 512] # all possible filter numbers
        # if previous convolution layer is now more than 32 less than current - fix 
        if solution[change] - solution[change - 1] < 32:
            new_solution[change - 1] = filter_number[filter_number.index(solution[change]) - 1] # decrement convolution filter count to next index so satisfies condition
        # if layer before previous convolution layer is now more than 32 less than current - fix 
        if solution[change - 1] - solution[change - 2] < 32:
            new_solution[change - 2] = filter_number[filter_number.index(solution[change -1 ]) - 1] # decrement convolution filter count to next index so satisfies condition
        # if the 4th convolution layer is now not more than 32 filters smaller than 3rd - fix
        if (new_solution[change] - solution[change + 1]) < 32:
            new_solution[change + 1] = filter_number[filter_number.index(solution[change + 1]) - 1] # decrement convolution filter count to next index so satisfies condition

    # if hyperparapeter 3 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 3:

        # possible values for hyperparameter 3 to obey architecture laws as must be at least 32 filters less than previous convolution
        if solution[change - 1] == 512:
            possible_filter_num = [32, 64, 128, 256]    # possible values when previous conv has 512 (maximum for that layer)
        elif solution[change-1] == 256:
            possible_filter_num = [32, 64, 128]         # possible values when previous conv has 256
        else:
            possible_filter_num = [32, 64]              # possible values when previous conv has 128 (minimum for that layer)
        
        current_filter_num = solution[change]                                   # current hyperparameter value
        current_filter_index = possible_filter_num.index(current_filter_num)    # current position of hyperparameter in list of possible

        # get direction of neighbour
        direction = neighbour_direction(possible_filter_num, current_filter_index)

        # get new neighbour in that direction
        if direction == 0:
            new_solution[change] = possible_filter_num[current_filter_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # otherwise if direction is 1 select neighbour above
        else:
            new_solution[change] = possible_filter_num[current_filter_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)
    

    # if hyperparapeter 4 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 4:
        # current hyperparapeter value
        possible_kernal_num = [3, 5, 7]         # possible values for hyperparameter 4 to obey architecture laws
        current_kernal_num = solution[change]   # current hyperparameter value
        current_kernal_index = possible_kernal_num.index(current_kernal_num)    # current position of hyperparameter in list of possible

        # get direction of neighbour
        direction = neighbour_direction(possible_kernal_num, current_kernal_index)

        # get new neighbour in that direction
        if direction == 0:
            new_solution[change] = possible_kernal_num[current_kernal_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # otherwise if direction is 1 select neighbour above
        else:
            new_solution[change] = possible_kernal_num[current_kernal_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)

        # check new solution meets design constraints - if not change required values so it does
        if solution[change + 1] > new_solution[change]:         # check if next kernal is larger than current - fix if is
            new_solution[change + 1] = new_solution[change]     # set next kernal to same size as current
        if solution[change + 2] > new_solution[change + 1]:     # check if last kernal is larger than second kernal
            new_solution[change + 2] = new_solution[change + 1] # set last kernal equal to second if it is

    # if hyperparapeter 5 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 5:
        # current hyperparapeter value
        possible_kernal_num = [3, 5, 7]         # possible values for hyperparameter 5 to obey architecture laws
        current_kernal_num = solution[change]   # current hyperparameter value
        current_kernal_index = possible_kernal_num.index(current_kernal_num)    # current position of hyperparameter in list of possible

        # get direction of neighbour
        direction = neighbour_direction(possible_kernal_num, current_kernal_index)

        # get new neighbour in that direction
        if direction == 0:
            new_solution[change] = possible_kernal_num[current_kernal_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # otherwise if direction is 1 select neighbour above
        else:
            new_solution[change] = possible_kernal_num[current_kernal_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)

        # check new solution meets design constraints - if not change required values so it does
        if solution[change + 1] > new_solution[change]:         # check if next kernal is larger than current - fix if is
            new_solution[change + 1] = new_solution[change]     # set next kernal to same size as current
        if solution[change - 1] < new_solution[change]:         # check if previous kernal is smaller than current kernal
            new_solution[change - 1] = new_solution[change]     # set last kernal equal to second if it is

    # if hyperparapeter 6 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 6:
        # current hyperparapeter value
        possible_kernal_num = [3, 5, 7]         # possible values for hyperparameter 6 to obey architecture laws
        current_kernal_num = solution[change]   # current hyperparameter value
        current_kernal_index = possible_kernal_num.index(current_kernal_num)    # current position of hyperparameter in list of possible

        # get direction of neighbour
        direction = neighbour_direction(possible_kernal_num, current_kernal_index)

        # get new neighbour in that direction
        if direction == 0:
            new_solution[change] = possible_kernal_num[current_kernal_index - 1]    # change the new solution to the hyperparameter one index below (neighbour below value)
        # otherwise if direction is 1 select neighbour above
        else:
            new_solution[change] = possible_kernal_num[current_kernal_index + 1]    # change the new solution to the hyperparameter one index above (neighbour above index)

        # check new solution meets design constraints - if not change required values so it does
        if solution[change - 1] < new_solution[change]:             # check if previous kernal is smaller
            new_solution[change - 1] = new_solution[change]         # if so set to same size
        if solution[change - 2] < new_solution[change - 1]:         # check if first kernal is smaller than previous
            new_solution[change - 2] = new_solution[change - 1]     # if so set to same size


    # if hyperparapeter 9 is being changed to neighbour state - must comply with laws of CNN model architecture
    if change == 9:

        less_than = solution[change] - 0.2  # maximum neighbours below current hyperparameter position
        more_than = 0.5 - solution[change]  # maximum neighbours above current hyperparameter position

        # determine direction of neighbour to use
        if less_than == 0:  # if no neighbours below
            new_solution[change] = (solution[change] + 0.01)    # increment hyperparameter by 0.01
        if more_than == 0:  # if no neighbours above
            new_solution[change] = (solution[change] - 0.01)    # decrement hyperparameter by 0.01
        # if neighbours above and below randomly generate direction of neighbour
        if less_than != 0 and more_than != 0:
            direction = random.randint(0,1) # randomly select 1 or 0 for direction
            # if direction is 0 find neighbour below
            if direction == 0:
                new_solution[change] = (solution[change] - 0.01)    # decrement hyperparameter by 0.01
            # otherwise find neighbour above
            elif direction == 1:
                new_solution[change] = (solution[change] + 0.01)    # increment hyperparameter by 0.01

    # return the neighbouring solution
    return new_solution


######################################################################################
################################ - Main code Run - ###################################

# set to 1 to obtain performance metrics for peak detection
test_SA_performance = 0
if test_SA_performance == 1:

    # best sol - 32 64 128 32 7 5 3 1 0.2 0.2]
    supply = [32, 64, 128, 64, 3, 3, 3, 1, 0.2, 0.35]   # CNN hyper parameters supplied to the SA for optimisation
    demand = [1, 1, 1, 1, 1]                            # the solution demand matrix (matrix of F1-score for each classificatio class where 1 = perfect)

    # print the supply hyperparameters
    print("Supply")
    print(supply)

    # print the demanded solution
    print("Demand")
    print(demand)

    # set SA parameters
    alpha = 0.80    # decrement temperature by 20% after each complete iteration cycle
    iterations = 5  # perfore 5 itterations at each temperature

    # run simmulated annealing
    final_solution, cost, cost_values, f1_scores = anneal(supply,demand,alpha,iterations) # get a final optimised solution from simulated annealing

    # print the demand, final solution, its cost, and f1-scores
    print("Demand: ", demand)
    print("Final Solution: ", final_solution)
    print("Final Cost: ", cost)
    print("Final F1-Scores: ", f1_scores)

    # plot the history of simulated annealing cost values for visual inspection
    plt.title("Error Function in Simmulated Annealing")
    plt.plot(cost_values)
    plt.grid(True)        
    plt.show()


######################################################################################
########################## - Test Performance Metrics - ##############################

# set to 1 to assess performance of CNN on classifying peaks from training dataset
test_CNN_performance = 0
if test_CNN_performance == 1:

    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    #training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # prepare the non-ideal data for the CNN - this tests classification performance on correctly identified peak indexes at peak maxima
    training_waveforms, training_class, test_waveforms, test_class = non_ideal_data_preperation("training.mat")

    # create and train the CNN, producng predicted classes for input waveforms
    test_class_predictions = CNN_classifier(training_waveforms, training_class, test_waveforms, test_class, batch_size=128, epochs=100, optimisation_params=final_solution )

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