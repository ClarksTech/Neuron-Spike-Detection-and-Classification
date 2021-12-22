######################################################################################
############################### - Import Libraries - #################################

from sklearn.metrics import confusion_matrix
import numpy as np 
import matplotlib.pyplot as plt
import random

######################################################################################
################## - Identify correct and incorrect peak indexes - ###################
def get_peak_detection_correct_and_incorrect_index(known_index, Class, predicted_index):

    # sort know indexes into ascending order
    #known_index = sorted(known_index, reverse=False)
    known_index, Class = zip(*sorted(zip(known_index, Class)))
    known_index, Class = (list(i) for i in zip(*sorted(zip(known_index, Class))))

    # containers for the correct and incorrect peak indexes
    correct_predict_index = []
    incorrect_predict_index = []
    correct_predict_maxima_index = []
    correct_predict_maxima_class = []

    # loop through all predicted indexes
    for x in range(len(predicted_index)):
        peak_found = 0                          # initilise peak found
        peak = predicted_index[x]               # get current peak index
        # to allow for descrepancy between peak maxima and peak start step back a maximum fo 50 to see if peak exists
        for var in range((peak), (peak-50), -1):                # initial peak point may be within margine of error -50 to expected position
                if var in known_index:                          # check if potential peak start matches known index
                    correct_predict_index.append(var)           # if found save correct peak position to correct peak index
                    correct_predict_maxima_index.append(peak)   # if found save the prodicted peak maxima
                    position_found = known_index.index(var)     # get position of the peak within known index
                    correct_predict_maxima_class.append(Class[position_found])
                    known_index[position_found] = 0             # set position to 0 to avoid same point being identified as correct twice
                    peak_found = 1
                    break
        if peak_found == 0:                                     # if no peak was found
            incorrect_predict_index.append(peak)                # add to incorrect peak list             

    # return correct and incorrect peak index location lists
    return incorrect_predict_index, correct_predict_index, correct_predict_maxima_index, correct_predict_maxima_class


######################################################################################
################## - Peak Detection Confusion Matrix Parameters - ####################
def get_peak_detection_confusion_matrix_params(known_index, incorrect_predicted_index, correct_predicted_index, num_samples):

    # initialise counters
    true_negative = 0
    false_negative = 0
    true_positive = 0
    false_positive = 0

    # create two zero lists of length equal to number of samples
    known_zero_position = [0]*num_samples           # list for known zero positions
    predicted_zero_position = [0]*num_samples       # list for predicted zero positions

    # for the know index values - set zero list at corresponding position to 1 indicating peak
    for x in range(len(known_index)):
        known_zero_position[known_index[x]] = 1

    # for the correct predicted index values - set zero list at corresponding position to 1 indicating peak
    for x in range(len(correct_predicted_index)):
        predicted_zero_position[correct_predicted_index[x]] = 1

    # for the incorrect predicted index values - set zero list at corresponding position to 1 indicating peak
    for x in range(len(incorrect_predicted_index)):
        predicted_zero_position[incorrect_predicted_index[x]] = 1

    # step through each sample positoin and determin if peak detection is tp, fp, tn, fn
    for x in range(num_samples):
        # if the known and predicted sample are both 0 - it is true negative
        if known_zero_position[x] == 0 and predicted_zero_position[x] == 0:
            true_negative = true_negative + 1               # increment true negative counter
        # if the known sample is a peak but the predicted is not - it is false negative
        if known_zero_position[x] == 1 and predicted_zero_position[x] == 0:
            false_negative = false_negative + 1             # increment false negative counter
        # if the known sample is a peak and the predicted is a peak - it is true positive
        if known_zero_position[x] == 1 and predicted_zero_position[x] == 1:
            true_positive = true_positive + 1               # increment true positive counter
        # if the known sample is not a peak and the prediction is a peak - is a false positive
        if known_zero_position[x] == 0 and predicted_zero_position[x] == 1:
            false_positive = false_positive + 1             # increment false positive counter

    # verify each class summs to test set size - set to 1 to run verification
    verify = 0
    if verify == 1:
        # for each class
            print("Does Confusion Matrix sum to Samples? ", true_positive + false_positive + false_negative + true_negative == num_samples) # compare sum to total number of values

    # return tp, fp, tn, fn confusion matrix parameters
    return true_positive, false_positive, true_negative, false_negative



######################################################################################
#################### - Classifier Confusion Matrix parameters - ######################
def get_confusion_matrix_params(known_values, predicted_values, num_classes):

    # decrement classes to start count from 0
    predicted_values[:] = [Class - 1 for Class in predicted_values]
    known_values[:] = [Class - 1 for Class in known_values]

    # Generate the confustion matrix for all known and test results
    confusion_mtrx = confusion_matrix(known_values, predicted_values)

    # Get true positives - defined by the diagonal of the confusion matrix
    true_positive = np.diag(confusion_mtrx)

    # get false positives - sum of class column minus the diagonal value
    false_positive = []
    # for each class
    for i in range(num_classes):
        false_positive.append(sum(confusion_mtrx[:,i]) - confusion_mtrx[i,i])

    # get false negatives - sum of class row minus the diagonal value
    false_negative = []
    # for each class
    for i in range(num_classes):
        false_negative.append(sum(confusion_mtrx[i,:]) - confusion_mtrx[i,i])

    # get true negatives - delete all correctly identified values and sum remainder
    true_negative = []
    # for each class
    for i in range(num_classes):
        col_deleted_confusion_mtrx = np.delete(confusion_mtrx, i, 1)                        # delete correct column
        col_and_row_deleted_confusion_mtrx = np.delete(col_deleted_confusion_mtrx, i, 0)    # delete correct row
        true_negative.append(sum(sum(col_and_row_deleted_confusion_mtrx)))                  # sum confusion matrix

    # verify each class summs to test set size - set to 1 to run verification
    verify = 0
    if verify == 1:
        # for each class
        for i in range(num_classes):
            print("Does Confusion Matrix of class ", i+1, " sum to Samples? ", true_positive[i] + false_positive[i] + false_negative[i] + true_negative[i] == len(known_values)) # compare sum to total number of values

    # return the confusion matrix parameters
    return true_positive, true_negative, false_positive, false_negative


######################################################################################
######################## - Calculate Performance Metrics - ###########################
def gen_performance_metrics(true_positive, true_negative, false_positive, false_negative):

    # evaluate classification precision
    precision = true_positive/(true_positive+false_positive)

    # evaluate classification recall
    recall = true_positive/(true_positive+false_negative)

    # evaluate F1-Score
    F1_score = 2*((precision*recall)/(precision+recall))

    # return precision, recall, and F1 score performance metrics for given confusion matrix params
    return precision, recall, F1_score


######################################################################################
######## - Compare submission class waveforms to training class waveforms - ##########
def submission_confidence_comparison(training_waveforms, training_class, submission_waveforms, submission_class):

    # containers for training class waveforms
    training_class_1_waveforms = []
    training_class_2_waveforms = []
    training_class_3_waveforms = []
    training_class_4_waveforms = []
    training_class_5_waveforms = []

    # containers for submission class waveforms
    submission_class_1_waveforms = []
    submission_class_2_waveforms = []
    submission_class_3_waveforms = []
    submission_class_4_waveforms = []
    submission_class_5_waveforms = []

    # seperate training class into classes
    for y in range(len(training_class)):
        if training_class[y] == 1:
            training_class_1_waveforms.append(training_waveforms[y])    # add to class 1
        if training_class[y] == 2:
            training_class_2_waveforms.append(training_waveforms[y])    # add to class 2
        if training_class[y] == 3:
            training_class_3_waveforms.append(training_waveforms[y])    # add to class 3
        if training_class[y] == 4:
            training_class_4_waveforms.append(training_waveforms[y])    # add to class 4
        if training_class[y] == 5:
            training_class_5_waveforms.append(training_waveforms[y])    # add to class 5

    # seperate submission class into classes
    for y in range(len(submission_class)):
        if submission_class[y] == 1:
            submission_class_1_waveforms.append(submission_waveforms[y])    # add to class 1
        if submission_class[y] == 2:
            submission_class_2_waveforms.append(submission_waveforms[y])    # add to class 2
        if submission_class[y] == 3:
            submission_class_3_waveforms.append(submission_waveforms[y])    # add to class 3
        if submission_class[y] == 4:
            submission_class_4_waveforms.append(submission_waveforms[y])    # add to class 4
        if submission_class[y] == 5:
            submission_class_5_waveforms.append(submission_waveforms[y])    # add to class 5


    
    # plot class 1 training waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(training_class_1_waveforms))
        ax.plot(training_class_1_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Training Dataset Class 1 Waveform" )     # class and training or submission
    plt.show()  

    # plot class 1 submission waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(submission_class_1_waveforms))
        ax.plot(submission_class_1_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Submission Dataset Class 1 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 2 training waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(training_class_2_waveforms))
        ax.plot(training_class_2_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Training Dataset Class 2 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 2 submission waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(submission_class_2_waveforms))
        ax.plot(submission_class_2_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Submission Dataset Class 2 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 3 training waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(training_class_3_waveforms))
        ax.plot(training_class_3_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Training Dataset Class 3 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 3 submission waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(submission_class_3_waveforms))
        ax.plot(submission_class_3_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Submission Dataset Class 3 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 4 training waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(training_class_4_waveforms))
        ax.plot(training_class_4_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Training Dataset Class 4 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 4 submission waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(submission_class_4_waveforms))
        ax.plot(submission_class_4_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Submission Dataset Class 4 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 5 training waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(training_class_5_waveforms))
        ax.plot(training_class_5_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Training Dataset Class 5 Waveform" )     # class and training or submission
    plt.show() 

    # plot class 5 submission waveforms
    fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
    for i in range(10):                             # plot every waveform in peak detected waveform array
        random_waveform = random.randint(0, len(submission_class_5_waveforms))
        ax.plot(submission_class_5_waveforms[random_waveform])             # subplot
    plt.xlabel("Time Sample")                           # x axis is time sample
    plt.ylabel("Amplitude")                             # y axis is amplitude
    plt.title("Submission Dataset Class 5 Waveform" )     # class and training or submission
    plt.show() 

    return