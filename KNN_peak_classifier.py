######################################################################################
############################### - Import Libraries - #################################

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import convolutional_peak_detection as pd
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

    # split 70/30 for training and validation
    split_len = int(0.70*len(Index))
    train_class = Class[:split_len]
    train_waveforms = index_waveforms[:split_len]
    test_class = Class[split_len:]
    test_waveforms = index_waveforms[split_len:]

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

    # split 70/30 index and waveforms 70% train 30% test
    split_len = int(0.70*len(correct_predict_maxima_index))                # define list split point as 85%
    train_class = correct_predict_maxima_class[:split_len]                 # store first 85% in training class list
    train_waveforms = correct_predicted_index_waveforms[:split_len]   # store first 85% in training waveforms list
    test_class = correct_predict_maxima_class[split_len:]                  # store last 15% in test class list
    test_waveforms = correct_predicted_index_waveforms[split_len:]    # store last 15% in test waveform list

    # return the test and train waveforms and classes
    return train_waveforms, train_class, test_waveforms, test_class

######################################################################################
#################################### - PCA - #########################################
def data_PCA(train_waveforms, test_waveforms, num_PCA_components):
    # Select number of components to extract
    pca = PCA(n_components = num_PCA_components)

    # Fit to the training data
    pca.fit(train_waveforms)

    # Determine amount of variance explained by components
    print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))

    # Plot the explained variance
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Variance Explained by Extracted Componenents')
    plt.ylabel('Variance')
    plt.xlabel('Principal Components')
    plt.show()

    # Extract the principal components from the training data
    PCA_train_waveforms = pca.fit_transform(train_waveforms)
    # Transform the test data using the same components
    PCA_test_waveforms = pca.transform(test_waveforms)

    # return the test and train waveforms in the desired number of principal components
    return PCA_train_waveforms, PCA_test_waveforms

######################################################################################
#################################### - KNN - #########################################

def KNN_clasifier(PCA_train_waveforms, train_class, PCA_test_waveforms, test_class):
    # Create a KNN classification system with k = 5
    # Uses the p2 (Euclidean) norm
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(PCA_train_waveforms, train_class)

    # Feed the test data in the classifier to get the predictions
    test_class_predictions = knn.predict(PCA_test_waveforms)

    # return the predicted classifications
    return test_class_predictions

######################################################################################
############################ - Performance Metrics - #################################

# set to 1 to assess performance of CNN on classifying peaks from training dataset
test_KNN_performance = 1
if test_KNN_performance == 1:

    # prepare the ideal data for the CNN - using known peak locations so evaluation is of classification only
    #training_waveforms, training_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

    # prepare the non-ideal data for the CNN - this tests classification performance on correctly identified peak indexes at peak maxima
    train_waveforms, train_class, test_waveforms, test_class = non_ideal_data_preperation("training.mat")

    # find the PCA components of the train and test waveforms for KNN
    PCA_train_waveforms, PCA_test_waveforms = data_PCA(train_waveforms, test_waveforms, num_PCA_components=6)

    # Perform KNN to produce classes for input waveforms
    test_class_predictions = KNN_clasifier(PCA_train_waveforms, train_class, PCA_test_waveforms, test_class)

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