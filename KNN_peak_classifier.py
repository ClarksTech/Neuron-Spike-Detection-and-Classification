######################################################################################
############################### - Import Libraries - #################################

import numpy
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
#################################### - PCA - #########################################
def data_PCA(train_waveforms, test_waveforms, num_PCA_components):
    # Select number of components to extract
    pca = PCA(n_components = num_PCA_components)

    # Fit to the training data
    pca.fit(train_waveforms)

    # Determine amount of variance explained by components
    print("Total Variance Explained: ", numpy.sum(pca.explained_variance_ratio_))

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

    return PCA_train_waveforms, PCA_test_waveforms

######################################################################################
#################################### - KNN - #########################################

def KNN_clasifier(PCA_train_waveforms, train_class, PCA_test_waveforms, test_class):
    # Create a KNN classification system with k = 5
    # Uses the p2 (Euclidean) norm
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(PCA_train_waveforms, train_class)

    # Feed the test data in the classifier to get the predictions
    pred = knn.predict(PCA_test_waveforms)

    # Check how many were correct
    scorecard = []

    for i, sample in enumerate(PCA_test_waveforms):
        # Check if the KNN classification was correct
        if round(pred[i]) == test_class[i]:
            scorecard.append(1)
        else:
            scorecard.append(0)
        pass

    # Calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print("Performance = ", (scorecard_array.sum() / scorecard_array.size) * 100, '%')

    test_class_predictions = pred

    return test_class_predictions

######################################################################################
############################ - Performance Metrics - #################################

# set to 1 to assess performance of CNN on classifying peaks from training dataset
test_KNN_performance = 1
if test_KNN_performance == 1:

    # prepare the ideal data for the KNN - using known peak locations so evaluation is of classification only
    train_waveforms, train_class, test_waveforms, test_class = ideal_data_preperation("training.mat")

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