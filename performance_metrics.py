from sklearn.metrics import confusion_matrix
import numpy as np 

def get_confusion_matrix_params(known_values, predicted_values, num_classes):

    # Generate the confustion matrix for all known and test results
    confusion_mtrx = confusion_matrix(known_values, predicted_values)

    # Get true positives - defined by the diagonal of the confusion matrix
    true_positive = np.diag(confusion_mtrx)

    # get false positives - sum of class column minus the diagonal value
    false_positive = []
    for i in range(num_classes):
        false_positive.append(sum(confusion_mtrx[:,i]) - confusion_mtrx[i,i])

    # get false negatives - sum of class row minus the diagonal value
    false_negative = []
    for i in range(num_classes):
        false_negative.append(sum(confusion_mtrx[i,:]) - confusion_mtrx[i,i])

    # get true negatives - delete all correctly identified values and sum remainder
    true_negative = []
    for i in range(num_classes):
        col_deleted_confusion_mtrx = np.delete(confusion_mtrx, i, 1)
        col_and_row_deleted_confusion_mtrx = np.delete(col_deleted_confusion_mtrx, i, 0)
        true_negative.append(sum(sum(col_and_row_deleted_confusion_mtrx)))

    # verify each class summs to test set size - set to 1 to run verification
    verify = 0
    if verify == 1:
        for i in range(num_classes):
            print(true_positive[i] + false_positive[i] + false_negative[i] + true_negative[i] == len(known_values))

    return true_positive, true_negative, false_positive, false_negative

    