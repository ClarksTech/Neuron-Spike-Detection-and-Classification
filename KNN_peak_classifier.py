# Numpy for useful maths
import numpy
# Sklearn contains some useful CI tools
# PCA
from sklearn.decomposition import PCA
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
# Matplotlib for plotting
import matplotlib.pyplot as plt
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

# split 70/30 for training and validation
split_len = int(0.70*len(Index))
train_class = Class[:split_len]
train_waveforms = index_waveforms[:split_len]
test_class = Class[split_len:]
test_waveforms = index_waveforms[split_len:]

# Separate labels from training data
train_data = train_waveforms
train_labels = train_class
test_data = test_waveforms
test_labels = test_class

# Select number of components to extract
pca = PCA(n_components = 6)

# Fit to the training data
pca.fit(train_data)

# Determine amount of variance explained by components
print("Total Variance Explained: ", numpy.sum(pca.explained_variance_ratio_))

# Plot the explained variance
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance Explained by Extracted Componenents')
plt.ylabel('Variance')
plt.xlabel('Principal Components')
plt.show()

# Extract the principal components from the training data
train_ext = pca.fit_transform(train_data)
# Transform the test data using the same components
test_ext = pca.transform(test_data)

# Create a KNN classification system with k = 5
# Uses the p2 (Euclidean) norm
knn = KNeighborsClassifier(n_neighbors=5, p=2)
knn.fit(train_ext, train_labels)

# Feed the test data in the classifier to get the predictions
pred = knn.predict(test_ext)

# Check how many were correct
scorecard = []

for i, sample in enumerate(test_data):
    # Check if the KNN classification was correct
    if round(pred[i]) == test_labels[i]:
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
# Calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("Performance = ", (scorecard_array.sum() / scorecard_array.size) * 100, '%')