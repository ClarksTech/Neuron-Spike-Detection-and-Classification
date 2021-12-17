import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pylab


##################################################################
################# - Import the training dataset - ################
def load_training_dataset(dataset):
    # import from matlab file type to store in arrays
    dataset_all = spio.loadmat(dataset, squeeze_me=True)
    Data_stream = dataset_all['d']                          # array for whole data stream
    sample_rate = 25000                                     # known sample rate of data (Hz)
    Index = dataset_all['Index']                            # array for index of known spikes
    Class = dataset_all['Class']                            # array for class of known spikes
                                
    # return all arrays containing training dataset values and known sample rate
    return Data_stream, Index, Class, sample_rate

##################################################################
############### - Import the submission dataset - ################
def load_submission_dataset(dataset):
    # import from matlab file type to store in arrays
    dataset_all = spio.loadmat(dataset, squeeze_me=True)
    Data_stream = dataset_all['d']                          # array for whole data stream
    sample_rate = 25000                                     # known sample rate of data (Hz)

    # return all arrays containing training dataset values and known sample rate
    return Data_stream, sample_rate

##################################################################
########## - Perform Wavelet Filtering on Datastream - ###########
def wavelet_filter_datastream(datastream):
    # using Daubechies(4) wavelet filter
    wavelet = "db4"    
    maxlevel = 6

    # Initialize array for new filtered data
    filtered_data_stream = np.empty((1, len(datastream)))      
    	
	# Decompose the signal
    decomposed = pywt.wavedec(datastream, wavelet, level=maxlevel)
	# Destroy the approximation coefficients
    decomposed[0][:] = 0
	# Reconstruct the signal and save it
    filtered_data_stream = pywt.waverec(decomposed, wavelet)
       
    return filtered_data_stream

##################################################################
############ - Perform Peak Detection on Datastream - ############
def peak_detection(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return signals, avgFilter, stdFilter
    

##################################################################
########## - Perform Peak window index and extraction - ##########
def peak_index_detection(processed_datastream, filtered_datastream):
    peak_found_index = []
    peak_found_waveform = []

    for x in range(0,len(processed_datastream)):
        if processed_datastream[x] == 1:
            ones_count = 0
            for point in range(1,11):
                if processed_datastream[(x-point)] == 1:
                    ones_count = ones_count + 1
            if ones_count == 0:
                peak_found_index.append(x)
                peak_found_waveform.append(filtered_datastream[(x-50):(x+50)])


    return peak_found_index, peak_found_waveform

##############################################################################################
################################## - Main Code Run - #########################################

# load the matlab data into variables
datastream, Index, Class, sample_rate = load_training_dataset("training.mat")
#datastream, sample_rate = load_submission_dataset("submission.mat")

# filter the datastream using level 6 wavelet filter
filtered_data_stream = wavelet_filter_datastream(datastream)
plt.plot(filtered_data_stream)
plt.show()

# detect the peaks in the resulting datastream - store peak index and waveform in variables 0.4896, 0.42
signal, avgfilter, stdfilter = peak_detection(filtered_data_stream, 50, 3.3, 1)
peak_found_index, peak_found_waveform = peak_index_detection(signal, filtered_data_stream)

plt.plot(signal)
plt.show()
# sort know indexes into ascending order
Index_sorted = sorted(Index, reverse=False)

# print length of known indexes
print("known number of peaks: ", len(Index_sorted))

# print length of found indexes
print("Detected number of peaks: ", len(peak_found_index))

# plot all found waveforms of peaks on same axis to verify detection is finding peaks
fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
for i in range(len(peak_found_waveform)):       # plot every waveform in peak detected waveform array
    ax.plot(peak_found_waveform[i])             # subplot
plt.show()                                      # show the plot of peak waveforms

# check if peak index found matches known peak index
correct_index = []
for x in range(len(peak_found_index)):
    peak_start = peak_found_index[x]
    # create range for initial peak start variance from maxima
    for var in range((peak_start+50), (peak_start-50), -1):         # initial peak point may be within margine of error +-10 to expected position
            if var in Index_sorted:                                 # check if potential peak start matches known index
                correct_index.append(peak_start)                    # if found increment correct index counter
                position_found = Index_sorted.index(var)
                #Index_sorted[position_found] = 0                    # set to 0 to avoid same point being identified as correct twice
                break
# display the number of correctly detected peaks and peake detection performance
print("Number of detected peaks matching known peaks: ", len(correct_index), " Peak detection performance = ", len(correct_index)/len(Index_sorted))

# save as CSV
np.savetxt("index.csv", Index_sorted, delimiter = ",")
np.savetxt("index_found.csv", peak_found_index, delimiter = ",")
