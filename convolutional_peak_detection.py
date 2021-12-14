import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import pywt


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
def convolution_peak_detection(filtereddatastream, threshold, windowsize):
    # initialise containers for detected peaks and their waveforms
    peak_found_index = []
    peak_maxima_index = []
    peak_found_waveform = []
    datastream_length = int(len(filtereddatastream))
    # Convolve window over every data position of datastream to find peaks
    for x in range(int(windowsize/2),int(datastream_length-(windowsize/2))):
        # get window values arround current datapoint
        windowmin = int(x-(windowsize/2))                       # minimum index for current location
        windowmax = int(x+(windowsize/2))                       # maximum index for current location
        window = filtereddatastream[windowmin:windowmax]        # populate the window with in range values

        # calculate mean of window for peak threshold decision
        mean = abs(np.mean(window))

        # determine posibility of peak by comparing current data point value to threshold
        if (filtereddatastream[x] > mean+threshold):            # possible peaks when dava larger than mean + threshold
            peak_pos = x                                        # store potential peak position

            # check it is largest of surrounding i.e. not false summit
            window_surround = window[int((windowsize/2)-10):int((windowsize/2)+10)]      # create surrounding window 10 either side

            # if current data point is largest in surrounding window - peak has been detected
            if filtereddatastream[x] >= np.max(window_surround):

                # check is actual peak not noise spike by verifying all surrounding datapoints above threshold
                peak_points_above_thresh = 0        # initalise counter for points above threshold at peak
                # check peak points 2 either side of possible peak
                for x in range(8,13):
                    # if the value being checked is greater than mean + 1/2 of threshold
                    if window_surround[x] >= (mean + (threshold/2)):
                        peak_points_above_thresh = peak_points_above_thresh + 1 # increment counter
                # peak confirmed if more than 100 % was above threshold
                if peak_points_above_thresh == 5:
                    peak_maxima_index.append(peak_pos)      # add peak index to array
                    peak_found_waveform.append(window)      # add peak waveform to array

                    # convert index array from maxima of spike to start of spike index
                    count_below_mean = 0                                    # initialise count of points below mean
                    for var in range(peak_pos, (peak_pos-50), -1):          # start of peak point may be up to 50 before maxima peak
                        if filtereddatastream[var] <= mean :                # check if point becomes reduced to mean signifying start of peak
                            count_below_mean = count_below_mean + 1
                        if count_below_mean >= 2:                           # if point has been below mean for 2 counts (anomalie rejection)
                            peak_found_index.append(var)                    # when start point found store in holding array
                            break

    # return arrays of peak index and waveforms found
    return peak_found_index, peak_found_waveform, peak_maxima_index

##############################################################################################
################################## - Main Code Run - #########################################

# load the matlab data into variables
datastream, Index, Class, sample_rate = load_training_dataset("training.mat")
#datastream, sample_rate = load_submission_dataset("submission.mat")

# filter the datastream using level 6 wavelet filter
filtered_data_stream = wavelet_filter_datastream(datastream)

# detect the peaks in the resulting datastream - store peak index and waveform in variables 0.4896, 0.42
peak_found_index, peak_found_waveform, peak_maxima_index= convolution_peak_detection(filtered_data_stream, 0.42, 50)

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
    for var in range((peak_start+10), (peak_start-10), -1):         # initial peak point may be within margine of error +-10 to expected position
            if var in Index_sorted:                                 # check if potential peak start matches known index
                correct_index.append(peak_start)                    # if found increment correct index counter
                position_found = Index_sorted.index(var)
                Index_sorted[position_found] = 0                    # set to 0 to avoid same point being identified as correct twice
                break
# display the number of correctly detected peaks and peake detection performance
print("Number of detected peaks matching known peaks: ", len(correct_index), " Peak detection performance = ", len(correct_index)/len(Index_sorted))

# save as CSV
np.savetxt("index.csv", Index_sorted, delimiter = ",")
np.savetxt("index_found.csv", peak_found_index, delimiter = ",")
