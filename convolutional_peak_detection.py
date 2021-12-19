######################################################################################
############################### - Import Libraries - #################################

import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import pywt
import performance_metrics as pm


######################################################################################
########################### - Import the training dataset - ##########################
def load_training_dataset(dataset):
    # import from matlab file type to store in arrays
    dataset_all = spio.loadmat(dataset, squeeze_me=True)
    Data_stream = dataset_all['d']                          # array for whole data stream
    sample_rate = 25000                                     # known sample rate of data (Hz)
    Index = dataset_all['Index']                            # array for index of known spikes
    Class = dataset_all['Class']                            # array for class of known spikes
                                
    # return all arrays containing training dataset values and known sample rate
    return Data_stream, Index, Class, sample_rate

######################################################################################
######################### - Import the submission dataset - ##########################
def load_submission_dataset(dataset):
    # import from matlab file type to store in arrays
    dataset_all = spio.loadmat(dataset, squeeze_me=True)
    Data_stream = dataset_all['d']                          # array for whole data stream
    sample_rate = 25000                                     # known sample rate of data (Hz)

    # return all arrays containing training dataset values and known sample rate
    return Data_stream, sample_rate

######################################################################################
#################### - Perform Wavelet Filtering on Datastream - #####################
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
	# Reconstruct the signal
    filtered_data_stream = pywt.waverec(decomposed, wavelet)
       
    return filtered_data_stream

######################################################################################
###################### - Perform Peak Detection on Datastream - ######################
def convolution_peak_detection(filtereddatastream, threshold, windowsize):
    # initialise containers for detected peaks and their waveforms
    peak_start_index = []
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
                            peak_start_index.append(var)                    # when start point found store in holding array
                            break

    # return arrays of peak index and waveforms found
    return peak_start_index, peak_found_waveform, peak_maxima_index


######################################################################################
############################## - Performance metrics - ###############################


test_peak_detection_performance = 0
if test_peak_detection_performance == 1:

    # load the matlab data into variables
    datastream, Index, Class, sample_rate = load_training_dataset("training.mat")
    #datastream, sample_rate = load_submission_dataset("submission.mat")

    # filter the datastream using level 6 wavelet filter
    filtered_data_stream = wavelet_filter_datastream(datastream)

    # detect the peaks in the resulting datastream - store peak index and waveform in variables 0.4896, 0.42
    peak_start_index, peak_found_waveform, peak_maxima_index= convolution_peak_detection(filtered_data_stream, 0.42, 50)

    # print length of known indexes
    print("known number of peaks: ", len(Index))

    # print length of found indexes
    print("Detected number of peaks: ", len(peak_maxima_index))

    # set to 1 to plot the peak waveforms detected by peak detection
    plot_peaks = 0
    if plot_peaks == 1:
        # plot all found waveforms of peaks on same axis to verify detection is finding peaks
        fig, ax = plt.subplots(figsize=(15, 5))         # use subplot for single axis
        for i in range(len(peak_found_waveform)):       # plot every waveform in peak detected waveform array
            ax.plot(peak_found_waveform[i])             # subplot
        plt.show()         

    # get the correct and incorrect peak index lists
    incorrect_peak_index, correct_peak_index = pm.get_peak_detection_correct_and_incorrect_index(Index, peak_maxima_index)

    # get the true positive, false positive, true negative and false negative peak detections
    tp, fp, tn, fn = pm.get_peak_detection_confusion_matrix_params(Index, incorrect_peak_index, correct_peak_index, len(datastream))

    # print the true positive, false positive, true negative, and false negative values for peak detection
    print("Peak Detection TP=", tp, " FP=", fp," TN=", tn, " FN=", fn)

    # evaluate peak detection precision
    precision = tp/(tp+fp)
    print("Overall Precision = ", precision)

    # evaluate peak detection recall
    recall = tp/(tp+fn)
    print("Overall Recall = ", recall)

    # evaluate peak detection accuracy
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    print("Overall Accuracy = ", accuracy)

    # evaluate F1-Score
    f1 = 2*((precision*recall)/(precision+recall))
    print("F1 - Score = ", f1)
