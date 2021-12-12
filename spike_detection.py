import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, firwin
import pywt


#band pass
#def filter_data(data, low, high, sf, order=2):
    # Determine Nyquist frequency
#    nyq = sf/2

    # Set bands
#    low = low/nyq
#    high = high/nyq

    # Calculate coefficients
#    b, a = butter(order, [low, high], btype='band')

    # Filter signal
#   filtered_data = lfilter(b, a, data)
    
#    return filtered_data

def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):
    
    # Calculate threshold based on data mean
    thresh = np.mean(np.abs(data)) *tf

    # Find positions wherere the threshold is crossed
    pos = np.where(data > thresh)[0] 
    pos = pos[pos > spike_window]

    # Extract potential spikes and align them to the maximum
    spike_samp = []
    wave_form = np.empty([1, spike_window*2])
    for i in pos:
        if i < data.shape[0] - (spike_window+1):
            # Data from position where threshold is crossed to end of window
            tmp_waveform = data[i:i+spike_window*2]
            
            # Check if data in window is below upper threshold (artifact rejection)
            if np.max(tmp_waveform) < max_thresh:
                # Find sample with maximum data point in window
                tmp_samp = np.argmax(tmp_waveform) +i
                index_found.append(tmp_samp)
                
                # Re-center window on maximum sample and shift it by offset
                tmp_waveform = data[tmp_samp-(spike_window-offset):tmp_samp+(spike_window+offset)]

                # Append data
                spike_samp = np.append(spike_samp, tmp_samp)
                wave_form = np.append(wave_form, tmp_waveform.reshape(1, spike_window*2), axis=0)
    
    # Remove duplicates
    ind = np.where(np.diff(spike_samp) > 1)[0]
    spike_samp = spike_samp[ind]
    wave_form = wave_form[ind]
    
    return spike_samp, wave_form


mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
index_found = []
Class = mat['Class']
sample_rate = 25000
max_time = 1440000/25000
x = np.linspace(0,max_time,d.size)

plt.plot(x,d)
plt.show()


x = np.linspace(0,max_time,d.size)


# We will use the Daubechies(4) wavelet
wname = "db4"
maxlevel = 6
data = np.atleast_2d(d)
numwires, datalength = data.shape
	
# Initialize the container for the filtered data
fdata = np.empty((numwires, datalength))
	
for i in range(numwires):
	# Decompose the signal
	c = pywt.wavedec(data[i,:], wname, level=maxlevel)
	# Destroy the approximation coefficients
	c[0][:] = 0
	# Reconstruct the signal and save it
	fdata[i,:] = pywt.waverec(c, wname)

	if fdata.shape[0] == 1:
		spike_d = fdata.ravel() # If the signal is 1D, return a 1D array
	else:
		spike_d = fdata # Otherwise, give back the 2D array



plt.plot(x,spike_d)
plt.show()

spike_samp, wave_form = get_spikes(spike_d, spike_window=50, tf=8, offset=20)


np.random.seed(10)
fig, ax = plt.subplots(figsize=(15, 5))

for i in range(len(wave_form)):
    spike = np.random.randint(0, wave_form.shape[0])
    ax.plot(wave_form[spike, :])

ax.set_xlim([0, 90])
ax.set_xlabel('# sample', fontsize=20)
ax.set_ylabel('amplitude [uV]', fontsize=20)
ax.set_title('spike waveforms', fontsize=23)
plt.show()

Index_sorted = sorted(Index, reverse=False)
print(len(Index))

index_found = list(dict.fromkeys(index_found))
print(len(index_found))

#index_as_set = set(Index)
#intersection = index_as_set.intersection(index_found)
#intersection_list = list(intersection)
#print(intersection_list)
#print(len(intersection_list))


print()