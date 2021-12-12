import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import pywt

mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
index_found = []
peak_found_waveform = []
Class = mat['Class']
sample_rate = 25000
max_time = 1440000/25000
x = np.linspace(0,max_time,d.size)

plt.plot(x,d)
plt.show()


x = np.linspace(0,max_time,d.size)

# filter
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

plt.plot(spike_d)
plt.show()

# Convolve window to find peaks
threshold = 0.9
x = 51
for x in range(len(spike_d)-1):
    # get window values
    windowmin = x-50
    windowmax = x+50
    window = spike_d[windowmin:windowmax]

    mean = abs(np.mean(window))


    if (spike_d[x] > mean+threshold):
        pssble_pk = 1
    else:
        pssble_pk = 0
    
    if pssble_pk == 1:


        window_surround = window[40:60]
        #if on largest in window
        if spike_d[x] >= np.max(window_surround):
            pk_in_window = 1
        else:
            pk_in_window = 0

        if pk_in_window == 1:
                peak_pos = x
                index_found.append(peak_pos)
                peak_found_waveform.append(window)



Index_sorted = sorted(Index, reverse=False)
print(len(Index))

index_found = list(dict.fromkeys(index_found))
print(len(index_found))

fig, ax = plt.subplots(figsize=(15, 5))

for i in range(20):
    ax.plot(peak_found_waveform[i])



#plt.plot(peak_found_waveform[0])
plt.show()



index_as_set = set(Index)
intersection = index_as_set.intersection(index_found)
intersection_list = list(intersection)
#print(intersection_list)
print(len(intersection_list))
