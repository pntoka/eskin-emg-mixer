This project is related to an e-skin sensor that gives 16x16 digital signal readings from a handle that has 2 force sensors. The project will start with trying to corelate the e-skin signal with the force readings as well with emg signals measured at the same time.

The task that will be recorded is grasping the handle with different forces and holding for set time.

Requirements needed for the project is:
- a way to record the emg and e-skin+force measurements at the same time
- a gui that informs the user that they are holding at the correct constant force for the specified time
- data processing for force readings
- data processing for emg data from c3d file
- data processing for e-skin readings
- simple visualisation of the measurements

Below is some information on the emg data processing:

```
def high_pass_filter(data, cutoff=20, fs=2000, order=4):
# Function to apply a high-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def low_pass_filter(data, cutoff=5, fs=2000, order=4):
    # Function to apply a low-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def band_pass_filter(data, lowcut=20, highcut=500, fs=2000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# High-pass filter
        emg_data = band_pass_filter(emg_data, lowcut = 50, highcut = 400, fs=fs)

        # center
        emg_data = emg_data - np.mean(emg_data)

        # Rectify the signal (take the absolute value)
        emg_data = np.abs(emg_data)

        # Low-pass filter to smooth the rectified signal (optional)
        emg_data = low_pass_filter(emg_data, cutoff=5, fs=fs)
```

The emg signals are recorded at 2000Hz, with 100ms as one data package, containing 8 channels, in which only channel 1 and 2 are used for recording.

The TXT files contain the recorded electromyography (EMG) signals.
The EMG data-processing procedure should include the following steps:

1. Apply the appropriate filters to remove noise and unwanted frequency components from the recorded signal.
2. Identify the maximum EMG amplitude in the first channel during the maximum-force condition.
3. Calculate the average EMG value over the maximum-force interval ( or peak interval).
4. Use this value as the reference for EMG normalisation.
5. Normalise the remaining EMG signals by dividing them by the reference maximum EMG value.
6. This process produces the normalised EMG activity for each measurement.

Grasping tasks:

- maximum squeeze for a set time and then rest
- hold for set time at constant force

### Calibration

The idea behind this is to take the e-skin and force measurements along with the emg data and investigate the corelation between the two so that we can get a force 