from scipy import signal
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a =  butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def extractColor(frame1,frame2):
    g1 = np.mean(frame1[:,:,1])
    g2 = np.mean(frame2[:,:,1])
    return (g1+g2)/2

def FFT( interpolated,L):
    interpolated = np.hamming(L) * interpolated
    norm = interpolated/np.linalg.norm(interpolated)
    return np.fft.rfft(norm*40)
