import cv2
import time
import numpy as np
from face_detection import FaceDetection
from scipy import signal
from process import butter_bandpass_filter, extractColor, FFT
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# import process
frame_in = np.zeros((10, 10, 3), np.uint8)
frame_out = np.zeros((10, 10, 3), np.uint8)

samples = []
buffer_size = 100

fft = []
bpms_order = []
freqs = []
t0 = time.time()
bpm = 0
fd = FaceDetection()
peaks = []

class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Prepare the camera...
        # self.process = Process()
        print("Camera warming up ...")
        time.sleep(1)
        print("Camera Ready")

    def get_frame(self):

        s, img = self.cap.read()
        if s:  # frame captures without errors...

            pass

        return img

    def release_camera(self):
        self.cap.release()

def main():
   cam = Camera()
   fps_F = 0
   bpms_F = 0
   times = []
   data_buffer = []
   bpms = []
   while True:
        cam1=cam.get_frame()
        frame_in = cv2.flip(cam1,1)
        # process starts
        frame, face_frame, ROI1, ROI2, status, mask = fd.face_detect(frame_in)
        frame_out = frame
        # average of 2 ROI's
        g = extractColor(ROI1,ROI2)
        L = len(data_buffer)

        if(abs(g-np.mean(data_buffer))>10 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = data_buffer[-1]
        times.append(time.time() - t0)
        data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > buffer_size:
            data_buffer = data_buffer[-buffer_size:]
            times = times[-buffer_size:]
            bpms = bpms[-buffer_size//2:]
            L = buffer_size

        processed = np.array(data_buffer)
        if L == buffer_size:

            fps = float(L) / (times[-1] - times[0])#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(times[0], times[-1], L)

            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, times, processed) #interpolation by 1
            raw = FFT(interpolated,L)#do real fft with the normalization multiplied by 10

            freqs = float(fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs

            fft = np.abs(raw)**2#get amplitude spectrum

            idx = np.where((freqs > 50) & (freqs < 150))#the range of frequency that HR is supposed to be within
            pruned = fft[idx]
            pfreq = freqs[idx]

            freqs = pfreq
            fft = pruned

            idx2 = np.argmax(pruned)#max in the range can be HR

            bpm = freqs[idx2]
            bpms.append(bpm)

            processed = butter_bandpass_filter(processed,0.6,2,fps,order = 3)
        if(mask.shape[0]!=10):
            out = np.zeros_like(face_frame)
            mask = mask.astype(np.bool)
            out[mask] = face_frame[mask]
            if(processed[-1]>np.mean(processed)):
                out[mask,2] = 180 + processed[-1]*10
            face_frame[mask] = out[mask]
            if(np.mean(bpms)>50 and np.mean(bpms)<150):
                bpms_order.append(np.mean(bpms))
                if(len(bpms_order)%20==0):
                    # print(np.mean(bpms_order),"  ", fps)
                    fps_F = fps
                    bpms_F = np.mean(bpms_order)
            BLK = np.zeros(frame_in.shape, np.uint8)
            x_offset=y_offset=0
            BLK[y_offset:y_offset+face_frame.shape[0], x_offset:x_offset+face_frame.shape[1]] = face_frame
            BLK = cv2.rectangle(BLK, (0,face_frame.shape[1]), (256,face_frame.shape[1]+65), (255, 255, 255), -1)
            if(fps_F!=0):
                BLK = cv2.putText(BLK, "FPS : "+str(round(fps_F,2)), (0,face_frame.shape[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                BLK = cv2.putText(BLK, "Pulse : "+str(round(bpms_F,2)), (0,face_frame.shape[1]+55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,0), 2, cv2.LINE_AA)
            else :
                BLK = cv2.putText(BLK, ("FPS : -,-"), (0,face_frame.shape[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                BLK = cv2.putText(BLK, ("Pulse : -,-"), (0,face_frame.shape[1]+55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,0), 2, cv2.LINE_AA)
        else:
            BLK = np.zeros(frame_in.shape, np.uint8)


        final_image = np.zeros((10,10,3),dtype=np.uint8)
        final_image = cv2.hconcat([frame_in, BLK])
        cv2.imshow("Frame", final_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   return ()

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
