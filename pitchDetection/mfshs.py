# -*- coding:utf-8 -*- 

import librosa 
import numpy as np      
import os
import math

from utils.parameters import hopsize_t


fftLength = 8192
windowLength = 2048
sampleRate = 44100
h = 0.8
hammingWindow = np.zeros(windowLength)
fftResult = np.zeros(int(fftLength/2))
frameSize = 2048
H = 5
hopSize = int(hopsize_t*sampleRate)


# 全局修改hamming�?
def callHamming():
    global hammingWindow
    for i in range(0,windowLength):
        hammingWindow[i] = float(0.54-0.46*math.cos(2*math.pi*i/(windowLength-1)))

class MFSHS(object):
    frequency = []
    pitch = []
    spectrum = []
    def __init__(self,audio_data):
        super(MFSHS, self).__init__()
        self.audio_data = audio_data

    def frame(self):
        y = np.zeros(frameSize/2)
        x = np.hstack([y,self.audio_data,y])
        nFrame = np.floor((len(x)-frameSize)/hopSize)+1
        nFrame = int(nFrame)
        xFrame = np.zeros([nFrame,frameSize])
        curPos = 0
        # endTime = time.time()
        # print (endTime - startTime)
        # 计算hamming窗，只计算一次，全局使用
        callHamming()
        for index in xrange(nFrame):
            xFrame[index,:] = x[curPos:curPos+frameSize]
            meanAmp = np.mean(np.abs(xFrame[index,:]))
            note = self.getNode(xFrame[index,:]) if meanAmp>0.005 else 0
            if meanAmp<0.005:
                self.spectrum.append(np.zeros(int(fftLength/8)))
            self.pitch.append(note)
            curPos = curPos+hopSize
        result = {'pitch':self.pitch,'frequency':self.frequency}
        return result

    def getNode(self,data):
        # preLength = int(math.log(sampleRate/3,2))
        # postLength = preLength*2
        # fftLengthResult = preLength if (abs(preLength-sampleRate/3) < abs(postLength-sampleRate/3)) else postLength
        # print fftLengthResult
        fPitchResult = self.calculatePitcher(data)
        fPitchResult = 0 if fPitchResult <= 50 else fPitchResult
        fNote = (69+12*math.log(fPitchResult/440)/math.log(2)) if fPitchResult > 0 else 0
        fNote = (fNote-20) if (fNote > 0) else fNote
        self.frequency.append(fPitchResult)
        return fNote

    def calculatePitcher(self,rawMicDat):
        global fftResult
        allFFTResult = np.zeros(fftLength)
        allFFTResult[0:windowLength] = rawMicDat*hammingWindow
        fftResultNoPhase = np.fft.fft(allFFTResult)
        fftResultNoPhase = np.abs(fftResultNoPhase)
        fftResult[0:int(fftLength/2)] = np.zeros(int(fftLength/2))
        fftResult[0:int(fftLength/8)] = fftResultNoPhase[0:int(fftLength/8)]
        self.spectrum.append(fftResult[0:int(fftLength/8)])
        return self.calculateMFSHPitch()

    def calculateMFSHPitch(self):
        maxResultIndex = (np.where(fftResult == max(fftResult)))[0][0]
        p = np.zeros(H)
        for i in range(0,H):
            p[i] = 0
            L = min(10, int(math.floor((i+1)*fftLength/8/(maxResultIndex+1))))
            for j in range(0,L-1):
                if round((j+1)*(maxResultIndex+1)/(i+1)) != 0:
                    p[i] += fftResult[int(round((j+1)*(maxResultIndex+1)/(i+1))-1)]*pow(h,j)
        maxPIndex = (np.where(p == max(p)))[0][0]
        f0 = round((maxResultIndex+1)/(maxPIndex+1))/fftLength
        if f0 > 1100.0/sampleRate:
            f0 = 0
        fPitch = f0*sampleRate
        return fPitch

    def calPSF(self):
        M = 4
        PSF = np.empty(shape=(0),dtype=np.float32)
        Spectrum = np.array(self.spectrum,dtype=type(self.spectrum))
        for i in xrange(len(self.pitch)):
            if self.pitch[i]>0 and i>0:     
                P1 = int(self.pitch[i]+20)-4
                P2 = int(self.pitch[i]+20)+4
                fre1 = 440*pow(2,(P1-69)/12.0)
                fre2 = 440*pow(2,(P2-69)/12.0)
                kf1 = int(fre1*fftLength / float(sampleRate))
                kf2 = int(fre2*fftLength / float(sampleRate))
                start = i-M if i-M>=0 else 0
                spec = np.zeros((kf2-kf1))
                for j in range(start,i):
                    tempSpec = Spectrum[j,kf1:kf2]
                    for k in range(len(tempSpec)):
                        spec[k]+=tempSpec[k]
                spec/=float(i-start)
                decspec = Spectrum[i,kf1:kf2]
                diff = np.maximum(decspec-spec,0)
                PSF = np.append(PSF,np.sum(diff))
                if np.sum(diff)>0:
                    print np.sum(diff)
            else:
                PSF = np.append(PSF,0)
        return PSF

    def saveArray(self,filename,Array_list):
        with open(filename,"w") as f:
            for arr in Array_list:
                f.write(str(arr)+"\n")





