from __future__ import division
from __future__ import print_function
from __future__ import absolute_import 

import tensorflow as tf 
import numpy as np 
import math 
import time 
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


hopsize = 512
frameSize = 4096
windowLength = 2048

import librosa
time_interval = 512.0/44100

def calhammingWindow():
	hammingWindow = np.zeros(windowLength)
	for i in range(windowLength):
		hammingWindow[i] = float(0.54-0.46*math.cos(2*math.pi*i/(windowLength-1)))
	return hammingWindow

def calhanningWindow():
	hanningWindow = np.zeros(windowLength)
	for i in range(windowLength):
		hanningWindow[i] = float(0.50-0.50*math.cos(2*math.pi*i/(windowLength-1)))
	return hanningWindow

def pad_center(window,win_len,new_len):
	if win_len>=new_len:
		return window,new_len
	new_win = np.zeros(new_len)
	start = int((new_len-win_len)/2)
	new_win[start:start+win_len] = window
	return new_win,new_len

class onset_classifier(object):
	"""docstring for onset_classifier"""
	def __init__(self, model_dir,wav_data,needed_fft_len=512,n_fft=4096,window="hamm"):
		super(onset_classifier, self).__init__()
		self.graph = self.load_graph(model_dir) 
		self.param_set()
		self.wav_data = wav_data
		self.window = calhammingWindow() if window=='hamm' else calhanningWindow
		self.n_fft = n_fft
		self.needed_fft_len = needed_fft_len
		self.onset_threshold = 0.5
		self.onset_frame = []

	def load_graph(self,model_dir):
		assert  tf.gfile.Exists(model_dir),print('model not exists')
		with tf.gfile.GFile(model_dir,"rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			with tf.Graph().as_default() as graph:
				tf.import_graph_def(graph_def,name="")
				return graph

	def detect(self):
		padding = np.zeros(hopsize*4)
		x = np.hstack([padding,self.wav_data,padding])
		nframe = int(np.floor((len(x)-frameSize)/hopsize)+1)
		curPos = 0
		spectrum = np.zeros((nframe,self.needed_fft_len))

		
		for index in range(nframe):
			cur_data = x[curPos:curPos+frameSize]
			spectrum[index] = self.spectrumOfFrame(cur_data)
			curPos+=hopsize

		startime = time.time()
		for index in range(0,nframe-9):
			input_data = spectrum[index:index+9]
			input_data = np.expand_dims(input_data,axis=0)
			input_data = np.transpose(input_data,(0,2,1))
			onset_prob = self.predict(input_data)
			if onset_prob[0]>self.onset_threshold:
				onset_time = (index+4)*time_interval
				self.onset_frame.append(int(onset_time*100))
		print(time.time()-startime)
		self.get_onset_frame()


	def spectrumOfFrame(self,data):
		allfftresult = np.zeros(self.n_fft)
		new_win,new_len = pad_center(self.window,windowLength,self.n_fft)
		allfftresult[0:new_len] = data*new_win
		fftresult = np.abs(np.fft.fft(allfftresult))
		return fftresult[0:self.needed_fft_len]


	def param_set(self):
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth =True
		self.sess = tf.Session(graph=self.graph,config=config)
		self.inputs = self.graph.get_tensor_by_name("input:0")
		self.output = self.graph.get_tensor_by_name("output:0")


	def predict(self,input_data):
		result = self.sess.run(self.output,feed_dict={self.inputs:input_data})
		return result


	def get_onset_frame(self):
		result_onset_frame = []
		i = 0

		while i<len(self.onset_frame):
			candi_frame = []
			j = i
			while j<len(self.onset_frame):
				if (self.onset_frame[j]-self.onset_frame[i])<13:
					candi_frame.append(self.onset_frame[j])
				else:
					break
				j+=1
			candi_frame = np.array(candi_frame)
			result_onset_frame.append(int(np.mean(candi_frame)))
			i = j
		return result_onset_frame

	def __del__(self):
		self.sess.close()




if __name__=='__main__':
	data_wav, fs_wav = librosa.load("./data/9.3/1/1.mp3",sr=44100)
	onset_detector = onset_classifier('./cnnModels/frozen_fft_onset.pb',data_wav)
	input_data = np.random.randn(1,512,9)
	onset_detector.detect()
	#print(result)