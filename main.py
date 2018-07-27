import os 
import pickle
import librosa
import time 
import collections
import madmom
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from keras.models import load_model
from audio_preprocessing import get_log_mel_madmom
from audio_preprocessing import feature_reshape
from utils.parameters import hopsize_t
from utils.parameters import varin
from utils.utilFunctions import smooth_obs
from utils.peakDetection import findPeak
from pitchDetection.mfshs import MFSHS
from dataFunction import saveJson,parse_musescore,pitch_Note
import warnings
import time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _main(wav_file):

	start_time = time.time()
	data_wav, fs_wav = librosa.load(wav_file,sr=44100)
	mfshs = MFSHS(data_wav)
	pitchResult = mfshs.frame()
	pitches = np.array(pitchResult['pitch'])
	frequency = np.array(pitchResult['frequency'])
	print 'pitch detection time:',time.time()-start_time


	root_path = os.path.join(os.path.dirname(__file__))
	joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')


	start_time = time.time()

	# load keras joint cnn model
	model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))
	# load log mel feature scaler
	scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'))

	log_mel_old = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1)
	log_mel = scaler_joint.transform(log_mel_old)
	log_mel = feature_reshape(log_mel, nlen=7)
	log_mel = np.expand_dims(log_mel, axis=1)
	obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)

	print 'cnn detection time: ',time.time()-start_time

	# post-processing the detection function
	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0

	start_time = time.time()
	#print sf_onset_frame
	score_note,pauseLoc = parse_musescore('./data/93/1A_22_final.json')
	#print(len(score_note))
	resultOnset = findPeak(obs_syllable,frequency,pitches,len(score_note))
	filename_json = os.path.splitext(wav_file)[0]+".json"
	#std_filename = './data/audio1/test_midi.txt'
	result_info = saveJson(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,1)
	print 'post-processing time :' ,time.time()-start_time
	#draw_result(std_filename,pitches,resultOnset['onset_frame'])
	filename_pitch = os.path.splitext(wav_file)[0]+"_pitch.txt"
	mfshs.saveArray(filename_pitch,pitches)
	filename_prob = os.path.splitext(wav_file)[0]+"_prob.txt"
	mfshs.saveArray(filename_prob,obs_syllable)
	filename_score = os.path.splitext(wav_file)[0]+"_score.txt"
	mfshs.saveArray(filename_score,score_note)
	for pit_time in resultOnset['onset_time']:
		print pit_time
		#pass


if __name__=='__main__':
	root_path = os.path.join(os.path.dirname(__file__),'data','95')
	wav_file = [os.path.join(root_path,file) for file in os.listdir(root_path) if file.endswith("mp3") or file.endswith("wav")]
	_main(wav_file[0])





