import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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
from dataFunction import saveJson,draw_result,parse_musescore,pitch_Note
import warnings
warnings.filterwarnings('ignore')



def _main(wav_file):
	score_pitch = parse_musescore('./data/audio1/1A_22_final.json')
	root_path = os.path.join(os.path.dirname(__file__))
	joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')
	# load keras joint cnn model
	model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))
	# load log mel feature scaler
	scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'))
	data_wav, fs_wav = librosa.load(wav_file,sr=44100)
	mfshs = MFSHS(data_wav)
	pitchResult = mfshs.frame()
	pitches = np.array(pitchResult['pitch'])
	frequency = np.array(pitchResult['frequency'])

	log_mel_old = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1)
	log_mel = scaler_joint.transform(log_mel_old)
	log_mel = feature_reshape(log_mel, nlen=7)
	log_mel = np.expand_dims(log_mel, axis=1)
	obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)
	# post-processing the detection function
	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0


	log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(wav_file, num_bands=24,num_channels=1,frame_size=2048, hop_size=441)
	superflux_3 = madmom.features.onsets.superflux(log_filt_spec)
	onset_frame = np.argwhere((superflux_3/superflux_3.max())>0.17).flatten()

	resultOnset = findPeak(obs_syllable,frequency,pitches,onset_frame)

	filename = './data/test.json'
	std_filename = './data/audio1/test_midi.txt'
	result_info = saveJson(filename,pitches,resultOnset['onset_frame'])
	#draw_result(std_filename,pitches,resultOnset['onset_frame'])
	notes = pitch_Note(result_info)
	print  resultOnset



if __name__=='__main__':
	root_path = os.path.join(os.path.dirname(__file__),'data','audio1')
	wav_file = os.path.join(root_path,'test.mp3')
	_main(wav_file)





