from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os 
import pickle
import time 
import warnings
import librosa
import numpy as np

from keras.models import load_model
from audio_preprocessing import get_log_mel_madmom
from audio_preprocessing import feature_reshape
from utils.parameters import hopsize_t
from utils.parameters import varin
from utils.utilFunctions import smooth_obs
from utils.peakDetection import findPeak,smooth_pitches
from pitchDetection.mfshs import MFSHS
from dataFunction import saveJson,parse_musescore,pitch_Note,post_proprocess
from alignment import sw_alignment

from draw import *
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""


score_json_name=['1A_35','1A_22','C_01','C_02','C_03',
				'C_04','C_05','Jingle_Bells','Poem_Chorus_final',
				'Yankee_Doodle_final','You_and_Me']

wav_files,score_json = [],[]
est_files = []
def _main(wav_file,score_file,est_file=None):
	print(wav_file)
	data_wav, fs_wav = librosa.load(wav_file,sr=44100)
	#start_time = time.time()
	start_time = time.time()
	
	mfshs = MFSHS(data_wav)
	mfshs.frame()

	pitches = mfshs.pitches
	energes = mfshs.energes
	zeroAmploc = mfshs.zeroAmploc
	#print('pitch detection time:',time.time()-start_time)

	root_path = os.path.join(os.path.dirname(__file__))
	joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')

	# load keras joint cnn model
	model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))
	# load log mel feature scaler
	scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'))


	log_mel_old = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1)
	log_mel = scaler_joint.transform(log_mel_old)
	log_mel = feature_reshape(log_mel, nlen=7)
	log_mel = np.expand_dims(log_mel, axis=1)

	#start_time = time.time()
	obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)
	#print('cnn detection time: ',time.time()-start_time)

	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0

	#start_time = time.time()

	score_note,pauseLoc = parse_musescore(score_file)

	resultOnset = findPeak(obs_syllable,pitches,score_note,est_file)
	filename_json = os.path.splitext(wav_file)[0]+".json"
	#print('post-processing time :' ,time.time()-start_time)
	
	Note_and_onset = pitch_Note(pitches,resultOnset['onset_frame'],score_note)
	#draw_energe(energes,resultOnset['onset_frame'],zeroAmploc)
	score_note = np.array(score_note)
	result_loc_info = sw_alignment(score_note,Note_and_onset['notes'])

	#result_info,paddingzero_frame = saveJson(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,0)
	result_info,det_Note = post_proprocess(filename_json,pitches,resultOnset['onset_frame'],zeroAmploc,score_note,pauseLoc,result_loc_info,0)

	#print("total time:",time.time()-start_time)
	filename_pitch = os.path.splitext(wav_file)[0]+"_pitch.txt"
	mfshs.saveArray(filename_pitch,pitches)
	filename_onset = os.path.splitext(wav_file)[0]+"_onset.txt"
	mfshs.saveArray(filename_onset,resultOnset['onset_time'])
	filename_score = os.path.splitext(wav_file)[0]+"_score.txt"
	mfshs.saveArray(filename_score,score_note)
	filename_detnote = os.path.splitext(wav_file)[0]+"_detnote.txt"
	mfshs.saveArray(filename_detnote,np.round(np.array(det_Note),2))

	return result_info['score']


def get_file(root_path):
	path_list = [os.path.join(root_path,file) for file in os.listdir(root_path)]
	for path in path_list:
		if os.path.isdir(path):
			get_file(path)
		elif os.path.isfile(path):
			if (path.endswith("wav") or path.endswith("mp3")):
				wav_files.append(path)
			elif path.endswith("est"):
				est_files.append(path)
			else:
				filename = os.path.splitext(os.path.basename(path))[0]
				if filename in score_json_name:
					score_json.append(path)




if __name__=='__main__':
	root_path = os.path.join(os.path.dirname(__file__),'data','test')
	get_file(root_path)
	for i in range(len(wav_files)):
		score = _main(wav_files[i],score_json[i])



