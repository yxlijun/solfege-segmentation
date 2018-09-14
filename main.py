from __future__ import division
from __future__ import print_function
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
from utils.peakDetection import findPeak,smooth_pitches
from pitchDetection.mfshs import MFSHS
from dataFunction import saveJson,parse_musescore,pitch_Note,post_proprocess
from alignment import sw_alignment
import warnings
import time

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

	print("mfshe time :",time.time()-start_time)
	pitches = mfshs.pitches
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

	score_note = np.array(score_note)
	result_loc_info = sw_alignment(score_note,Note_and_onset['notes'])

	#result_info,paddingzero_frame = saveJson(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,0)
	result_info,paddingzero_frame = post_proprocess(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,result_loc_info,0)

	print("total time:",time.time()-start_time)

	filename_pitch = os.path.splitext(wav_file)[0]+"_pitch.txt"
	mfshs.saveArray(filename_pitch,pitches)
	filename_onset = os.path.splitext(wav_file)[0]+"_onset.txt"
	mfshs.saveArray(filename_onset,resultOnset['onset_time'])
	filename_score = os.path.splitext(wav_file)[0]+"_score.txt"
	mfshs.saveArray(filename_score,score_note)
	filename_detnote = os.path.splitext(wav_file)[0]+"_detnote.txt"
	mfshs.saveArray(filename_detnote,Note_and_onset['notes'])

	#filename_img = os.path.splitext(wav_file)[0]+"_oriest.jpg"
	#draw_result(pitches,resultOnset['onset_frame'],paddingzero_frame,filename_img)
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

def draw_result(pitches,onset_frame,paddingzero_frame,filename_img):
	det_pitch = np.zeros(len(pitches))    
	offset_frame = onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches))

	for idx,cur_onset_frame in enumerate(onset_frame):
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame]
		det_pitch[cur_onset_frame:cur_offset_frame] = pitch

	onset_time = []
	for i in range(len(pitches)):
		if i in onset_frame:
			onset_time.append(100)
		else:
			onset_time.append(0)

	add_onset_time = []
	for i in range(len(pitches)):
		if i in paddingzero_frame:
			add_onset_time.append(100)
		else:
			add_onset_time.append(0)

	fig = plt.figure()
	plt.scatter(range(len(pitches)), pitches, color = 'r', s = 10,marker = '.',linewidths=0.001)
	plt.plot(range(len(pitches)),onset_time,color='g')
	plt.plot(range(len(pitches)),add_onset_time,color='b')

	#plt.show()
	plt.savefig(filename_img)



if __name__=='__main__':
	from multiprocessing import Process,Manager
	from timeit import Timer


	root_path = os.path.join(os.path.dirname(__file__),'data','test','20')
	get_file(root_path)
	
	
	#process_list = []
	for i in range(len(wav_files)):
		score = _main(wav_files[i],score_json[i])
		'''
		p = Process(target=_main,args=(wav_files[i],score_json[i]))
		p.start()
		process_list.append(p)
	for process in process_list:
		process.join()
	'''



