import os 
import pickle
import librosa
import numpy as np
from keras.models import load_model
from audio_preprocessing import get_log_mel_madmom
from audio_preprocessing import feature_reshape
from utils.parameters import hopsize_t
from utils.utilFunctions import smooth_obs
from utils.peakDetection import findPeak
from pitchDetection.mfshs import MFSHS
from dataFunction import saveJson,parse_musescore,pitch_Note,post_proprocess
from alignment import sw_alignment
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _main(wav_file,input_json,output_json,mode):
	root_path = os.path.join(os.path.dirname(__file__))
	joint_cnn_model_path = os.path.join(root_path, 'cnnModels', 'joint')
	# load keras joint cnn model
	model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))
	# load log mel feature scaler
	scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'))
	data_wav, fs_wav = librosa.load(wav_file,sr=44100)
	mfshs = MFSHS(data_wav)
	mfshs.frame()
	pitches = mfshs.pitches
	#frequency = np.array(pitchResult['frequency'])

	log_mel_old = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1)
	log_mel = scaler_joint.transform(log_mel_old)
	log_mel = feature_reshape(log_mel, nlen=7)
	log_mel = np.expand_dims(log_mel, axis=1)
	obs_syllable, obs_phoneme = model_joint.predict(log_mel, batch_size=128, verbose=2)
	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0

	#print sf_onset_frame
	score_note,pauseLoc = parse_musescore(input_json)
	resultOnset = findPeak(obs_syllable,pitches,score_note)
	Note_and_onset = pitch_Note(pitches,resultOnset['onset_frame'],score_note)
	score_note = np.array(score_note)
	result_loc_info = sw_alignment(score_note,Note_and_onset['notes'])

	#result_info = saveJson(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,mode)
	result_info,paddingzero_frame = post_proprocess(output_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,result_loc_info,mode)


if __name__=='__main__':
	root_path = os.path.join(os.path.dirname(__file__),'data','test')
	wav_file = [os.path.join(root_path,file) for file in os.listdir(root_path) if file.endswith("mp3") or file.endswith("wav")]
	input_json = 'input.json'
	output_json = 'output.json'
	mode = 0
	_main(wav_file[0],input_json,output_json,mode)





