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
import warnings
import time

import alignment
import swalign
from dtw import dtw
from sklearn.metrics.pairwise import manhattan_distances
import collections

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MATCH_COST = 0
INSERT_COST = 1
DELETE_COST = 2


score_json_name=['1A_35','1A_22','C_01','C_02','C_03',
				'C_04','C_05','Jingle_Bells','Poem_Chorus_final',
				'Yankee_Doodle_final','You_and_Me']

wav_files,score_json = [],[]
est_files = []
def _main(wav_file,score_file,est_file=None):
	print(wav_file)
	start_time = time.time()
	data_wav, fs_wav = librosa.load(wav_file,sr=44100)
	mfshs = MFSHS(data_wav)
	pitchResult = mfshs.frame()
	pitches = np.array(pitchResult['pitch'])
	frequency = np.array(pitchResult['frequency'])
	#print 'pitch detection time:',time.time()-start_time

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

	#print 'cnn detection time: ',time.time()-start_time
	obs_syllable = np.squeeze(obs_syllable)
	obs_syllable = smooth_obs(obs_syllable)
	obs_syllable[0] = 1.0
	obs_syllable[-1] = 0.0

	start_time = time.time()

	score_note,pauseLoc = parse_musescore(score_file)

	resultOnset = findPeak(obs_syllable,frequency,pitches,score_note,est_file)
	filename_json = os.path.splitext(wav_file)[0]+".json"
	#print 'post-processing time :' ,time.time()-start_time

	split_onset = findzero(pitches,resultOnset['onset_frame'])
	Note_and_onset = pitch_Note(pitches,resultOnset['onset_frame'],score_note)
	#score_note = np.array(score_note)-12 if result_info['is_octive'] else np.array(score_note)
	score_note = np.array(score_note)
	result_loc_info = sw_test(score_note,Note_and_onset['notes'])

	#result_info,paddingzero_frame = saveJson(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,0)
	result_info,paddingzero_frame = post_proprocess(filename_json,pitches,resultOnset['onset_frame'],score_note,pauseLoc,result_loc_info,0)

	#matrix_cost(score_note,Note_and_onset['notes'],resultOnset['onset_frame'],split_onset,pitches)

	filename_pitch = os.path.splitext(wav_file)[0]+"_pitch.txt"
	mfshs.saveArray(filename_pitch,pitches)
	filename_onset = os.path.splitext(wav_file)[0]+"_onset.txt"
	mfshs.saveArray(filename_onset,resultOnset['onset_time'])
	filename_score = os.path.splitext(wav_file)[0]+"_score.txt"
	mfshs.saveArray(filename_score,score_note)
	filename_detnote = os.path.splitext(wav_file)[0]+"_detnote.txt"
	mfshs.saveArray(filename_detnote,Note_and_onset['notes'])

	filename_img = os.path.splitext(wav_file)[0]+"_oriest.jpg"
	draw_result(pitches,resultOnset['onset_frame'],paddingzero_frame,filename_img)
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


def findzero(pitches,onset_frame):
	onset = onset_frame[0:-1].tolist()
	offset = onset_frame[1:].tolist()
	onset.append(offset[-1])
	offset.append(len(pitches)-10)
	result = []
	for i in xrange(len(onset)):
		pitch = pitches[onset[i]:offset[i]]
		#pitch = smooth_pitches(pitch)
		std_value = np.std(pitch[-10:-1])
		if std_value>2:
			#print offset[i]
			result.append(offset[i])
		else:
			result.append(-1)
	return np.array(result)


def matrix_cost(score_note,det_note,onset_frame,split_onset,pitches):
	onset = onset_frame[0:-1].tolist()
	offset = onset_frame[1:].tolist()
	onset.append(offset[-1])
	offset.append(len(pitches)-10)
	indices = np.where(offset==split_onset)[0]
	indices = np.insert(indices,0,-1)
	split_pitch = []
	map_index = []
	count = 0 
	for idx,_ in enumerate(indices,start=1):
		pitch = []
		_index = []
		if idx==len(indices):
			break
		for i in range(indices[idx-1]+1,indices[idx]+1):
			pitch.append(det_note[i])
			_index.append(count)
			count+=1
		map_index.append(_index)
		split_pitch.append(pitch)

	split_pitch = np.array(split_pitch)
	cost = []
	cost_map = []
	#matrix = np.full((len(det_note),len(score_note)),0)
	matrix = np.full((len(split_pitch),len(score_note)),0)
	for i,pitch in enumerate(split_pitch):
		_len = len(pitch)
		start = map_index[i][0]-2 if (map_index[i][0]-2)>=0 else 0
		#end = map_index[i][-1]+3 if (map_index[i][-1]+3)<=len(score_note) else len(score_note)
		end = map_index[i][-1]+3
		print(start, end)
		per_cost=collections.OrderedDict()
		_cost = []
		for idx in range(start,end):
			if idx+_len>end:
				break
			std_note = score_note[idx:idx+_len]
			_l = len(std_note) if len(std_note)<=len(pitch) else len(pitch)
			diff = np.sum(np.power(std_note[:_l]-pitch[:_l],2))
			per_cost[idx] = diff
			_cost.append(diff)
			index = idx if (idx<len(score_note)) else len(score_note)-1
			matrix[i][index] = diff

		cost.append(_cost)
		cost_map.append(per_cost)
	test(cost)
	for mat in matrix:
		for m in mat:
			#print('',end=' ')
			print('{:<4d}'.format(m),end='')
		print('\t')
	print(matrix.shape)


def test(cost_matrix):
	_data = cost_matrix[1:]
	inputs = [cost_matrix[0]]
	for data in _data:
		inputs.append(data)
		inputs = traversing(inputs)

def traversing(inputs):
	lengths = []
	totalLength = 1;
	temp = []
	for row in inputs:
		lengths.append(len(row))
		totalLength *= len(row);
	for i in range(totalLength):
		j = 0
		_squence = []
		for _len in lengths:
			_squence.append(inputs[j][i % _len])
			i = int(i / _len)
			j += 1
		temp.append(sum(_squence))
	print(len(temp))
	result = [temp]
	return result


def sw_test(score_note,det_note):
	det_note = np.array(det_note)
	score_diff = score_note[1:]-score_note[0:-1]
	_lt = np.where(score_diff<0)[0]
	_eq = np.where(score_diff==0)[0]
	_gt = np.where(score_diff>0)[0]
	score_diff[_lt] = 1
	score_diff[_eq] = 2
	score_diff[_gt] = 3
	
	det_diff = det_note[1:]-det_note[0:-1]
	_lt = np.where(det_diff<0)[0]
	_eq = np.where(det_diff==0)[0]
	_gt = np.where(det_diff>0)[0]	
	det_diff[_lt] = 1
	det_diff[_eq] = 2
	det_diff[_gt] = 3
	

	#print(score_diff)
	#print(det_diff)
	dist, cost, acc, path = dtw(det_diff,score_diff, manhattan_distances)
	#print(path)
	match = 2
	mismatch = -1
	scoring = swalign.NucleotideScoringMatrix(match, mismatch)

	sw = swalign.LocalAlignment(scoring)  # you can also choose gap penalties, etc...

	query_str,ref_str = '',''
	for x in det_diff:
		query_str+=str(x)
	for x in score_diff:
		ref_str+=str(x)
	sw_ref_str,sw_query_str = WaterMan(ref_str,query_str)
	result_loc_info = locate(ref_str,query_str,sw_ref_str,sw_query_str)
	#alignment = sw.align(ref_str,query_str)
	#print(alignment.dump())
	return result_loc_info

def WaterMan(s1,s2):
	#s1 = "21331213332213321213312312331332213"
	#s2 = "1331213332133212133211312331332213"
	x=len(s1) 
	y=len(s2)
	opt = np.zeros((x+1,y+1))
	for i in range(x):
		opt[i][y] = DELETE_COST*(x-i)
	for j in range(y):
		opt[x][j] = DELETE_COST*(y-j)
	opt[x][y] = 0
	minxy = min(x,y)
	for k in range(1,minxy+1):
		for i in range(x-1,-1,-1):
			opt[i][y-k] = getMin(opt,i,y-k,s1,s2)
		for j in range(y-1,-1,-1):
			opt[x-k][j] = getMin(opt,x-k,j,s1,s2)
	for k in range(x-minxy,-1,-1):
		opt[k][0] = getMin(opt,k,0,s1,s2)
	for k in range(y-minxy,-1,-1):
		opt[0][k] = getMin(opt,0,k,s1,s2)
	i,j,a1,a2 = 0,0,"",""
	while (i<x and j<y):
		t = MATCH_COST+opt[i+1][j+1] if s1[i]==s2[j] else INSERT_COST+opt[i+1][j+1]
		if opt[i][j]==t:
			a1+=s1[i]
			a2+=s2[j]
			i+=1
			j+=1
		elif opt[i][j]==(opt[i+1][j]+DELETE_COST):
			a1+=s1[i]
			a2+='-'
			i+=1
		elif opt[i][j]==(opt[i][j+1]+DELETE_COST):
			a1+='-'
			a2+=s2[j]
			j+=1
	lenDiff = len(a1)-len(a2)
	for k in range(-lenDiff):
		a1+='-'
	for k in range(lenDiff):
		a2+='-'
	#print(a1,len(a1))
	#print(a2,len(a2))
	return a1,a2

def getMin(opt,x,y,s1,s2):
	x1 = opt[x][y+1]+2
	x2 = opt[x+1][y+1]+MATCH_COST if s1[x]==s2[y] else INSERT_COST+opt[x+1][y+1]
	x3 = opt[x+1][y]+DELETE_COST
	return min(x1,min(x2,x3))


def locate(ref_str,query_str,sw_ref_str,sw_query_str):
	locate_info = {}
	pading_zero_loc = []
	delte_loc = []
	ref_str = [str(x) for x in ref_str]
	query_str = [str(x) for x in query_str]
	for i in range(len(sw_ref_str)):
		if sw_ref_str[i]!='-' and sw_query_str[i]!='-':
			loc_ref = ref_str.index(sw_ref_str[i])
			loc_query = query_str.index(sw_query_str[i])
			ref_str[loc_ref] = -1
			query_str[loc_query] = -1
			#locate_info.append(loc_ref)
			locate_info[loc_query] = loc_ref
		elif sw_ref_str[i]!='-' and sw_query_str[i]=='-':
			loc_ref = ref_str.index(sw_ref_str[i])
			ref_str[loc_ref] = -1
			pading_zero_loc.append(loc_ref)
		elif sw_ref_str[i]=='-' and sw_query_str[i]!='-':
			loc_query = query_str.index(sw_query_str[i])
			query_str[loc_query] = -1
			delte_loc.append(loc_query)

	
	
	values = locate_info.values()
	for i in range(len(ref_str)+1):
		if (i not in values) and (i not in pading_zero_loc):
			pading_zero_loc.append(i)
	for zero_loc in pading_zero_loc:
		locate_info[10000-zero_loc] = zero_loc

	locate_info = sorted(locate_info.items(), lambda x, y: cmp(x[1], y[1]))

	#print(len(locate_info))
	#print(pading_zero_loc)
	#print(delte_loc)
    
	print(len(locate_info))
	result_loc_info = {
		'loc_info':locate_info,
		'zero_loc':pading_zero_loc
	}

	return result_loc_info

if __name__=='__main__':
	root_path = os.path.join(os.path.dirname(__file__),'data','test')
	get_file(root_path)
	total_score = []
	for i in range(len(wav_files)):
		score = _main(wav_files[i],score_json[i],est_files[i])
	#score = _main(wav_files[20],score_json[20])

	#	total_score.append(score)
	#print 'avg score:',sum(total_score)/len(total_score)





