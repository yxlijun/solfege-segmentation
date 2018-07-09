# - *- coding:utf-8 -*-

import json
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt
from utils.parameters import hopsize_t
from utils.utilFunctions import flag_pause

sample_ratio = 0.3

def saveJson(filename,pitches,onset_frame):
	result_info = []
	offset_frame = onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches)-1)
	for idx,cur_onset_frame in enumerate(onset_frame):
		pitch_info = {}
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame]
		voiced_length = flag_pause(pitch)
		slience_length = len(pitch)-voiced_length
		sample_voice_length = int(voiced_length*sample_ratio)
		sameple_slience_length = int(slience_length*sample_ratio)

		voice_indices = np.random.permutation(sample_voice_length)
		slience_indices = np.random.permutation(sameple_slience_length)
		voiced_pitch = pitch[:voiced_length][voice_indices]
		slience_pitch = pitch[voiced_length:][slience_indices]
		pitch = np.append(voiced_pitch,slience_pitch).tolist()
		pitch_info['onset'] = cur_onset_frame*hopsize_t*1000
		pitch_info['flag'] = sample_voice_length
		pitch_info['pitches'] = pitch
		result_info.append(pitch_info)
				

	with open(filename,'w') as f:
		json.dump(result_info,f)

	return result_info


def pitch_Note(result_info):
	det_pitches = []
	for _info in result_info:
		loc_flag = _info['flag']
		pitches = np.array(_info['pitches'][:loc_flag],dtype=int)
		pitches = pitches[np.where(pitches>20)[0]]
		unique_pitch = np.unique(pitches)
		number_dict = {}
		for _det in unique_pitch:
			count = pitches.tolist().count(_det)
			number_dict[_det] = count
		max_index = np.argmax(np.array(number_dict.values()))
		det_pitches.append(number_dict.keys()[max_index])
	return det_pitches

def parse_musescore(filename):
	with open(filename,'r') as fr:
		score_info = json.load(fr)
	linenumber = len(score_info['noteInfo'])
	score_pitches = []
	for number in range(linenumber):
		noteList = score_info['noteInfo'][number]['noteList']
		for note_info in noteList:
			if int(note_info['pitch'])!=0:
				score_pitches.append(int(note_info['pitch']))
	return score_pitches


def draw_result(std_filename,pitches,onset_frame):
	std_filename = unicode(std_filename,'utf-8')
	std_pitches,std_start,std_end = [],[],[]
	with open(std_filename) as f:
	    for line in f.readlines():
	        line = line.strip().split('\t')
	        std_pitches.append(float(line[0]))
	        std_start.append(int(float(line[1])*100))
	        std_end.append(int(float(line[2])*100))
	det_pitch = np.zeros(len(pitches))    
	offset_frame = onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches))

	for idx,cur_onset_frame in enumerate(onset_frame):
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame]
		det_pitch[cur_onset_frame:cur_offset_frame] = pitch

	std_frame,std_pitch = [],[]            
	for i in range(len(std_pitches)):
	    temp_frame,temp_pitch = [],[]
	    start,end = std_start[i],std_end[i]
	    for fr in range(start,end):
	        temp_pitch.append(std_pitches[i])
	        temp_frame.append(fr)
	    std_frame.append(temp_frame)
	    std_pitch.append(temp_pitch)

	onset_time = []
	for i in range(len(pitches)):
		if i in onset_frame:
			onset_time.append(100)
		else:
			onset_time.append(0)

	fig = plt.figure()
	plt.scatter(range(len(pitches)), det_pitch, color = 'b', s = 5,marker = '.',linewidths=0.001)
	for i in range(len(std_frame)):
		plt.scatter(std_frame[i],std_pitch[i],color='r',s=5,marker='.',linewidths=0.001)
	plt.plot(range(len(pitches)),onset_time,color='g')
	plt.show()


