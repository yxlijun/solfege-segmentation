# - *- coding:utf-8 -*-
from __future__ import division
import json
import numpy as np 
#import matplotlib 
#import matplotlib.pyplot as plt
from utils.parameters import hopsize_t
from utils.utilFunctions import flag_pause
from utils.Dtw import detNote_map_score,detNote_map_score_code,detNote_insert_score,detNote_insert_score_LLY

sample_ratio = 0.3


def filter_pitch(pitches,score_note,paddingzero=False):
	max_note,min_note = 60,20
	pitches = np.array(pitches)
	pitches[np.where(pitches>max_note)[0]] = 0.0
	pitches[np.where(pitches<min_note)[0]] = 0.0
	dpitches = np.copy(pitches)
	for i in range(len(dpitches)-2):
		indices = np.argsort(pitches[i:i+3])
		diff1,diff2 = abs(pitches[i+indices[0]]-pitches[i+indices[1]]),abs(pitches[i+indices[1]]-pitches[i+indices[2]])
		if diff1>2 and diff2<=2:
			dpitches[i+indices[0]] = dpitches[i+indices[2]]
		elif diff1<=2 and diff2>2:
			dpitches[i+indices[2]] = dpitches[i+indices[0]]

	zero_indices = np.where(dpitches==0)[0]
	if len(zero_indices)<=15 and len(zero_indices)>0:
		dpitches[zero_indices] = dpitches[0]
	elif len(zero_indices)>15:
		dpitches[zero_indices[0]:] = 0.0
	if paddingzero and len(zero_indices)<=15:
		dpitches = np.append(dpitches,np.zeros(15))
	return dpitches.tolist()


def process_pitch(pitches,onset_frame,score_note):
	result_info = []
	offset_frame = onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches)-1)
	for idx,cur_onset_frame in enumerate(onset_frame):
		pitch_info = {}
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame]
		voiced_length = flag_pause(pitch)
		pitch_info['onset'] = cur_onset_frame
		pitch_info['flag'] = voiced_length
		pitch_info['pitches'] = filter_pitch(pitch,score_note)
		result_info.append(pitch_info)
	return result_info


def pitch_Note(pitches,onset_frame,score_note):
	result_info = process_pitch(pitches,onset_frame,score_note)
	det_pitches = []
	det_onsets = []
	for _info in result_info:
		loc_flag = _info['flag']
		pitches = np.array(_info['pitches'][:loc_flag],dtype=int)
		pitches = pitches[np.where(pitches>20)[0]]
		unique_pitch = np.unique(pitches)
		number_dict = {}
		for _det in unique_pitch:
			count = pitches.tolist().count(_det)
			number_dict[_det] = count
		number_values = np.array(number_dict.values())
		if len(number_values)>0:
			max_index = np.argmax(number_values)
			det_pitches.append(number_dict.keys()[max_index])
			det_onsets.append(_info['onset'])

	Note_and_onset = {'notes':det_pitches,'onsets':det_onsets}
	return Note_and_onset


def find_flag(pitches):
	flag = 0
	for i,pitch in enumerate(pitches):
	    if pitch>=20:
	        flag = i+1
	return flag


def get_result_info(onset_frame,offset_frame,pitches,score_note,pauseLoc,equalZero=[]):
	result_info = []
	det_Note = []
	for idx,cur_onset_frame in enumerate(onset_frame):
		paddingzero = True if idx in pauseLoc else False
		pitch_info = {}
		if idx in equalZero:
			pitch_info['onset'] = cur_onset_frame*hopsize_t*1000
			pitch_info['flag'] = 0
			pitch_info['pitches'] = np.zeros(10).tolist()
			result_info.append(pitch_info)
			det_Note.append(0.0)
		else:
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
			pitch = filter_pitch(pitch,score_note,paddingzero)
			flag = find_flag(pitch)
			pitch_info['onset'] = cur_onset_frame*hopsize_t*1000
			pitch_info['flag'] = flag
			pitch_info['pitches'] = pitch
			result_info.append(pitch_info)
			note = np.array(pitch[:flag])
			det_Note.append(np.mean(note))

	return result_info,det_Note


def give_score(det_Note,score_note,mode):
	det_note_octive = np.array(det_Note) % 12
	score_note_octive = np.array(score_note) % 12 
	det_note = np.array(det_Note)
	score_note = np.array(score_note)

	diff_note = np.abs(det_note - score_note)
	diff_note_octive = np.abs(det_note_octive - score_note_octive)

	correct_note_octive = np.where(diff_note_octive<=1.5)[0]
	octive_note = np.where((diff_note>=10) & (diff_note<=14))[0]
	is_octive = (len(octive_note) / len(score_note))>0.6
	count = 0
	if mode==0:
		for i,note in enumerate(score_note):
			if abs(note - det_Note[i])<=1.5:
				count+=1
			elif note<=40 and ((det_Note[i]-note)>=10 and (det_Note[i]-note)<=14):
				count+=1
			elif note>=52 and ((note - det_Note[i])>=10 and (note - det_Note[i])<=14):
				count+=1
			elif (note>=40 and note<=52) and (abs(note - det_Note[i])>=10 and abs(note - det_Note[i])<=14):
				count+=1
	elif not is_octive and mode==1:
		count = len(np.where(diff_note<=1.5)[0])
	elif is_octive and mode==1:
		for i,note in enumerate(score_note):
			if note<40 and ((det_Note[i]-note)>=10 and (det_Note[i]-note)<=14):
				count+=1
			elif note>52 and ((note - det_Note[i])>=10 and (note - det_Note[i])<=14):
				count+=1
			elif (note>=40 and note<=52) and (abs(note - det_Note[i])>=10 and abs(note - det_Note[i])<=14):
				count+=1
	score = count *100.0 / len(score_note)
	return score,is_octive


def saveJson(filename,pitches,onset_frame,score_note,pauseLoc,mode):
	result_info = []
	det_Note = []
	discardData = (len(score_note)-len(onset_frame))>0.15*len(score_note)
	if discardData:
		pass
	elif len(onset_frame)==len(score_note):
		offset_frame = onset_frame[1:]
		offset_frame = np.append(offset_frame,len(pitches)-1)
		result_info,det_Note = get_result_info(onset_frame,offset_frame,pitches,score_note,pauseLoc)
		#print "keys .......1"
	else:
		Note_and_onset = pitch_Note(pitches,onset_frame,score_note)
		#modify_onset =  detNote_map_score_code(pitches,score_note,onset_frame)
		#modify_onset = detNote_insert_score(pitches,score_note,onset_frame)
		modify_onset = detNote_insert_score_LLY(pitches,score_note,onset_frame)
		equalZero = np.where(modify_onset==0)[0]
		offset_frame = []

		
		if len(equalZero)>(len(score_note)-len(onset_frame)):
			offset_frame_temp = []
			samescore_length_onsets = detNote_map_score(Note_and_onset,score_note)
			for i in range(1,len(samescore_length_onsets)):
				for j in range(i,len(samescore_length_onsets)):
					if samescore_length_onsets[j]-samescore_length_onsets[i-1]>0:
						offset_frame_temp.append(samescore_length_onsets[j])
						break
			offset_frame_temp.append(len(pitches)-1)
			offset_frame = np.array(offset_frame_temp)
			result_info,det_Note = get_result_info(samescore_length_onsets,offset_frame,pitches,pauseLoc)
			#print "keys .......2"
		else:
			for i in range(1,len(modify_onset)-1):
				if modify_onset[i] == 0:
					modify_onset[i] = int((modify_onset[i-1]+modify_onset[i+1])/2)
			offset_frame = modify_onset[1:]
			offset_frame = np.append(offset_frame,len(pitches)-1)
			result_info,det_Note = get_result_info(modify_onset,offset_frame,pitches,score_note,pauseLoc,equalZero)
			#print 'kesy ........3'
	score,is_octive= 0,False
	if len(det_Note)>0:
		score,is_octive = give_score(det_Note,score_note,mode)
	results = {
		'score':score,
		'is_octive':is_octive,
		'pitches_info':result_info
	}
	with open(filename,'w') as f:
		json.dump(results,f)
	print('score:',score)
	return result_info


def parse_musescore(filename):
	with open(filename,'r') as fr:
		score_info = json.load(fr)
	linenumber = len(score_info['noteInfo'])
	pauseLoc,pitchesLoc = [],[]
	score_pitches = []
	count = 0
	for number in range(linenumber):
		noteList = score_info['noteInfo'][number]['noteList']
		for note_info in noteList:
			if int(note_info['pitch'])!=0:
				score_pitches.append(int(note_info['pitch']))
				pitchesLoc.append(count)
			else:
				pauseLoc.append(count)
			count+=1
	for i,pause in enumerate(pauseLoc):
		index = np.where(np.array(pitchesLoc)<pause)[0]
		pauseLoc[i] = index[-1] if len(index)>0 else 0
		
	return score_pitches,pauseLoc

'''
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
'''

