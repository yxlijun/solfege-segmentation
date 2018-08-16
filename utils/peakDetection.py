
import numpy as np      
import collections
from .parameters import hopsize_t, onset_distance,min_continue_time
from utils.utilFunctions import flag_pause



def smooth_pitches(_pitches):
	if isinstance(_pitches,list):
		result_pitches = 	[]
		for pitches in _pitches:
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
			result_pitches.append(dpitches)
		return result_pitches
	else:
		max_note,min_note = 60,20
		pitches = np.array(_pitches)
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
		return dpitches

def findPeak(obs_syllable,frequency,pitches,score_note,est_file=None):
	obs_syllable=obs_syllable*100
	peak = collections.OrderedDict()
	score_length = len(score_note)
	for idx in range(1,len(obs_syllable)-1):
		if (obs_syllable[idx]-obs_syllable[idx-1]>0) and (obs_syllable[idx]-obs_syllable[idx+1]>0) and (obs_syllable[idx]>1.5):
			peak[idx] = obs_syllable[idx]


	if len(peak.keys())==0:
		result_info = {'onset_frame':[],'onset_time':[]}
		return result_info
	onset_1 = []
	if est_file is not None:
		with open(est_file,'r') as f:
			for line in f.readlines():
				#onset_1.append(int(float(line.strip())*100))
				onset_1.append(int(line.strip()))
			
		syllable_onset = onset_1[0:-1]
		syllable_offset = onset_1[1:]
	else:
		syllable_onset = peak.keys()[0:-1]
		syllable_offset = peak.keys()[1:]
	syllable_onset.append(syllable_offset[-1])
	syllable_offset.append(len(obs_syllable)-1)
	realOnset = []
	realCount = []
	avg_Pitch = []
	for x in xrange(len(syllable_onset)):
		pitch = pitches[syllable_onset[x]:syllable_offset[x]]
		count = 0
		cancidate_onset = np.empty(shape=(0),dtype=int)
		cancidate_count = np.empty(shape=(0),dtype=int)
		real_pitches = np.empty(shape=(0),dtype=np.float32)
		cancidate_pitch= []
		for i,det in enumerate(pitch,start=1):
			if i==len(pitch):
				if count>=min_continue_time:
					cancidate_onset = np.append(cancidate_onset,syllable_onset[x]+i-count)
					cancidate_count = np.append(cancidate_count,count)
					cancidate_pitch.append(real_pitches)
				break
			diff = abs(int(pitch[i])-int(pitch[i-1]))
			if int(det)==0 or diff>2 or int(det)<20:
				if count>=min_continue_time:
					cancidate_onset = np.append(cancidate_onset,syllable_onset[x]+i-count)
					cancidate_count = np.append(cancidate_count,count)
					cancidate_pitch.append(real_pitches)
				real_pitches = np.delete(real_pitches,np.arange(len(real_pitches)))
				count = 0
			elif diff<=2:
				count+=1
				real_pitches = np.append(real_pitches,pitch[i])

		cancidate_pitch = smooth_pitches(cancidate_pitch)
		cancidate_pitch = np.array(cancidate_pitch)
		
		if len(cancidate_onset)>0:
			if len(cancidate_count)>1 and max(cancidate_count)>=30:
				count_low_30_index = np.where(cancidate_count<30)[0]
				cancidate_count = np.delete(cancidate_count,count_low_30_index)
				cancidate_onset = np.delete(cancidate_onset,count_low_30_index)
				cancidate_pitch = np.delete(cancidate_pitch,count_low_30_index)

			for i in range(len(cancidate_count)):
				onset = cancidate_onset[i]
				_count = cancidate_count[i]
				_pitches = cancidate_pitch[i]
				if len(realOnset)==0:
					realOnset.append(onset)
					realCount.append(_count)
					avg_Pitch.append(np.mean(_pitches))
				else:
					if (onset-realOnset[-1])>onset_distance:
						pitch_array = cancidate_pitch[i]
						equal = 1 if abs(avg_Pitch[-1] - np.mean(pitch_array))<0.4 else 0
						length = 1 if (realCount[-1]<30 or _count<30) else 0
						t_pitches = pitches[np.arange(realOnset[-1],onset)]
						_idx = np.where((t_pitches>58.) | (t_pitches[i]==.0))[0]
						conti = 0 if len(_idx)>0 else 1

						startx = 1 if len(realCount) - int(score_length/10)<=0 else len(realCount) - int(score_length/10)
						endx = len(score_note)-1 if (len(realCount) + int(score_length/10))> (len(score_note)-1) else (len(realCount) + int(score_length/10))
						
						score_note = np.array(score_note)
						t_score = score_note[np.arange(startx,endx)]					
						c_score = score_note[np.arange(startx-1,endx-1)]
						idx = np.where(t_score==c_score)[0]
						note = 0 if len(idx)>0 else 1

						if (equal * conti * length * note == 0):
							realOnset.append(onset)
							realCount.append(_count)
							avg_Pitch.append(np.mean(pitch_array))
		'''
		if len(cancidate_onset)>0:
			cancidate_num = np.array(cancidate_count)
			max_index = np.argmax(cancidate_num)
			onset = cancidate_onset[max_index]
			if len(realOnset)==0:
				realOnset.append(onset)
			else:
				if (onset-realOnset[-1])>onset_distance:
					realOnset.append(onset)
		'''
	print len(realOnset),len(score_note)
	real_onset_frame = np.array(sorted(realOnset),dtype=np.int)
	if len(real_onset_frame)==score_length:
		onsets = real_onset_frame.copy()
		onsetResult = {'onset_frame':onsets,'onset_time':onsets*hopsize_t}
		return onsetResult

	offset_frame = real_onset_frame[1:]
	offset_frame = np.append(offset_frame,len(pitches)-1)

	result_info = []
	for idx,cur_onset_frame in enumerate(real_onset_frame):
		pitch_info = {}
		cur_offset_frame = offset_frame[idx]
		pitch = pitches[cur_onset_frame:cur_offset_frame].tolist()
		pitch_info['pitch_length'] = len(pitch)
		pitch_info['onset'] = cur_onset_frame
		pitch_info['offset'] = offset_frame[idx]
		pitch = smooth_pitches(np.array(pitch))
		flag = flag_pause(pitch)
		pitch_info['pitch_end'] = flag
		result_info.append(pitch_info)

	pitch_onset_frame = np.empty(shape=(0),dtype=int)
	pitch_on_length = np.empty(shape=(0),dtype=int)
	pitch_end_loc = np.empty(shape=(0),dtype=int)

	for info in result_info:
		pitch_on_length  = np.append(pitch_on_length,info['pitch_end'])
		pitch_onset_frame = np.append(pitch_onset_frame,info['onset'])
		pitch_end_loc = np.append(pitch_end_loc,info['pitch_end']+info['onset'])
	
	if len(real_onset_frame)>score_length:
		excessLength = len(real_onset_frame)-score_length
		del_onset = np.argsort(pitch_on_length)[0:excessLength]
		pitch_onset_frame = np.array(np.delete(pitch_onset_frame,del_onset))
		real_onset_frame = pitch_onset_frame

	onsetResult = {'onset_frame':real_onset_frame,'onset_time':real_onset_frame*hopsize_t}
	return onsetResult