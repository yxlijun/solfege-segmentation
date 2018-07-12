
import numpy as np      
import collections
from .parameters import hopsize_t, onset_distance,min_continue_time
from utils.utilFunctions import flag_pause

def mergeframe(sf_onset_frame):
	assert len(sf_onset_frame)>0,'sf_onset_frame must be have value'
	result_frame = [sf_onset_frame[0]]
	for idx,onset_frame in enumerate(sf_onset_frame):
		if onset_frame - result_frame[-1]>12:
			result_frame+=[onset_frame]
	print result_frame

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def findPeak(obs_syllable,frequency,pitches,score_length):
	obs_syllable=obs_syllable*100
	peak = collections.OrderedDict()
	for idx in range(1,len(obs_syllable)-1):
		if (obs_syllable[idx]-obs_syllable[idx-1]>0) and (obs_syllable[idx]-obs_syllable[idx+1]>0) and (obs_syllable[idx]>1):
			peak[idx] = obs_syllable[idx]
			
	syllable_onset = peak.keys()[0:-1]
	syllable_offset = peak.keys()[1:]
	syllable_onset.append(syllable_offset[-1])
	syllable_offset.append(len(obs_syllable)-1)
	realOnset = []

	for x in xrange(len(syllable_onset)):
		pitch = pitches[syllable_onset[x]:syllable_offset[x]]
		count = 0
		cancidate_onset = []
		cancidate_count = []
		for i,det in enumerate(pitch,start=1):
			if i==len(pitch):
				if count>=min_continue_time:
					cancidate_onset.append(syllable_onset[x]+i-count)
					cancidate_count.append(count)
				break
			diff = abs(int(pitch[i])-int(pitch[i-1]))
			if int(det)==0 or diff>2 or int(det)<20:
				if count>=min_continue_time:
					cancidate_onset.append(syllable_onset[x]+i-count)
					cancidate_count.append(count)
				count = 0
			elif diff<=2:
				count+=1
		if len(cancidate_onset)>0:
			cancidate_num = np.array(cancidate_count)
			max_index = np.argmax(cancidate_num)
			onset = cancidate_onset[max_index]
			if len(realOnset)==0:
				realOnset.append(onset)
			else:
				if (onset-realOnset[-1])>onset_distance:
					realOnset.append(onset)

	#for det in realOnset:
	#	print det*hopsize_t
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
