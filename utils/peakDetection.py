
import numpy as np      
import collections
from .parameters import hopsize_t
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

def findPeak(obs_syllable,frequency,pitches,score_length,sf_onset_frame=[]):
	obs_syllable=obs_syllable*100
	peak = collections.OrderedDict()
	for idx in range(1,len(obs_syllable)-1):
		if (obs_syllable[idx]-obs_syllable[idx-1]>0) and (obs_syllable[idx]-obs_syllable[idx+1]>0) and (obs_syllable[idx]>1):
			peak[idx] = obs_syllable[idx]

	syllable_onset = peak.keys()[0:-1]
	syllable_offset = peak.keys()[1:]
	syllable_onset.append(syllable_offset[-1])
	syllable_offset.append(len(obs_syllable)-1)
	boundaries_onsetframe = np.empty(shape=(0),dtype=np.int)
	boundaries_offsetframe = np.empty(shape=(0),dtype=np.int)
	realOnset = []

	for x in xrange(len(syllable_onset)):
		pitch = pitches[syllable_onset[x]:syllable_offset[x]]
		count = 0
		for i,det in enumerate(pitch,start=1):
			if i==len(pitch):
				break
			diff = abs(int(pitch[i])-int(pitch[i-1]))
			if int(det)==0 or diff>2 or int(det)<20:
				count = 0
			elif diff<=2:
				count+=1
			if count>=8:
				if len(realOnset)==0:
					realOnset.append(syllable_onset[x]+i-8)
					boundaries_onsetframe = np.append(boundaries_onsetframe,syllable_onset[x])
					boundaries_offsetframe = np.append(boundaries_offsetframe,syllable_offset[x])
				else:
					if ((syllable_onset[x]+i-8)-realOnset[-1])>15:
						realOnset.append(syllable_onset[x]+i-8)
						boundaries_onsetframe = np.append(boundaries_onsetframe,syllable_onset[x])
						boundaries_offsetframe = np.append(boundaries_offsetframe,syllable_offset[x])
				break

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
		pitch_info['pitch_end'] = flag+cur_onset_frame
		result_info.append(pitch_info)

	pitch_onset_frame = np.empty(shape=(0),dtype=int)
	pitch_on_length = np.empty(shape=(0),dtype=int)
	pitch_end_loc = np.empty(shape=(0),dtype=int)

	for info in result_info:
		pitch_on_length  = np.append(pitch_on_length,info['pitch_end'])
		pitch_onset_frame = np.append(pitch_onset_frame,info['onset'])
		pitch_end_loc = np.append(pitch_end_loc,info['pitch_end'])
	

	if len(real_onset_frame)>score_length:
		excessLength = len(real_onset_frame)-score_length
		del_onset = np.argsort(pitch_on_length)[0:excessLength]
		pitch_onset_frame = np.array(np.delete(pitch_onset_frame,del_onset))
	else:
		omissionLength = score_length-len(real_onset_frame)
		cancidate_onset = []
		cancidate_onset_length = []
		for idx,_detframe in enumerate(pitch_end_loc):
			omiss_pitch = pitches[_detframe:offset_frame[idx]]
			if len(omiss_pitch)>15:
				omiss_index = np.argwhere(omiss_pitch>25).ravel()
				omiss_index = np.add(omiss_index,_detframe)
				cancidate_onset.append(omiss_index)
				cancidate_onset_length.append(len(omiss_index))
		cancidate_onset_length = np.array(cancidate_onset_length)
		length_max = np.argsort(cancidate_onset_length)[-(omissionLength+5):]
		cancidate_onset = np.array(cancidate_onset)[length_max]

		max_contiguous_frame = []
		for cancidate in cancidate_onset:
			continuous =  group_consecutives(cancidate)
			con_length = np.array([len(x) for x in continuous])
			max_contiguous_frame.append(continuous[np.argsort(con_length)[-1]])

		continuous_length = np.array([len(x) for x in max_contiguous_frame])
		real_contious_range = np.array(max_contiguous_frame)[np.argsort(continuous_length)[::-1]]

		count = 0
		for _det in real_contious_range:
			_det_gt = np.where(real_onset_frame>_det[0])[0]
			if abs(_det[0] - real_onset_frame[_det_gt[0]])>15 and abs(_det[0] - real_onset_frame[_det_gt[0]-1])>15:
				real_onset_frame = np.append(real_onset_frame,_det[0])
				count+=1
				if count ==omissionLength:
					break
		pitch_onset_frame = np.sort(real_onset_frame)

	assert len(pitch_onset_frame)==score_length,'onset number must be same as scoremuse'
	onsetResult = {'onset_frame':pitch_onset_frame,'onset_time':pitch_onset_frame*hopsize_t}
	return onsetResult
