
import numpy as np      
import collections
from .parameters import hopsize_t


def findPeak(obs_syllable,frequency,pitches,sf_onset_frame=[]):
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

	loczero = np.where(pitches==0)[0]
	loczero1 = loczero[1:]-loczero[0:-1]
	zerocount = 0
	onsetZeroLoc = []
	for i,det in enumerate(loczero1):
		if det==1:
			zerocount+=1
		else:
			if zerocount>=7:
				onsetZeroLoc.append(loczero[i])
			zerocount = 0
	
	for zeroLoc in onsetZeroLoc:
		maxloc = np.where(np.array(realOnset)>=zeroLoc)[0]
		minloc = np.where(np.array(realOnset)<=zeroLoc)[0]
		if len(maxloc)>0 and len(minloc)>0:
			if abs(zeroLoc - realOnset[maxloc[0]])>20 and (abs(zeroLoc- realOnset[minloc[-1]]))>20:
				realOnset.append(zeroLoc)
				
	realOnset = np.array(sorted(realOnset),dtype=np.int)
	onsetResult = {'onset_frame':realOnset,'onset_time':realOnset*hopsize_t}
	return onsetResult
