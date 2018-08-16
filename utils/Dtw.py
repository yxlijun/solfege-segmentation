
#coding:utf-8
from __future__ import division
import numpy as np 
from itertools import combinations


def search(pitches,onset_frame):
	Note = []
	for i in range(len(onset_frame)):
		start = onset_frame[i]
		total,count = 0,0
		for j in range(10):
			if (abs(pitches[start+j]-pitches[start+j+1])<0.5):
				count+=1
				total+=pitches[start+j+1]
		Note.append(total/count)
	return Note

def diff_code(score_note):
	diffcode = []
	for idx in range(len(score_note)-1):
		if (score_note[idx]<score_note[idx+1]):
			diffcode.append(1)
		elif score_note[idx]>score_note[idx+1]:
			diffcode.append(-1)
		else:
			diffcode.append(0)
	return diffcode

def detNote_insert_score_LLY(pitches,score_note,onset_frame):
	diffLength = len(score_note)-len(onset_frame)
	insert_indices = list(combinations(range(len(onset_frame)+1), diffLength))
	Note = search(pitches,onset_frame)
	temp_p = diff_code(score_note)
	insert_Count = []
	for i in range(len(insert_indices)):
		count = 0
		insert_Note = Note[:]
		temp_q = []
		for j in range(diffLength):
			insert_Note.insert(insert_indices[i][j]+j,0)
		for idx in range(len(score_note)-1):
			if insert_Note[idx]==0:
				temp_q.append(temp_p[idx])
			elif insert_Note[idx+1]==0:
				temp_q.append(temp_p[idx])
			elif (insert_Note[idx]<insert_Note[idx+1]-0.5):
				temp_q.append(1)
			elif (insert_Note[idx]-insert_Note[idx+1]>0.5):
				temp_q.append(-1)
			else:
				temp_q.append(0)
		for n in range(len(score_note)-1):
			if (temp_p[n]==temp_q[n]):
				count+=1
		insert_Count.append(count)

	insert_loc = insert_indices[insert_Count.index(max(insert_Count))]
	insert_onset_frame = onset_frame.tolist()
	for loc in insert_loc:
		insert_onset_frame.insert(loc,0)
	return np.array(insert_onset_frame)

