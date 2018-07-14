
#coding:utf-8
from __future__ import division
import numpy as np 
from sklearn.metrics.pairwise import manhattan_distances
from dtw import dtw
from fuzzywuzzy import fuzz
from itertools import product,combinations
import copy 


def dtw_note(det_notes,score_note):
	det_notes = np.array(det_notes)%12
	score_note = np.array(score_note)%12
	dist, cost, acc, path = dtw(det_notes,score_note, manhattan_distances)
	return path



def detNote_map_score(Note_and_onset,score_note):
	det_notes = Note_and_onset['notes']
	det_onsets = np.array(Note_and_onset['onsets'])
	path = dtw_note(det_notes,score_note)
	score_length = len(score_note)
	det_path,score_path = np.array(path[0]),np.array(path[1])
	samescore_length_onsets = {}
	for index in range(score_length):
		indices = np.where(score_path==index)[0]
		det_indices = det_path[indices]
		samescore_length_onsets[index] = det_onsets[det_indices]
	onsets = []
	for value in samescore_length_onsets.values():
		onsets.append(value[0])
	return np.array(onsets)

def find(time, pitch):
    ''''
    find the elements
    '''
    list = []
    length_time = len(time)
    for i in range(0, length_time):
        list.append(pitch[int(time[i]) + 3])
    return list

def single_list(arr, target):
    '''
    找第一个出现的 target 的下标
    :param arr:
    :param target:
    :return:
    '''
    return arr.count(target)

def coding(list):
    '''
    后项差值编码
    :param list:
    :return:
    '''
    length = len(list)
    code = []
    for i in range(0, length - 1):
        l = list[i] - list[i + 1]
        if l > 0:
            code.append(1)
        else:
            code.append(0)
    code.append(0)
    return code

def code_change(code):
    '''
    编码成字符串序列
    :param code:
    :return:
    '''
    change_code = []
    code_length = len(code)
    for i in range(0,code_length):
        encode = int(code[i]) + 8
        encode = chr(encode + 97)
        change_code.append(encode)
    return change_code

def judgement_zero_length(llist,pos):
    tag = 0
    while llist[pos+1] == 0:
        tag = tag + 1
    return tag

def similar_judge(str1,str2):
    ratio = fuzz.QRatio(str1, str2, force_ascii=True, full_process=True)
    return ratio


def detNote_map_score_code(pitches,score_note,onset_frame):
	pitch,score,time = np.array(pitches,dtype=int),np.array(score_note),np.array(onset_frame)
	score_length = len(score)
	list1,timeoutputlist,outputlist = [],[],[]
	for i in range(0,score_length):
	    outputlist.append(0)
	for i in range(0,score_length):
	    timeoutputlist.append(0)
	list1 = find(time, pitch)
	score_code = coding(score)
	list_code = coding(list1)
	list_length = len(list_code)
	list_code2 = list_code
	list_code2 = [i + 0.5 for i in list_code2]
	list_code2 = [int(x) for x in list_code2]
	list_code = [int(x) for x in list_code]
	list_code = code_change(list_code)
	score_code = code_change(score_code)
	list_code2 = code_change(list_code2)
	k = min(score_length,list_length)
	for i in range(0,int(k/2)+1):
	    if score[i]==int(list1[i]) or score_code[i]==list_code[i] or score_code[i] == list_code2[i]:
	        outputlist[i]=int(list1[i])
	        timeoutputlist[i] = time[i]
	    elif score[-1-i]==list1[-1-i] or score_code[-1-i]==list_code[-1-i] or score_code[-1-i] == list_code2[-1-i]:
	        outputlist[score_length-1-i] = int(list1[-1-i])
	        timeoutputlist[score_length-1-i] = time[-1-i]
	'''
	原序列和编码序列进行匹配，我们默认序列前段和后端的匹配程度较高，因此从前和从后
	同时进行序列匹配映射，找到为0的元素所在位置，即为失配点，再从失配点重新进行映射
	'''
	zero_appear = single_list(outputlist,0) #序列中0出现的次数
	'''
	score_list 是 score 的 list 形式
	'''
	for i in range(0,zero_appear-1):
	    pos = outputlist.index(0)
	    if outputlist[pos+1] != 0:
	        outputlist[pos] = int(list1[pos])
	        timeoutputlist[pos] = time[pos]
	    else:
	        str1 = "".join(score_code[pos:pos+5])
	        str2 = "".join(list_code[pos:pos+5])
	        str3 = "".join(list_code2[pos:pos+5])
	        sim1 = similar_judge(str1,str2)
	        sim2 = similar_judge(str1,str3)
	        if sim1>0.6 or sim2>0.6:
	            outputlist[pos] = list1[pos]
	            timeoutputlist[pos] = time[pos]
	outputlist[-1] = list1[-1]
	outputlist[-2] = list1[-2]
	outputlist[-3] = list1[-3]
	timeoutputlist[-1] = time[-1]
	timeoutputlist[-2] = time[-2]
	timeoutputlist[-3] = time[-3]
	outputlist = [int(x) for x in outputlist]
	for i in range(1,score_length):
	    if timeoutputlist[i]<=timeoutputlist[i-1]:
	        outputlist[i] = 0
	        timeoutputlist[i] = 0
	return np.array(timeoutputlist)


def code_judge_score(score,new_list):
	score = np.array(score)
	new_list = np.array(new_list)
	mask = (score==new_list).astype(int)
	return np.sum(mask)
    

def detNote_insert_score(pitches,score_note,onset_frame):
	pitch,score,time = np.array(pitches,dtype=int),np.array(score_note),np.array(onset_frame)
	score_length = len(score_note)
	det_length = len(onset_frame)
	list1 = find(time, pitch)
	score_code = coding(score)
	list_code = coding(list1)
	need_to_insert = score_length - det_length
	new_list1 = copy.copy(list1)
	list_code = copy.copy(list_code)
	grade = []
	outputlist_list = []
	insert_indices = list(product(range(score_length), repeat=need_to_insert))
	for indices in insert_indices:
		indices = [int(i) for i in indices]
		for index in indices:
			new_list1.insert(index,99)
			list_code.insert(index,10)
		gpa = code_judge_score(score_code, list_code) + code_judge_score(score,new_list1)
		grade.append(gpa)
		cur_insert_indice = []
		for index in indices:
			new_list1.pop(index)
			list_code.pop(index)
			cur_insert_indice.append(index)
		outputlist_list.append(cur_insert_indice)

	pos = grade.index(max(grade))
	k = outputlist_list[pos]
	timeoutput = copy.copy(time).tolist()
	for i in range(len(k)):
		timeoutput.insert(k[i],0)
	return np.array(timeoutput)



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

