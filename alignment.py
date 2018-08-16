#-*- coding:utf-8 -*-

import numpy as np 

MATCH_COST = 0
INSERT_COST = 1
DELETE_COST = 2

def sw_alignment(score_note,det_note):
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

	query_str,ref_str = '',''
	for x in det_diff:
		query_str+=str(x)
	for x in score_diff:
		ref_str+=str(x)
	sw_ref_str,sw_query_str = WaterMan(ref_str,query_str)
	result_loc_info = locate(ref_str,query_str,sw_ref_str,sw_query_str)
	return result_loc_info

def WaterMan(s1,s2):
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
	#print(len(locate_info))
	result_loc_info = {
		'loc_info':locate_info,
		'zero_loc':pading_zero_loc
	}

	return result_loc_info
