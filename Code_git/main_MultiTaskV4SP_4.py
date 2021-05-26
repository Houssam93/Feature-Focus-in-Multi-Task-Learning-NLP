from joblib import Parallel, delayed
import os
import numpy as np
np.random.seed(44)
import sys
from itertools import product 
import itertools
import glob
import pandas as pd
from operator import add






#name_Datas=['Rumen','Weeds','Root9','Bless']
name_Datas=['Weeds']
#a=[[0,0,0],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

#[cos_words,cosG,KullG,paths,boradcos,kull1_2,kullbroad,cosG1d,KullG1d,LSTM]
aNull=[0,0,0,0,0,0,0,0,0,0]

a_cos=[]
a_kull=[]
a_aux=[]

for i in product([0,1],repeat=4):
    a_aux.append(list(i))

for aux in a_aux:
	aref1=aNull.copy()
	aref2=aNull.copy()
	if aux[0]==1:
		aref1[0]=1
		aref2[4]=1
	if aux[1]==1:
		aref1[1]=1
		aref2[5]=1
	if aux[2]==1:
		aref1[2]=1
		aref2[6]=1
	if aux[3]==1:
		aref1[3]=1
		aref2[7]=1
	a_cos.append(aref1)
	a_kull.append(aref2)
a_cos.remove(aNull)
a_kull.remove(aNull)
a_paths=[[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,1]]
a_all=[1,1,1,1,1,1,1,1,1,1]


def func_best_combi(name_data):


	files=glob.glob('../Results/Shared_Private'+name_data+'*')

	df_final=pd.DataFrame()
	for file in files :
	    df=pd.read_csv(file)
	    df_final=df_final.append(df)

	col1=df_final.columns[0]
	df_final=df_final.drop([col1],axis=1).reset_index(drop=True)

	list_cos=[]
	list_kull=[]
	list_paths=[]
	for i in range(len(df_final)) :
		list_cos.append(list(df_final.iloc[i][30:40]) in a_cos)
		list_kull.append(list(df_final.iloc[i][30:40]) in a_kull)
		list_paths.append(list(df_final.iloc[i][30:40]) in a_paths)
	df_final['cos_cat']=list_cos
	df_final['kull_cat']=list_kull
	df_final['paths_cat']=list_paths

	f1_moy=[]
	for i in range(len(df_final)):
	    f1_moy.append((df_final['f1_1'][i]+df_final['f1_2'][i])/2)

	df_final['f1_m']=f1_moy



	val_ind_cos=np.argmax(df_final[df_final.cos_cat==True]['f1_m'])
	

	val_ind_kull=np.argmax(df_final[df_final.kull_cat==True]['f1_m'])


	val_ind_paths=np.argmax(df_final[df_final.paths_cat==True]['f1_m'])
	

	ltask_cos=list(df_final.iloc[val_ind_cos][30:40])
	

	ltask_kull=list(df_final.iloc[val_ind_kull][30:40])
	

	ltask_paths=list(df_final.iloc[val_ind_paths][30:40])

	a_final=[]
	a_final.append(ltask_cos)

	a_final.append(ltask_kull)

	a_final.append(ltask_paths)	


	a_aux2=[]
	for i in product([0,1],repeat=3):
		a_aux2.append(list(i))
	a_aux2.remove([0,0,0])
	a_final_global=[]
	for a in a_aux2:
		aref=aNull.copy()
		for i in range(len(a)):
			if a[i]==1:
				aref=list( map(add, aref, a_final[i]) )
				a_final_global.append(aref)
	unique_data = [list(x) for x in set(tuple(x) for x in a_final_global)]
	fdata=[]
	for x in unique_data :
		x=[(y>0).astype(int) for y in x]
		fdata.append(x)
	ffdata=[list(x) for x in set(tuple(x) for x in fdata)]

	return ffdata

def main_func(FirstRun):
	if FirstRun :
		a_final=a_final=[aNull]+a_cos+a_kull+a_paths
		it=[]
		for name_data in name_Datas:
			for a in a_final:
					value_bool='../Results/Shared_Private_JOSE'+name_data+str(a)+'.csv' in glob.glob('../Results/Shared_Private_JOSE'+name_data+'*')
					print(value_bool)
					
					if value_bool:
						print('file exist')
						print(a)
						print('************************')
						print('************************')
						print('************************')
						print('************************')
						print('************************')
						print('************************')
						print('************************')
						print('************************')

						

					else :
						
						print('this file is added')
						print(a)
						it_aux=list(itertools.product([name_data],[a]))
						it+=it_aux


		
	else :
		it=[]
		for name_data in name_Datas :
			a_final=func_best_combi(name_data)
			for a in a_final:
				value_bool='../Results/Shared_Private_JOSE'+name_data+str(a)+'.csv' in glob.glob('../Results/Shared_Private_JOSE'+name_data+'*')
				print(value_bool)
				
				if value_bool:
					print('file exist')
					print(a)
					print('************************')
					print('************************')
					print('************************')
					print('************************')
					print('************************')
					print('************************')
					print('************************')
					print('************************')

					

				else :
					
					print('this file is added')
					print(a)
					it_aux=list(itertools.product([name_data],[a]))
					it+=it_aux
	return it
def func(name,list_a) :

	os.system("python3 MultiTaskV4SP.py "+name+" "+str(list_a))


	
#n_jobs=len(list(itertools.product(nb_n1,nb_n2)))
#n_jobs=len(it)
#if n_jobs>17 :
#	n_jobs=16
#n_jobs=8
n_jobs=2




print('False')
it=main_func(True)
Parallel(n_jobs=n_jobs)(delayed(func)(name,list_a) for (name,list_a) in it)

it=main_func(False)
Parallel(n_jobs=n_jobs)(delayed(func)(name,list_a) for (name,list_a) in it)

