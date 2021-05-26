import time
from itertools import product 
import itertools
import re
from operator import add

num_epochs=2000
#name_Data="Rumen"

batch=64

#name_Data='Root9'
#name_Data='Weeds'
import pandas as pd
import glob
import numpy as np

def func_best_result(task):
	
	files=glob.glob('../Results/NN_Baseline_'+task+'*')

	df_final=pd.DataFrame()
	for file in files :
	    df=pd.read_csv(file)
	    df_final=df_final.append(df)

	col1=df_final.columns[0]
	df_final=df_final.drop([col1],axis=1).reset_index(drop=True)
	val_ind_task1=np.argmax(df_final['f1_1'])
	val_ind_task2=np.argmax(df_final['f1_2'])
	ltask1=list(df_final.iloc[val_ind_task1][10:])
	ltask2=list(df_final.iloc[val_ind_task2][10:])


	#df_final.to_excel('../Results_Clean/Results_global_'+task+'.xlsx')
	return ltask1,ltask2

exec(open('prepare_data_multi_taskJose.py').read())
#exec(open('prepare_data_multi_task.py').read())






def list_of_input(list_in):
    a=[]
    b=[]
    for i in product([0,1],repeat=len(list_in)):
        a.append(list(i))
    for j in range(len(a)):
        list2=a[j]
        list3=[in1-inlist2 for in1,inlist2 in zip(list_in,list2)]
        if -1 not in list3 and sum(list3)>0:
            b.append(list3)
    return b
#############################ATTEEEEEEEEEEEEENTIOON
#input_dim=len(data["Task1-Random"]["x_train"][0])
input_dim=3429
exec(open('Models_Multi_TaskSP.py').read())

is_sharedBool=True

def main_multi_task(data,name_model,Multi,is_input,is_shared,nb_occ):
    out_data1=[]

   
    for i in range(nb_occ):
        
        
        df_max=train_model(data,name_model,Multi,is_input,is_shared)
        print('model_train*****************')
      
        out_data1.append(df_max)
        
    df_concat_max=pd.concat(out_data1)
    by_row_index = df_concat_max.groupby(df_concat_max.index)
    df_means_max = by_row_index.mean()
    return df_means_max


#name_models=[ModelSharedPrivate,NN_Baseline,Model_AllShared]
#name_models=[NN_Baseline]
name_models=[ModelSharedPrivate]
#name_models=[Model_AllShared]
#str_models=['NN_Baseline']
#str_models=['Model_AllShared']
str_models=['ModelSharedPrivate']
def main(name_Data,inp) :
	#name_Data='Rumen'

	print(name_Data)
	try :
		data=make_data(name_Data)
	except :
		print('------------------------')
		print('*************************')
		print('------------------------')

		print(name_Data)
	
	

	str_f=''
	for str_a in inp :
		str_f+=str_a
	

	#is_input=[int(s) for s in re.findall(r'\b\d+\b', str_f)]
	#print('is _ input')
	#print(is_input)
	if is_sharedBool :
		is_shared=[int(s) for s in re.findall(r'\b\d+\b', str_f)]
	else :
		is_input=[int(s) for s in re.findall(r'\b\d+\b', str_f)] 
	#print(is_input)



	nb_occ=1

	df_restit=pd.DataFrame()


#[cos_words,cosG,Kull,paths,boradcos,kull1_2,kullbroad]		
	for i in range(len(name_models)) :
			if str_models[i]=='NN_Baseline' :
				Multi=False
			else : Multi=True
			if str_models[i]=='NN_Baseline' or str_models[i]=='Model_AllShared' :
				
				is_shared=[]
				modeel = name_models[i]
				df_final=main_multi_task(data,modeel,Multi,is_input,is_shared,nb_occ)
				df_final['name_model']=str_models[i]
				df_final['cos_words_In']=is_input[0]
				df_final['cosG_In']=is_input[1]
				df_final['broad_cos']=is_input[4]
				df_final['KullG']=is_input[2]
				df_final['paths']=is_input[3]
				df_final['Kull']=is_input[5]
				df_final['KullBroad']=is_input[6]
				df_final['cosG1d']=is_input[7]
				df_final['KullG1d']=is_input[8]
				df_final['LSTM']=is_input[9]



				
				
				df_restit=df_restit.append(df_final)
				
			
			else :
				
				
				
				is_input1,is_input2=func_best_result(name_Data)
				print('is_input')
				print(is_input1)
				print(is_input2)
	
				modeel = name_models[i]
				df_final=main_multi_task(data,modeel,Multi,[is_input1,is_input2],is_shared,nb_occ)
				df_final['name_model']=str_models[i]
				
				df_final['cos_words_In1']=is_input1[0]
				df_final['cosG_In1']=is_input1[1]
				df_final['broad_cos1']=is_input1[2]
				df_final['cosG1d1']=is_input1[3]
				df_final['Kull1']=is_input1[4]
				df_final['KullG1']=is_input1[5]
				df_final['KullBroad1']=is_input1[6]
				df_final['KullG1d1']=is_input1[7]
				df_final['paths1']=is_input1[8]
				df_final['LSTM1']=is_input1[9]
				
				
				
				
				

				df_final['cos_words_In2']=is_input2[0]
				df_final['cosG_In2']=is_input2[1]
				df_final['broad_cos2']=is_input2[2]
				df_final['cosG1d2']=is_input2[3]
				df_final['Kull2']=is_input2[4]
				df_final['KullG2']=is_input2[5]
				df_final['KullBroad2']=is_input2[6]
				df_final['KullG1d2']=is_input2[7]
				df_final['paths2']=is_input2[8]
				df_final['LSTM2']=is_input2[9]


				df_final['cos_words_In_S']=is_shared[0]
				df_final['cosG_In_S']=is_shared[1]
				df_final['broad_cos_S']=is_shared[2]
				df_final['cosG1d_S']=is_shared[3]
				df_final['Kull_S']=is_shared[4]
				df_final['KullG_S']=is_shared[5]
				df_final['KullBroad_S']=is_shared[6]
				df_final['KullG1d_S']=is_shared[7]
				df_final['paths_S']=is_shared[8]
				df_final['LSTM_S']=is_shared[9]

				df_restit=df_restit.append(df_final)


	print("on enregistre")
	df_restit.to_csv('../Results/Shared_Private_JOSE'+str(name_Data)+str(is_shared)+'.csv')
	#df_restit.to_csv('../Results/Multi_Task_'+str(name_Data)+str(is_input)+'.csv')

if __name__ == "__main__":
	#print(sys.argv[1])
	#print(sys.argv[2:])

	main(sys.argv[1],sys.argv[2:])