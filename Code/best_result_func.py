import pandas as pd
import glob
import numpy as np

def func_best_result(task)
	task='Bless'
	files=glob.glob('../Results/Multi_Task_'+task+'*')

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


	df_final.to_excel('../Results_Clean/Results_global_'+task+'.xlsx')
	return ltask1 ltask2
