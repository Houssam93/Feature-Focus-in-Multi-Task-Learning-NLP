from sklearn.utils import shuffle
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#
import scipy.spatial as sp
import pandas as pd
import numpy as np
import tensorflow.keras

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Add, concatenate , Subtract, Activation , average,multiply,add, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.dummy import DummyClassifier
from tensorflow.keras.layers import Bidirectional
import json
import numpy as np
import os
import sys
import math as mth
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


np.random.seed(44)



#Params : Lecture des paths
link_paths='../Data/wiki_path_2best.csv'
df_paths= pd.read_csv(link_paths)
df_paths=df_paths.drop( ['Unnamed: 0'],axis=1)
df_int2=df_paths.groupby(['word1','word2','path']).size().reset_index(name='counts')
df_int2 = df_int2.sort_values(['counts'],ascending=0).reset_index(drop=True)


#Ad paths to the pair of words

def add_paths(df,df_int2):
    df=df.rename(columns={"w1": "word1", "w2": "word2"})
    words1=df['word1'].values
    words2=df['word2'].values
    col=[]
    col2=[]
    result = pd.merge(df, df_int2, how='left', on=['word1', 'word2'])
    
    result=result.fillna('')
    for word1 , word2 in zip(words1,words2):
        dataFrame=(result[result['word1'] == word1 ][result['word2']==word2]).reset_index()
        try :
            if len(dataFrame )>=2 :
                col.append(dataFrame['path'][0])
                col2.append(dataFrame['path'][1])
            else :
                col.append(dataFrame['path'][0])
                col2.append('')
        except :
            print(word1,word2)

         
    return col,col2

#Function to load embeddings
def load_embeddings(path, dimension):
    f = open(path, encoding="utf8").read().splitlines()
    vectors = {}
    for i in f:
        elems = i.split()
        vectors[" ".join(elems[:-dimension])] =  np.array(elems[-dimension:]).astype(float)
    return vectors


#Load Embeddings
embeddings = load_embeddings("../Data/Embeddings/glove.6B.300d.txt", 300)




# load embeddings and mark the words that are in the embeddings dictionary
def outer(x1,x2):
    c=np.outer(x1,x2)
    c.reshape(-1)
    return c
def func_prob(x):
    return np.arctan(x)/mth.pi+0.5
def func_prob2(x):
    return 1/(1+mth.exp(-x))
def prod(x1,x2):
    c=[a*b for (a,b) in zip(x1,x2)]
    return c

def Kullback1(x1,x2):
    c=[func_prob2(a)*mth.log(func_prob2(a)/func_prob2(b)) for (a,b) in zip(x1,x2)]
    return c
def Kullback2(x1,x2):
    c=[func_prob2(b)*mth.log(func_prob2(a)/func_prob2(b)) for (a,b) in zip(x1,x2)]
    return c
def func_rap(x1,x2):
    c=[a/b for (a,b) in zip(x1,x2)]
    return c
#aux=[cos_words,cosG,Kull,paths,boradcos,kull1_2,kullbroad]        
def get_vector_representation_of_word_pairs(dataframe):
    x1 = [embeddings[word] for word in dataframe.w1.values]
    x2 =[embeddings[word] for word in dataframe.w2.values]
    x_path1=dataframe.embed_path1.values
    x_path2=dataframe.embed_path2.values
    x_occ_1=dataframe.nb_occ1.values
    x_occ_2=dataframe.nb_occ2.values
    #print(len(x1),len(x2),len(x_path1),len(x_path2),len(x_occ_1),len(x_occ_2))
    y = dataframe.Category.values

    out_col=[embed_path(path,embeddings)for path in dataframe.col1.values]
    out_col2=[embed_path(path,embeddings)for path in dataframe.col2.values]
    #Concatenation

    x = np.hstack((x1, x2)) 

    cosine=np.diag(1 - sp.distance.cdist(x1,x2, 'cosine'))
    cosine=np.reshape(cosine,(-1,1))
    x= np.hstack((x,cosine)) 
    

    c=[prod(x11,x22) for (x11,x22) in zip(x1,x2)]
        
    x = np.hstack((x, c))
    
    c1=[Kullback1(x11,x22) for (x11,x22) in zip(x1,x2)]
    c2=[Kullback2(x11,x22) for (x11,x22) in zip(x1,x2)]
    
    x = np.hstack((x, c1))
    x= np.hstack((x, c2))

    x=np.hstack((x, list(x_path1)))
    x=np.hstack((x, list(x_path2)))
    x=np.hstack((x, (x_occ_1).reshape(len(x),-1)))
    x=np.hstack((x,(x_occ_2).reshape(len(x),-1)))
    
    broadcos=[cosine for i in range(300)]
    br=np.reshape(broadcos,(300,len(x))).T
    x=np.hstack((x, br))
    
    kull1 = [sum(e) for e in c1]
    kull2 = [sum(e) for e in c2]
    kull1 = np.reshape(kull1,(-1,1))
    kull2 = np.reshape(kull1,(-1,1))

    x = np.hstack((x, kull1))
    x = np.hstack((x, kull2))

    KullBr1=[kull1 for i in range(300)]
    KullBr2=[kull2 for i in range(300)]

    br1=np.reshape(KullBr1,(300,len(x))).T
    br2=np.reshape(KullBr2,(300,len(x))).T

    x= np.hstack((x, br1))
    x= np.hstack((x, br2))



    return x, y ,out_col,out_col2



def bal_data(name_data):
    df1Root9,df2Root9,df3Root9=three_data_aux(name_data[0:-4])
    l=min(len(df1Root9),len(df2Root9),len(df3Root9))
    df1=df1Root9[:l]
    df2=df2Root9[:l]
    df3=df3Root9[:l]
    return df1,df2,df3
    
def prep_df(df):
        cols =df.columns
        for i in range(len(cols)) :
            for j in range(len(df)):
                df[df.columns[i]].values[j]=df[df.columns[i]].values[j][0:-2]
        return df
    
    
#Load the data,  the Output is 3 Dataframes : df1 for task1 ,df2 for task2 and df3 for Random Pairs
def three_data_aux(name_data):
    Rumen,Root9,Bless,Cogalex,Weeds=False,False,False,False,False,
    if name_data == 'Rumen':
        Rumen=True
    if name_data == 'Root9':
        Root9=True
    if name_data == 'Bless':
        Bless=True
    if name_data == 'Cogalex':
        Cogalex=True
    if name_data == 'Weeds':
        Weeds=True
    

    if Weeds:
        Rumen=True
        task1Weeds="HYPER"
        task2Weeds="COO"
        link2="../Data/coordpairs2_wiki100.json"
        link1="../Data/entpairs2_wiki100.json"
    if Cogalex :
        task1="HYPER"
        task2="SYN"
        link1="../Data/CogALexV_train_v1/gold_task2.txt"
        link2="../Data/CogALexV_test_v1/gold_task2.txt"
    if Rumen :
        task1="HYPER"
        task2="SYN"
        link="../Data/RUMEN/RumenPairs.txt"
    if Root9 :
        link_hyper="../Data/ROOT9/ROOT9_hyper.txt"
        link_coord="../Data/ROOT9/ROOT9_coord.txt"
        link_random="../Data/ROOT9/ROOT9_random.txt"
        task1= "HYPER"
        task2= "COORD"
    if Bless :
        task1= "HYPER"
        task2= "MERO"
        link_coord="../Data/BLESS/BLESS_mero.txt"
        link_hyper="../Data/BLESS/BLESS_hyper.txt"
        link_random="../Data/BLESS/BLESS_random.txt"

    def get_names(cat):
        if cat == 0 : return "RANDOM"
        if cat == 1: return task1
        if cat == 2: return task2
    def get_names_Weeds1(cat):
        if cat == 0 : return "RANDOM"
        if cat == 1: return task1Weeds
    def get_names_Weeds2(cat):
        if cat == 0 : return "RANDOM"
        if cat == 1: return task2Weeds



    
    if Rumen :
        dff = pd.read_csv(link)
        dff.rename(columns={"W1":"w1", "W2":"w2","rel":"Category"}, inplace=True)
        dff["Category"] = dff["Category"].apply(get_names)
        df = dff.loc[dff.Category == task2]
        df2 = dff.loc[dff.Category == task1]
        df3 = dff.loc[dff.Category == "RANDOM"]
        #print(len(df),len(df2),len(df3))
    if Root9 or Bless:

        df = pd.read_csv(link_coord,header=None,sep = '\t')
        df.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df=prep_df(df)
        df2 = pd.read_csv(link_hyper,header=None,sep = '\t')
        df2.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df2=prep_df(df2)
        df3 = pd.read_csv(link_random,header=None,sep = '\t')
        df3.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df3=prep_df(df3)
    if Cogalex:

        dff1 = pd.read_csv(link1,header=None,sep = '\t')
        dff2 = pd.read_csv(link2,header=None,sep = '\t')
        dff1.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff2.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff3=pd.concat([dff1,dff2])
        #print(list(set(dff3.Category.values.tolist())))
        df = dff3.loc[dff3.Category == task2]
        df2 = dff3.loc[dff3.Category == task1]
        df3 = dff3.loc[dff3.Category == "RANDOM"]
    if Weeds :
        json_data=open(link1).read()
        data = json.loads(json_data)
        dff=pd.DataFrame(data)
        dff.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff["Category"] = dff["Category"].apply(get_names_Weeds1)
        df2 = dff.loc[dff.Category == task1Weeds]
        #df3 = dff.loc[dff.Category == "RANDOM"]
        #df3=df3[0:len(df2)]
        #print("taille 0,1 pour entpairs",len(df2),len(df3))
        
        json_data2=open(link2).read()
        data2 = json.loads(json_data2)
        dff2=pd.DataFrame(data2)
        dff2.rename(index=str,columns={0:"w1", 1:"w2",2:"Category"},inplace=True)
        dff2["Category"] = dff2["Category"].apply(get_names_Weeds2)
        df = dff2.loc[dff2.Category == task2Weeds]
        #df=df[0:len(df2)]
        #print("taille 0,1 pour coord",len(df))
        
        
    return df,df2,df3

#Perpare_data
def three_data(name_data):
    
    if name_data == 'Rumen' or name_data =='Root9' or name_data =='Bless'or name_data =='Cogalex' or name_data =='Weeds' :
        
        df1,df2,df3= three_data_aux(name_data)
   
    if name_data == 'Root9+Bless+Weeds':
        df1Root9,df2Root9,df3Root9=three_data_aux('Root9')
        df1Bless,df2Bless,df3Bless=three_data_aux('Bless')
        df1Weeds,df2Weeds,df3Weeds=three_data_aux('Weeds')
   
        df1=pd.concat([df1Root9,df1Bless,df1Weeds])
        df2=pd.concat([df2Root9,df2Bless,df2Weeds])
        df3=pd.concat([df3Root9,df3Bless])
        
    if name_data == 'Sym':
        link_coord_Bless="../Data/BLESS/BLESS_coord.txt"
        
        df = pd.read_csv(link_coord_Bless,header=None,sep = '\t')
        df.rename(index=str,columns={0:"w1", 2:"w2",1:"Category"},inplace=True)
        df=prep_df(df)
        
        
        df1Root9,df2Root9,df3Root9=three_data_aux('Root9')
        df1Rumen,df2Rumen,df3Rumen=three_data_aux('Rumen')
   
        df1=pd.concat([df1Root9,df])
        df2=df1Rumen
        df3=df3Rumen
    if name_data == 'Root9_Bal' or name_data == 'Bless_Bal' or name_data == 'Cogalex_Bal' or name_data == 'Weeds_Bal' :
         df1,df2,df3=bal_data(name_data)
        
    return df1,df2,df3

def transform_str_to_list(str_list):
    return [float(a[1:-1]) for a in str_list.split(',')]

#FUnction to embedd paths
def embed_path(path,embeddings):
    res=np.zeros(10*300)
    if len(path)>0 :
        words_path=path.split(' ')
        for i in range(len(words_path)):
            if words_path[i] in embeddings:
                res[i*300:(i+1)*300]=embeddings[words_path[i]]
    res=res.reshape(10,300)
    return res
def make_data(name_data): 

    df1,df2,df3=three_data(name_data)
    df_paths=pd.read_csv('../Data/embed_paths_clean.csv')
    df_paths=df_paths.drop( ['Unnamed: 0'],axis=1)
    df_paths=df_paths.rename(columns={"word1": "w1",  "word2":'w2'})

    col1,col2=add_paths(df1,df_int2)
    df1['col1']=col1
    df1['col2']=col2
    
    col1,col2=add_paths(df2,df_int2)
    df2['col1']=col1
    df2['col2']=col2
    
    col1,col2=add_paths(df3,df_int2)
    df3['col1']=col1
    df3['col2']=col2

    words_coord = list(set(df1.w1.values.tolist() + df1.w2.values.tolist()))
    words_hyper = list(set(df2.w1.values.tolist() + df2.w2.values.tolist()))
    words_random = list(set(df3.w1.values.tolist() + df3.w2.values.tolist()))

    words_ = sorted(list(set(words_coord+words_hyper+words_random)))
    #words_ = sorted(list(set(words_coord+words_hyper+words_random1+words_random2)))
    words_train, words_test =train_test_split(words_, test_size=0.4, random_state=1344)

    df_all=df1.append(df2).append(df3)
    df_all["known_words"] = df_all.apply(lambda l: l["w1"] in embeddings and l["w2"] in embeddings, axis =1  )
    words1=df_all['w1'].values
    words2=df_all['w2'].values

    result = pd.merge(df_paths, df_all, how='right', on=['w1', 'w2'])
    
    result=result.fillna('')
   

    colpath1=[]
    colpath2=[]
    nb_occ1=[]
    nb_occ2=[]
    embed_path1=[]
    embed_path2=[]

    for word1 , word2 in zip(words1,words2):
            dataFrame=(result[result['w1'] == word1 ][result['w2']==word2]).reset_index()
        
            if len(dataFrame )>=2 :
                colpath1.append(dataFrame['path'][0])
                colpath2.append(dataFrame['path'][1])
                nb_occ1.append(dataFrame['counts'][0])
                nb_occ2.append(dataFrame['counts'][1])
                embed_path1.append(transform_str_to_list(dataFrame['new_embed'][0]))
                embed_path2.append(transform_str_to_list(dataFrame['new_embed'][1]))
            elif dataFrame['path'][0]=='':
                colpath1.append('')
                colpath2.append('')
                nb_occ1.append(0)
                nb_occ2.append(0)
                embed_path1.append(list(np.zeros(512)))
                embed_path2.append(list(np.zeros(512)))
            else : 
                colpath1.append(dataFrame['path'][0])
                colpath2.append('')
                nb_occ1.append(dataFrame['counts'][0])
                nb_occ2.append(0)
                embed_path1.append(transform_str_to_list(dataFrame['new_embed'][0]))
                embed_path2.append(list(np.zeros(512)))


    df_all['col_path1']=colpath1
    df_all['col_path2']=colpath2
    df_all['nb_occ1']=nb_occ1
    df_all['nb_occ2']=nb_occ2
    df_all['embed_path1']=embed_path1
    df_all['embed_path2']=embed_path2



    # Given the words in the train and test parts, mark the pairs as training or testing, when both words of aa pair belong to the train or test vocabulary.
    df_all["is_train"] = df_all.apply(lambda l : l["w1"] in words_train and l["w2"] in words_train and l["known_words"] == True, axis=1 )
    df_all["is_test"] = df_all.apply(lambda l : l["w1"] in words_test and l["w2"] in words_test and l["known_words"] == True, axis=1)


    name_cates=(list(set(df_all['Category'].values.tolist())))
    name_cates=[tx.upper() for tx in name_cates]
    try :
        name_cates.remove("RANDOM")
        task_random="RANDOM"
    except : 
        name_cates.remove("RAND") 
        task_random="RAND"
    print(name_cates[0])
    print(name_cates[1])
    df_all['Category']=name_cates=[tx.upper() for tx in df_all['Category']]
    df_train=df_all.loc[df_all.is_train==True]

    df_test=df_all.loc[df_all.is_test==True]

    xtrainTask1, ytrainTask1,paths1_trainT1,paths2_trainT1 = get_vector_representation_of_word_pairs(df_train[df_train.Category==name_cates[0]])
    xtestTask1, ytestTask1,paths1_testT1,paths2_testT1   = get_vector_representation_of_word_pairs(df_test[df_test.Category==name_cates[0]])





    xtrainTask2, ytrainTask2,paths1_trainT2,paths2_trainT2 = get_vector_representation_of_word_pairs(df_train[df_train.Category==name_cates[1]])
    xtestTask2, ytestTask2,paths1_testT2,paths2_testT2   = get_vector_representation_of_word_pairs(df_test[df_test.Category==name_cates[1]])


    xtrainRando, ytrainRando,paths1_train_Rando,paths2_train_Rando = get_vector_representation_of_word_pairs(df_train[df_train.Category==task_random])
    xtestRando, ytestRando ,paths1_test_Rando,paths2_test_Rando  = get_vector_representation_of_word_pairs(df_test[df_test.Category==task_random])



    x_train_T1, x_train_T2,  = np.vstack((xtrainTask1, xtrainRando)), np.vstack((xtrainTask2, xtrainRando)),
                                
    y_train_T1, y_train_T2 = [1]*len(xtrainTask1) + [0]*len(xtrainRando), [1]*len(xtrainTask2) + [0]*len(xtrainRando)

    paths_train_T1_P1=np.vstack((paths1_trainT1, paths1_train_Rando))
    paths_train_T1_P2=np.vstack((paths2_trainT1, paths2_train_Rando))

    paths_train_T2_P1=np.vstack((paths1_trainT2, paths1_train_Rando))
    paths_train_T2_P2=np.vstack((paths2_trainT2, paths2_train_Rando))



    x_test_T1, x_test_T2 = np.vstack((xtestTask1, xtestRando)), np.vstack((xtestTask2, xtestRando))
    y_test_T1, y_test_T2 = [1]*len(xtestTask1) + [0]*len(xtestRando), [1]*len(xtestTask2) + [0]*len(xtestRando)

    paths_test_T1_P1=np.vstack((paths1_testT1, paths1_test_Rando))
    paths_test_T1_P2=np.vstack((paths2_testT1, paths2_test_Rando))

    paths_test_T2_P1=np.vstack((paths1_testT2, paths1_test_Rando))
    paths_test_T2_P2=np.vstack((paths2_testT2, paths2_test_Rando))



    x_train_T1, y_train_T1,paths_train_T1_P1,paths_train_T1_P2 = shuffle(x_train_T1, y_train_T1,paths_train_T1_P1,paths_train_T1_P2, random_state=1234)
    x_train_T2, y_train_T2,paths_train_T2_P1,paths_train_T2_P2 = shuffle(x_train_T2, y_train_T2,paths_train_T2_P1,paths_train_T2_P2, random_state=1234)
    x_test_T1, y_test_T1,paths_test_T1_P1,paths_test_T1_P2 = shuffle(x_test_T1, y_test_T1,paths_test_T1_P1,paths_test_T1_P2, random_state=1234)
    x_test_T2, y_test_T2,paths_test_T2_P1,paths_test_T2_P1 = shuffle(x_test_T2, y_test_T2,paths_test_T2_P1,paths_test_T2_P1, random_state=1234)
    #assert len(x_train_1) == len(y_train_1)
    #assert len(x_train_2) == len(y_train_2)
    #assert len(x_test_1) == len(y_test_1)
    #assert len(x_test_2) == len(y_test_2)
    data = {}
    for name, x_train, y_train, x_test, y_test,x_train_P1,x_train_P2,x_test_P1,x_test_P2 in zip(["Task1-Random","Task2-Random"], [x_train_T1, x_train_T2], [y_train_T1, y_train_T2], [x_test_T1, x_test_T2], [y_test_T1, y_test_T2],[paths_train_T1_P1,paths_train_T2_P1],[paths_train_T1_P2,paths_train_T2_P2],[paths_test_T1_P1,paths_test_T2_P1],[paths_test_T1_P2,paths_test_T2_P2]):   
            # Perform the splits in train, validation, unlabeled
            #x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, stratify=y_train, test_size=0.6, random_state=1234,)
            x_train, x_valid,x_train_P1,x_valid_P1,x_train_P2,x_valid_P2, y_train, y_valid = train_test_split(x_train,x_train_P1,x_train_P2,y_train, stratify=y_train,  test_size=0.30, random_state=1234,)


            # keep the train/validation/test splits so that hey can be used with multitask learning and the results are comparable between them
            #print('name '+name)
            data[name]={"x_train": x_train, "y_train":y_train,"x_train_P1":x_train_P1,"x_train_P2":x_train_P2,  "x_valid":x_valid, "y_valid":y_valid,"x_valid_P1":x_valid_P1,"x_valid_P2":x_valid_P2, "x_test":x_test,  "y_test":y_test ,"x_test_P1":x_test_P1,"x_test_P2":x_test_P2} 

    return data


def three_data_bis(name_data):
    if name_data=='BlessExp':
        linkTrain="../Data/datasets/BLESS/train.tsv"
        linkTest="../Data/datasets/BLESS/test.tsv"
        linkVal="../Data/datasets/BLESS/val.tsv"
        dfTrain= pd.read_csv(linkTrain,header=None,sep = '\t')
        dfTest= pd.read_csv(linkTest,header=None,sep = '\t')
        dfVal= pd.read_csv(linkVal,header=None,sep = '\t')
        dfTrain.columns=['w1','w2','Category']
        dfTest.columns=['w1','w2','Category']
        dfVal.columns=['w1','w2','Category']
        filter1=dfTrain['Category'].isin(['hyper','mero','random'])
        filter2=dfTest['Category'].isin(['hyper','mero','random'])
        filter3=dfVal['Category'].isin(['hyper','mero','random'])

        dfTrain=dfTrain[filter1]

        dfTest=dfTest[filter2]

        dfVal=dfVal[filter3]

    if name_data=='Root9Exp':
        linkTrain="../Data/datasets/ROOT09/train.tsv"
        linkTest="../Data/datasets/ROOT09/test.tsv"
        linkVal="../Data/datasets/ROOT09/val.tsv"
        dfTrain= pd.read_csv(linkTrain,header=None,sep = '\t')
        dfTest= pd.read_csv(linkTest,header=None,sep = '\t')
        dfVal= pd.read_csv(linkVal,header=None,sep = '\t')
        dfTrain.columns=['w1','w2','Category']
        dfTest.columns=['w1','w2','Category']
        dfVal.columns=['w1','w2','Category']
        

    dfTrain["known_words"] = dfTrain.apply(lambda l: l["w1"] in embeddings and l["w2"] in embeddings, axis =1  )
    dfTest["known_words"] = dfTest.apply(lambda l: l["w1"] in embeddings and l["w2"] in embeddings, axis =1  )
    dfVal["known_words"] = dfVal.apply(lambda l: l["w1"] in embeddings and l["w2"] in embeddings, axis =1  )

    dfTrain["is_train"]=dfTrain.apply(lambda l : True and l["known_words"] == True, axis=1 )
    dfTrain["is_test"]=dfTrain.apply(lambda l : False , axis=1 )
    dfTrain["is_valid"]=dfTrain.apply(lambda l : False , axis=1 )

    dfTest["is_train"]=dfTest.apply(lambda l : False , axis=1 )
    dfTest["is_test"]=dfTest.apply(lambda l : True and l["known_words"] == True, axis=1 )
    dfTest["is_valid"]=dfTest.apply(lambda l : False , axis=1 )

    dfVal["is_train"]=dfVal.apply(lambda l : False , axis=1 )
    dfVal["is_test"]=dfVal.apply(lambda l : False , axis=1 )
    dfVal["is_valid"]=dfVal.apply(lambda l : True and l["known_words"] == True, axis=1 )


    df_all=dfTrain.append(dfTest).append(dfVal)

    return df_all


def make_data_bis(name_data): 

    df_all=three_data_bis(name_data)
    df_paths=pd.read_csv('../Data/embed_paths_clean.csv')
    df_paths=df_paths.drop( ['Unnamed: 0'],axis=1)
    df_paths=df_paths.rename(columns={"word1": "w1",  "word2":'w2'})

    col1,col2=add_paths(df_all,df_int2)
    df_all['col1']=col1
    df_all['col2']=col2
    
    

    words1=df_all['w1'].values
    words2=df_all['w2'].values

    result = pd.merge(df_paths, df_all, how='right', on=['w1', 'w2'])
    
    result=result.fillna('')
   

    colpath1=[]
    colpath2=[]
    nb_occ1=[]
    nb_occ2=[]
    embed_path1=[]
    embed_path2=[]

    for word1 , word2 in zip(words1,words2):
            dataFrame=(result[result['w1'] == word1 ][result['w2']==word2]).reset_index()
        
            if len(dataFrame )>=2 :
                colpath1.append(dataFrame['path'][0])
                colpath2.append(dataFrame['path'][1])
                nb_occ1.append(dataFrame['counts'][0])
                nb_occ2.append(dataFrame['counts'][1])
                try :
                    embed_path1.append(transform_str_to_list(dataFrame['new_embed'][0]))
                except : embed_path1.append(list(np.zeros(512)))
                try :
                    embed_path2.append(transform_str_to_list(dataFrame['new_embed'][1]))
                except : embed_path2.append(list(np.zeros(512)))
            elif dataFrame['path'][0]=='':
                colpath1.append('')
                colpath2.append('')
                nb_occ1.append(0)
                nb_occ2.append(0)
                embed_path1.append(list(np.zeros(512)))
                embed_path2.append(list(np.zeros(512)))
            else : 
                colpath1.append(dataFrame['path'][0])
                colpath2.append('')
                nb_occ1.append(dataFrame['counts'][0])
                nb_occ2.append(0)
                embed_path1.append(transform_str_to_list(dataFrame['new_embed'][0]))
                embed_path2.append(list(np.zeros(512)))


    df_all['col_path1']=colpath1
    df_all['col_path2']=colpath2
    df_all['nb_occ1']=nb_occ1
    df_all['nb_occ2']=nb_occ2
    df_all['embed_path1']=embed_path1
    df_all['embed_path2']=embed_path2






    name_cates=(list(set(df_all['Category'].values.tolist())))
    name_cates=[tx.upper() for tx in name_cates]
    try :
        name_cates.remove("RANDOM")
        task_random="RANDOM"
    except : 
        name_cates.remove("RAND") 
        task_random="RAND"
    print(name_cates[0])
    print(name_cates[1])
    df_all['Category']=name_cates=[tx.upper() for tx in df_all['Category']]

    df_train=df_all.loc[df_all.is_train==True]
    df_valid=df_all.loc[df_all.is_valid==True]
    df_test=df_all.loc[df_all.is_test==True]

    xtrainTask1, ytrainTask1,paths1_trainT1,paths2_trainT1 = get_vector_representation_of_word_pairs(df_train[df_train.Category==name_cates[0]])
    xvalidTask1, yvalidTask1,paths1_validT1,paths2_validT1 = get_vector_representation_of_word_pairs(df_valid[df_valid.Category==name_cates[0]])
    xtestTask1, ytestTask1,paths1_testT1,paths2_testT1   = get_vector_representation_of_word_pairs(df_test[df_test.Category==name_cates[0]])





    xtrainTask2, ytrainTask2,paths1_trainT2,paths2_trainT2 = get_vector_representation_of_word_pairs(df_train[df_train.Category==name_cates[1]])
    xvalidTask2, yvalidTask2,paths1_validT2,paths2_validT2 = get_vector_representation_of_word_pairs(df_valid[df_valid.Category==name_cates[1]])
    xtestTask2, ytestTask2,paths1_testT2,paths2_testT2   = get_vector_representation_of_word_pairs(df_test[df_test.Category==name_cates[1]])


    xtrainRando, ytrainRando,paths1_train_Rando,paths2_train_Rando = get_vector_representation_of_word_pairs(df_train[df_train.Category==task_random])
    xvalidRando, yvalidRando,paths1_valid_Rando,paths2_valid_Rando = get_vector_representation_of_word_pairs(df_valid[df_valid.Category==task_random])
    xtestRando, ytestRando ,paths1_test_Rando,paths2_test_Rando  = get_vector_representation_of_word_pairs(df_test[df_test.Category==task_random])



    x_train_T1, x_train_T2,  = np.vstack((xtrainTask1, xtrainRando)), np.vstack((xtrainTask2, xtrainRando)),
                                
    y_train_T1, y_train_T2 = [1]*len(xtrainTask1) + [0]*len(xtrainRando), [1]*len(xtrainTask2) + [0]*len(xtrainRando)

    paths_train_T1_P1=np.vstack((paths1_trainT1, paths1_train_Rando))
    paths_train_T1_P2=np.vstack((paths2_trainT1, paths2_train_Rando))

    paths_train_T2_P1=np.vstack((paths1_trainT2, paths1_train_Rando))
    paths_train_T2_P2=np.vstack((paths2_trainT2, paths2_train_Rando))



    x_test_T1, x_test_T2 = np.vstack((xtestTask1, xtestRando)), np.vstack((xtestTask2, xtestRando))
    y_test_T1, y_test_T2 = [1]*len(xtestTask1) + [0]*len(xtestRando), [1]*len(xtestTask2) + [0]*len(xtestRando)

    paths_test_T1_P1=np.vstack((paths1_testT1, paths1_test_Rando))
    paths_test_T1_P2=np.vstack((paths2_testT1, paths2_test_Rando))

    paths_test_T2_P1=np.vstack((paths1_testT2, paths1_test_Rando))
    paths_test_T2_P2=np.vstack((paths2_testT2, paths2_test_Rando))


    x_valid_T1, x_valid_T2 = np.vstack((xvalidTask1, xvalidRando)), np.vstack((xvalidTask2, xvalidRando))
    y_valid_T1, y_valid_T2 = [1]*len(xvalidTask1) + [0]*len(xvalidRando), [1]*len(xvalidTask2) + [0]*len(xvalidRando)

    paths_valid_T1_P1=np.vstack((paths1_validT1, paths1_valid_Rando))
    paths_valid_T1_P2=np.vstack((paths2_validT1, paths2_valid_Rando))

    paths_valid_T2_P1=np.vstack((paths1_validT2, paths1_valid_Rando))
    paths_valid_T2_P2=np.vstack((paths2_validT2, paths2_valid_Rando))



    x_train_T1, y_train_T1,paths_train_T1_P1,paths_train_T1_P2 = shuffle(x_train_T1, y_train_T1,paths_train_T1_P1,paths_train_T1_P2, random_state=1234)
    x_train_T2, y_train_T2,paths_train_T2_P1,paths_train_T2_P2 = shuffle(x_train_T2, y_train_T2,paths_train_T2_P1,paths_train_T2_P2, random_state=1234)
    x_test_T1, y_test_T1,paths_test_T1_P1,paths_test_T1_P2 = shuffle(x_test_T1, y_test_T1,paths_test_T1_P1,paths_test_T1_P2, random_state=1234)
    x_test_T2, y_test_T2,paths_test_T2_P1,paths_test_T2_P1 = shuffle(x_test_T2, y_test_T2,paths_test_T2_P1,paths_test_T2_P1, random_state=1234)
    x_valid_T1, y_valid_T1,paths_valid_T1_P1,paths_valid_T1_P2 = shuffle(x_valid_T1, y_valid_T1,paths_valid_T1_P1,paths_valid_T1_P2, random_state=1234)
    x_valid_T2, y_valid_T2,paths_valid_T2_P1,paths_valid_T2_P1 = shuffle(x_valid_T2, y_valid_T2,paths_valid_T2_P1,paths_valid_T2_P1, random_state=1234)
    #assert len(x_train_1) == len(y_train_1)
    #assert len(x_train_2) == len(y_train_2)
    #assert len(x_test_1) == len(y_test_1)
    #assert len(x_test_2) == len(y_test_2)
    data = {}
    for name, x_train, y_train, x_test, y_test,x_valid, y_valid,x_train_P1,x_train_P2,x_test_P1,x_test_P2,x_valid_P1,x_valid_P2 in zip(["Task1-Random","Task2-Random"], [x_train_T1, x_train_T2], [y_train_T1, y_train_T2], 
        [x_test_T1, x_test_T2], [y_test_T1, y_test_T2],[x_valid_T1, x_valid_T2], [y_valid_T1, y_valid_T2],[paths_train_T1_P1,paths_train_T2_P1],[paths_train_T1_P2,paths_train_T2_P2],
        [paths_test_T1_P1,paths_test_T2_P1],[paths_test_T1_P2,paths_test_T2_P2],[paths_valid_T1_P1,paths_valid_T2_P1],[paths_valid_T1_P2,paths_valid_T2_P2]):   
           

            # keep the train/validation/test splits so that hey can be used with multitask learning and the results are comparable between them
            #print('name '+name)
            data[name]={"x_train": x_train, "y_train":y_train,"x_train_P1":x_train_P1,"x_train_P2":x_train_P2,  "x_valid":x_valid, "y_valid":y_valid,"x_valid_P1":x_valid_P1,"x_valid_P2":x_valid_P2, "x_test":x_test,  "y_test":y_test ,"x_test_P1":x_test_P1,"x_test_P2":x_test_P2} 

    return data