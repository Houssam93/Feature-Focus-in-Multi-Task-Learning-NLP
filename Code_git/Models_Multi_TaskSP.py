import itertools

def ModelSharedPrivate(nb_n1,nb_n2,is_input1,is_input2,is_shared):

    inputs = Input(shape=(input_dim,),name='main_input')
    lstm_out=200

    aux_input1=Input(shape=(10,300,),name='aux_input1')
    aux_input2=Input(shape=(10,300,),name='aux_input2')

    words1_2=Lambda(lambda x: x[:,0:600])(inputs)
    
    cos_words=Lambda(lambda x: x[:,600:601])(inputs)
    cosG=Lambda(lambda x: x[:,601:901])(inputs)
    KullG=Lambda(lambda x: x[:,901:1501])(inputs)
    paths=Lambda(lambda x: x[:,1501:2527])(inputs)
    boradcos=Lambda(lambda x: x[:,2527:2827])(inputs)
    kull=Lambda(lambda x: x[:,2827:2829])(inputs)
    kullbroad=Lambda(lambda x: x[:,2829:3429])(inputs)
    aux=[cos_words,cosG,boradcos,0,kull,KullG,kullbroad,0,paths,0]
    ind_true=[0,1,2,4,5,6]
   
    main_input1=Lambda(lambda x: x[:,:])(words1_2)
    main_input2=Lambda(lambda x: x[:,:])(words1_2)
    for ind in ind_true :
        if is_input1[ind] ==1 :

            main_input1=concatenate([main_input1,aux[ind]])
        if is_input2[ind] ==1 :
            main_input2=concatenate([main_input2,aux[ind]])

    if is_input1[8] ==1 or is_input1[9] ==1  :
        main_input1=concatenate([main_input1,aux[8]])

    if is_input2[8] ==1 or is_input2[9] ==1  :
        main_input2=concatenate([main_input2,aux[8]])

    
    if is_input1[3] :
        xcosg1=Dense(1, activation='sigmoid')(cosG)
        main_input1=concatenate([main_input1,xcosg1])

    if is_input1[7] :
        xkullg1=Dense(1, activation='sigmoid')(KullG)
        main_input1=concatenate([main_input1,xkullg1])

    if is_input2[3] :
        xcosg2=Dense(1, activation='sigmoid')(cosG)
        main_input2=concatenate([main_input2,xcosg2])

    if is_input2[7] :
        xkullg2=Dense(1, activation='sigmoid')(KullG)
        main_input2=concatenate([main_input1,xkullg2])


    #if is_input1[9]:
    #    lstm_layer1_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
    #    lstm_layer2_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
    #    lstm_layer_global_1=average([lstm_layer1_1,lstm_layer2_1])
    #    main_input1=concatenate([main_input1,lstm_layer_global_1])


    #if is_input2[9]:
    #    lstm_layer1_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
    #    lstm_layer2_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
    #    lstm_layer_global_2=average([lstm_layer1_2,lstm_layer2_2])
    #    main_input2=concatenate([main_input2,lstm_layer_global_2])





    shared_input=Lambda(lambda x: x[:,:])(words1_2)
    for ind in ind_true :
        if is_shared[ind] ==1 :
            shared_input=concatenate([shared_input,aux[ind]])

    if is_shared[3] :
        xcosgS=Dense(1, activation='sigmoid')(cosG)
        shared_input=concatenate([shared_input,xcosgS])

    if is_shared[7] :
        xkullgS=Dense(1, activation='sigmoid')(KullG)
        shared_input=concatenate([shared_input,xkullgS])


    if is_shared[9]:
        lstm_layer1_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_s=average([lstm_layer1_s,lstm_layer2_s])
        shared_input=concatenate([shared_input,lstm_layer_global_s])



    xshared = Dense(nb_n1, activation='sigmoid')(shared_input)
   

    
    concatenated1 = concatenate([main_input1,xshared])
    concatenated2 = concatenate([main_input2,xshared])

    xprivate1 = Dense(nb_n2, activation='sigmoid')(concatenated1)
    xprivate2= Dense(nb_n2, activation='sigmoid')(concatenated2)
    #xcommun= Dense(50,activation='sigmoid')(concatenated)
    #xafter= Dense(25,activation='sigmoid')(xcommun)
    
    
    #hyper0 = Dense(5, activation='sigmoid')(concatenated_hyper)
    out1 = Dense(1, activation='sigmoid', name='task1_output')(xprivate1)
    #hyper=Dropout(0.5)(hyper)

    
     
                

    #coord0 = Dense(5, activation='sigmoid')(concatenated_coord)
    out2 = Dense(1, activation='sigmoid', name='task2_output')(xprivate2)
    
    
    
    model1 = Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out1])
    
    model2= Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out2])  
    



    model1.compile(optimizer='Adam', loss='binary_crossentropy')
    model2.compile(optimizer='Adam', loss='binary_crossentropy')
    return model1, model2




def ModelSharedPrivate_ALL_LSTM(nb_n1,nb_n2,is_input1,is_input2,is_shared):

    inputs = Input(shape=(input_dim,),name='main_input')
    lstm_out=200

    aux_input1=Input(shape=(10,300,),name='aux_input1')
    aux_input2=Input(shape=(10,300,),name='aux_input2')

    words1_2=Lambda(lambda x: x[:,0:600])(inputs)
    
    cos_words=Lambda(lambda x: x[:,600:601])(inputs)
    cosG=Lambda(lambda x: x[:,601:901])(inputs)
    KullG=Lambda(lambda x: x[:,901:1501])(inputs)
    paths=Lambda(lambda x: x[:,1501:2527])(inputs)
    boradcos=Lambda(lambda x: x[:,2527:2827])(inputs)
    kull=Lambda(lambda x: x[:,2827:2829])(inputs)
    kullbroad=Lambda(lambda x: x[:,2829:3429])(inputs)
    aux=[cos_words,cosG,boradcos,0,kull,KullG,kullbroad,0,paths,0]
    ind_true=[0,1,2,4,5,6,8]
   
    main_input1=Lambda(lambda x: x[:,:])(words1_2)
    main_input2=Lambda(lambda x: x[:,:])(words1_2)
    for ind in ind_true :
        if is_input1[ind] ==1 :

            main_input1=concatenate([main_input1,aux[ind]])
        if is_input2[ind] ==1 :
            main_input2=concatenate([main_input2,aux[ind]])

   

    
    if is_input1[3] :
        xcosg1=Dense(1, activation='sigmoid')(cosG)
        main_input1=concatenate([main_input1,xcosg1])

    if is_input1[7] :
        xkullg1=Dense(1, activation='sigmoid')(KullG)
        main_input1=concatenate([main_input1,xkullg1])

    if is_input2[3] :
        xcosg2=Dense(1, activation='sigmoid')(cosG)
        main_input2=concatenate([main_input2,xcosg2])

    if is_input2[7] :
        xkullg2=Dense(1, activation='sigmoid')(KullG)
        main_input2=concatenate([main_input1,xkullg2])


    if is_input1[9]:
        lstm_layer1_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_1=average([lstm_layer1_1,lstm_layer2_1])
        main_input1=concatenate([main_input1,lstm_layer_global_1])


    if is_input2[9]:
        lstm_layer1_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_2=average([lstm_layer1_2,lstm_layer2_2])
        main_input2=concatenate([main_input2,lstm_layer_global_2])





    shared_input=Lambda(lambda x: x[:,:])(words1_2)
    for ind in ind_true :
        if is_shared[ind] ==1 :
            shared_input=concatenate([shared_input,aux[ind]])

    if is_shared[3] :
        xcosgS=Dense(1, activation='sigmoid')(cosG)
        shared_input=concatenate([shared_input,xcosgS])

    if is_shared[7] :
        xkullgS=Dense(1, activation='sigmoid')(Kull)
        shared_input=concatenate([shared_input,xkullgS])


    if is_shared[9]:
        lstm_layer1_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_s=average([lstm_layer1_s,lstm_layer2_s])
        shared_input=concatenate([shared_input,lstm_layer_global_s])



    xshared = Dense(nb_n1, activation='sigmoid')(shared_input)
   

    
    concatenated1 = concatenate([main_input1,xshared])
    concatenated2 = concatenate([main_input2,xshared])

    xprivate1 = Dense(nb_n2, activation='sigmoid')(concatenated1)
    xprivate2= Dense(nb_n2, activation='sigmoid')(concatenated2)
    #xcommun= Dense(50,activation='sigmoid')(concatenated)
    #xafter= Dense(25,activation='sigmoid')(xcommun)
    
    
    #hyper0 = Dense(5, activation='sigmoid')(concatenated_hyper)
    out1 = Dense(1, activation='sigmoid', name='task1_output')(xprivate1)
    #hyper=Dropout(0.5)(hyper)

    
     
                

    #coord0 = Dense(5, activation='sigmoid')(concatenated_coord)
    out2 = Dense(1, activation='sigmoid', name='task2_output')(xprivate2)
    
    
    
    model1 = Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out1])
    
    model2= Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out2])  
    



    model1.compile(optimizer='Adam', loss='binary_crossentropy')
    model2.compile(optimizer='Adam', loss='binary_crossentropy')
    return model1, model2




def ModelSharedPrivate_Attention_Private_Shared(nb_n1,nb_n2,is_input1,is_input2,is_shared):

    inputs = Input(shape=(input_dim,),name='main_input')
    lstm_out=200

    aux_input1=Input(shape=(10,300,),name='aux_input1')
    aux_input2=Input(shape=(10,300,),name='aux_input2')

    words1_2=Lambda(lambda x: x[:,0:600])(inputs)
    
    cos_words=Lambda(lambda x: x[:,600:601])(inputs)
    cosG=Lambda(lambda x: x[:,601:901])(inputs)
    KullG=Lambda(lambda x: x[:,901:1501])(inputs)
    paths=Lambda(lambda x: x[:,1501:2527])(inputs)
    boradcos=Lambda(lambda x: x[:,2527:2827])(inputs)
    kull=Lambda(lambda x: x[:,2827:2829])(inputs)
    kullbroad=Lambda(lambda x: x[:,2829:3429])(inputs)
    aux=[cos_words,cosG,boradcos,0,kull,KullG,kullbroad,0,paths,0]
    ind_true=[0,1,2,4,5,6,8]
   
    main_input1=Lambda(lambda x: x[:,:])(words1_2)
    main_input2=Lambda(lambda x: x[:,:])(words1_2)
    for ind in ind_true :
        if is_input1[ind] ==1 :

            main_input1=concatenate([main_input1,aux[ind]])
        if is_input2[ind] ==1 :
            main_input2=concatenate([main_input2,aux[ind]])

   

    
    if is_input1[3] :
        xcosg1=Dense(1, activation='sigmoid')(cosG)
        main_input1=concatenate([main_input1,xcosg1])

    if is_input1[7] :
        xkullg1=Dense(1, activation='sigmoid')(KullG)
        main_input1=concatenate([main_input1,xkullg1])

    if is_input2[3] :
        xcosg2=Dense(1, activation='sigmoid')(cosG)
        main_input2=concatenate([main_input2,xcosg2])

    if is_input2[7] :
        xkullg2=Dense(1, activation='sigmoid')(KullG)
        main_input2=concatenate([main_input1,xkullg2])


    if is_input1[9]:
        lstm_layer1_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_1=average([lstm_layer1_1,lstm_layer2_1])
        main_input1=concatenate([main_input1,lstm_layer_global_1])


    if is_input2[9]:
        lstm_layer1_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_2=average([lstm_layer1_2,lstm_layer2_2])
        main_input2=concatenate([main_input2,lstm_layer_global_2])





    shared_input=Lambda(lambda x: x[:,:])(words1_2)
    for ind in ind_true :
        if is_shared[ind] ==1 :
            shared_input=concatenate([shared_input,aux[ind]])

    if is_shared[3] :
        xcosgS=Dense(1, activation='sigmoid')(cosG)
        shared_input=concatenate([shared_input,xcosgS])

    if is_shared[7] :
        xkullgS=Dense(1, activation='sigmoid')(KullG)
        shared_input=concatenate([shared_input,xkullgS])


    if is_shared[9]:
        lstm_layer1_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
        lstm_layer2_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
        lstm_layer_global_s=average([lstm_layer1_s,lstm_layer2_s])
        shared_input=concatenate([shared_input,lstm_layer_global_s])



    xshared = Dense(nb_n1, activation='sigmoid')(shared_input)
   
    auxSP1=[main_input1,xshared]
    auxSP2=[main_input2,xshared]

    att_wSP1 =Dense(2, activation='sigmoid')(concatenate(auxSP1))  

    att_wSP2 =Dense(2, activation='sigmoid')(concatenate(auxSP2))

    auxSP1_bis=[]
    auxSP2_bis=[]

    for i in range(2):
        attsp1=Lambda(lambda x: x[:,i:i+1])(att_wSP1)
        attsp2=Lambda(lambda x: x[:,i:i+1])(att_wSP2)

        x_inputSP1=Lambda(lambda x: x*attsp1)(auxSP1[i])

        x_inputSP2=Lambda(lambda x: x*attsp2)(auxSP2[i])
     
        auxSP1_bis.append(x_inputSP1)
        auxSP2_bis.append(x_inputSP2)
  


    concatenated1 = concatenate(auxSP1_bis)
    concatenated2 = concatenate(auxSP2_bis)


    xprivate1 = Dense(nb_n2, activation='sigmoid')(concatenated1)
    xprivate2= Dense(nb_n2, activation='sigmoid')(concatenated2)
    #xcommun= Dense(50,activation='sigmoid')(concatenated)
    #xafter= Dense(25,activation='sigmoid')(xcommun)
    
    
    #hyper0 = Dense(5, activation='sigmoid')(concatenated_hyper)
    out1 = Dense(1, activation='sigmoid', name='task1_output')(xprivate1)
    #hyper=Dropout(0.5)(hyper)

    
     
                

    #coord0 = Dense(5, activation='sigmoid')(concatenated_coord)
    out2 = Dense(1, activation='sigmoid', name='task2_output')(xprivate2)
    
    
    
    model1 = Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out1])
    
    model2= Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out2])  
    



    model1.compile(optimizer='Adam', loss='binary_crossentropy')
    model2.compile(optimizer='Adam', loss='binary_crossentropy')
    return model1, model2





def ModelSharedPrivate_ALL_Fusion(nb_n1,nb_n2,is_input1,is_input2,is_shared):

    inputs = Input(shape=(input_dim,),name='main_input')
    lstm_out=200

    aux_input1=Input(shape=(10,300,),name='aux_input1')
    aux_input2=Input(shape=(10,300,),name='aux_input2')

    words1_2=Lambda(lambda x: x[:,0:600])(inputs)
    
    cos_words=Lambda(lambda x: x[:,600:601])(inputs)
    cosG=Lambda(lambda x: x[:,601:901])(inputs)
    KullG=Lambda(lambda x: x[:,901:1501])(inputs)
    paths=Lambda(lambda x: x[:,1501:2527])(inputs)
    boradcos=Lambda(lambda x: x[:,2527:2827])(inputs)
    kull=Lambda(lambda x: x[:,2827:2829])(inputs)
    kullbroad=Lambda(lambda x: x[:,2829:3429])(inputs)
    

    xcosg1=Dense(1, activation='sigmoid')(cosG)

    xkullg1=Dense(1, activation='sigmoid')(KullG)
 
    xcosg2=Dense(1, activation='sigmoid')(cosG)
 
    xkullg2=Dense(1, activation='sigmoid')(KullG)
   
    lstm_layer1_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
    lstm_layer2_1=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
    lstm_layer_global_1=average([lstm_layer1_1,lstm_layer2_1])

 
    lstm_layer1_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
    lstm_layer2_2=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
    lstm_layer_global_2=average([lstm_layer1_2,lstm_layer2_2])


    
    aux1=[words1_2,cos_words,cosG,boradcos,xcosg1,kull,KullG,kullbroad,xkullg1,paths,lstm_layer_global_1]
    aux2=[words1_2,cos_words,cosG,boradcos,xcosg2,kull,KullG,kullbroad,xkullg2,paths,lstm_layer_global_2]
    dims=[600,1,300,300,1,2,600,600,1,1026,400]

    att_w1=Dense(11, activation='sigmoid')(concatenate(aux1))
    att_w2=Dense(11, activation='sigmoid')(concatenate(aux2))
    aux1_bis,aux2_bis=[],[]

    for i in range(11):
        att1=Lambda(lambda x: x[:,i:i+1])(att_w1)
        att2=Lambda(lambda x: x[:,i:i+1])(att_w2)
        aux_att1=[]
        aux_att2=[]
        if dims[i]>1:
            for j in range(dims[i]):
                aux_att1.append(att1)
                aux_att2.append(att2)
            att_vector1=concatenate(aux_att1)
            att_vector2=concatenate(aux_att2)
        else : 
            att_vector1=att1
            att_vector2=att2


        x_input1=multiply([att_vector1,aux1[i]])
        x_input2=multiply([att_vector2,aux2[i]])
        aux1_bis.append(x_input1)
        aux2_bis.append(x_input2)

    main_input1=concatenate(aux1_bis)
    main_input2=concatenate(aux2_bis)


    #for ind in ind_true :
    #    if is_shared[ind] ==1 :
    #        shared_input=concatenate([shared_input,aux[ind]])

  
    xcosgS=Dense(1, activation='sigmoid')(cosG)
    

    
    xkullgS=Dense(1, activation='sigmoid')(KullG)
  


   
    lstm_layer1_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input1)
    lstm_layer2_s=Bidirectional(LSTM(lstm_out, activation='tanh'))(aux_input2)
    lstm_layer_global_s=average([lstm_layer1_s,lstm_layer2_s])


    auxS=[words1_2,cos_words,cosG,boradcos,xcosgS,kull,KullG,kullbroad,xkullgS,paths,lstm_layer_global_s]

    att_wS=Dense(11, activation='sigmoid')(concatenate(auxS))
    auxS_bis=[]

    for i in range(11):
        attS=Lambda(lambda x: x[:,i:i+1])(att_wS)
        aux_attS=[]
        if dims[i]>1:
            for j in range(dims[i]):
                aux_attS.append(attS)
            att_vectorS=concatenate(aux_attS)
        else : att_vectorS=attS
        x_inputS=multiply([att_vectorS,auxS[i]])

        auxS_bis.append(x_inputS)
  
    shared_input=concatenate(auxS_bis)

    xshared = Dense(nb_n1, activation='sigmoid')(shared_input)
   

    
    
    auxSP1=[main_input1,xshared]
    auxSP2=[main_input2,xshared]

    att_wSP1 =Dense(2, activation='sigmoid')(concatenate(auxSP1))  

    att_wSP2 =Dense(2, activation='sigmoid')(concatenate(auxSP2))

    auxSP1_bis=[]
    auxSP2_bis=[]

    for i in range(2):
        attsp1=Lambda(lambda x: x[:,i:i+1])(att_wSP1)
        attsp2=Lambda(lambda x: x[:,i:i+1])(att_wSP2)
        aux_sp1=[]
        aux_sp2=[]
        if i==0:
            for j in range(3831):
                aux_sp1.append(attsp1)
                aux_sp2.append(attsp2)
        elif i==1 :
            for j in range(int(nb_n1)):
                aux_sp1.append(attsp1)
                aux_sp2.append(attsp2)
        att_vectorSP1=concatenate(aux_sp1)
        att_vectorSP2=concatenate(aux_sp2)


        x_inputSP1=multiply([att_vectorSP1,auxSP1[i]])
        x_inputSP2=multiply([att_vectorSP2,auxSP2[i]])

     
        auxSP1_bis.append(x_inputSP1)
        auxSP2_bis.append(x_inputSP2)
  


    concatenated1 = concatenate(auxSP1_bis)
    concatenated2 = concatenate(auxSP2_bis)

    xprivate1 = Dense(nb_n2, activation='sigmoid')(concatenated1)
    xprivate2= Dense(nb_n2, activation='sigmoid')(concatenated2)
    #xcommun= Dense(50,activation='sigmoid')(concatenated)
    #xafter= Dense(25,activation='sigmoid')(xcommun)
    
    
    #hyper0 = Dense(5, activation='sigmoid')(concatenated_hyper)
    out1 = Dense(1, activation='sigmoid', name='task1_output')(xprivate1)
    #hyper=Dropout(0.5)(hyper)

    
     
                

    #coord0 = Dense(5, activation='sigmoid')(concatenated_coord)
    out2 = Dense(1, activation='sigmoid', name='task2_output')(xprivate2)
    
    
    
    model1 = Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out1])
    
    model2= Model(inputs=[inputs,aux_input1,aux_input2], outputs=[out2])  
    



    model1.compile(optimizer='Adam', loss='binary_crossentropy')
    model2.compile(optimizer='Adam', loss='binary_crossentropy')
    return model1, model2





def train_model(data,name_model,Multi,is_input,is_shared,seuil=0.5):
    tmps1=time.time()

    nb_n2=['50']

    #nb_n1=['100','200']
    nb_n1=['100','200']

    #nb_n2=['2']
    #nb_n1=['1']
    


    acc_scores1,mif_scores1, acc_scores2,  mif_scores2= [], [], [], []
    precision_score1,precision_score2,recall_score1,recall_score2=[],[],[],[]
    inputs_test_T1=[data["Task1-Random"]["x_test"],data["Task1-Random"]["x_test_P1"],data["Task1-Random"]["x_test_P2"]]
    inputs_test_T2=[data["Task2-Random"]["x_test"],data["Task2-Random"]["x_test_P1"],data["Task2-Random"]["x_test_P2"]]
    inputs_valid_T1=[data["Task1-Random"]["x_valid"],data["Task1-Random"]["x_valid_P1"],data["Task1-Random"]["x_valid_P2"]]
    inputs_valid_T2=[data["Task2-Random"]["x_valid"],data["Task2-Random"]["x_valid_P1"],data["Task2-Random"]["x_valid_P2"]]
    for (n1,n2) in list(itertools.product(nb_n1,nb_n2)):
        if Multi==False :
            model1 = name_model(n1,n2,is_input)
            model2 = name_model(n1,n2,is_input)

        else : model1, model2 = name_model(n1,n2,is_input[0],is_input[1],is_shared)
        for epoch in range(num_epochs):


            a1=len(data["Task1-Random"]["x_train"])
            a2=len(data["Task2-Random"]["x_train"])
            

            idx1 = np.random.choice(np.arange(a1), batch, replace=False)
            idx2 = np.random.choice(np.arange(a2), batch, replace=False)

            inputs_train_T1=[data["Task1-Random"]["x_train"][idx1],data["Task1-Random"]["x_train_P1"][idx1],data["Task1-Random"]["x_train_P2"][idx1]]
            inputs_train_T2=[data["Task2-Random"]["x_train"][idx2],data["Task2-Random"]["x_train_P1"][idx2],data["Task2-Random"]["x_train_P2"][idx2]]
           
            model1.fit(inputs_train_T1, np.array(data["Task1-Random"]["y_train"])[idx1], epochs=1, validation_data=None, verbose=False, )
            model2.fit(inputs_train_T2, np.array(data["Task2-Random"]["y_train"])[idx2], epochs=1, validation_data=None, verbose=False, )
            if epoch%100==0 :
                preds1 = (model1.predict(inputs_test_T1, verbose=0)> seuil).astype(int)
                preds2 = (model2.predict(inputs_test_T2, verbose=0)> seuil).astype(int)

                preds1_valid = (model1.predict(inputs_valid_T1, verbose=0)> seuil).astype(int)
                preds2_valid = (model2.predict(inputs_valid_T2, verbose=0)> seuil).astype(int)

                acc_scores1.append([accuracy_score(data["Task1-Random"]["y_valid"], preds1_valid), 
                                    accuracy_score(data["Task1-Random"]["y_test"], preds1)])
                precision_score1.append([precision_score(data["Task1-Random"]["y_valid"], preds1_valid), 
                                         precision_score(data["Task1-Random"]["y_test"], preds1)])
                recall_score1.append([recall_score(data["Task1-Random"]["y_valid"], preds1_valid),
                                       recall_score(data["Task1-Random"]["y_test"], preds1)])
                
                
                acc_scores2.append([accuracy_score(data["Task2-Random"]["y_valid"], preds2_valid),
                                    accuracy_score(data["Task2-Random"]["y_test"], preds2)])
                precision_score2.append([precision_score(data["Task2-Random"]["y_valid"], preds2_valid), 
                                         precision_score(data["Task2-Random"]["y_test"], preds2)])
                recall_score2.append([recall_score(data["Task2-Random"]["y_valid"], preds2_valid), 
                                      recall_score(data["Task2-Random"]["y_test"], preds2)])

                mif_scores1.append([f1_score(data["Task1-Random"]["y_valid"], preds1_valid) , f1_score(data["Task1-Random"]["y_test"], preds1)])
                mif_scores2.append([f1_score(data["Task2-Random"]["y_valid"], preds2_valid) , f1_score(data["Task2-Random"]["y_test"], preds2)])

    df1=pd.DataFrame(acc_scores1, columns = ['acc1_valid', 'acc1_test'])
    df2=pd.DataFrame(mif_scores1, columns = ['mif1_valid', 'mif1_test'])
    df3=pd.DataFrame(acc_scores2, columns = ['acc2_valid', 'acc2_test'])
    df4=pd.DataFrame(mif_scores2, columns = ['mif2_valid', 'mif2_test'])
    df5=pd.DataFrame(precision_score1, columns = ['prec1_valid', 'prec1_test'])
    df6=pd.DataFrame(precision_score2, columns = ['pre2_valid', 'prec2_test'])
    df7=pd.DataFrame(recall_score1, columns = ['recall1_valid', 'recall1_test'])
    df8=pd.DataFrame(recall_score2, columns = ['recall2_valid', 'recall2_test'])
    
    frames=[df1,df2,df3,df4,df5,df6,df7,df8]
    
    df_final=pd.concat(frames,axis=1)
    
    ind1=np.argmax(df_final['mif1_valid'])
    ind2=np.argmax(df_final['mif2_valid'])
    
    acc1=df_final['acc1_test'][ind1]
    acc2=df_final['acc2_test'][ind2]
    
    mif1=df_final['mif1_test'][ind1]
    mif2=df_final['mif2_test'][ind2]
    
    prec1=df_final['prec1_test'][ind1]
    prec2=df_final['prec2_test'][ind2]
    
    recall1=df_final['recall1_test'][ind1]
    recall2=df_final['recall2_test'][ind2]
   
    tmps2=time.time()-tmps1
    tmps2=tmps2/60
    #new_data=[[acc_coord,'Nan'],[acc_hyper,'Nan'],[mif_coord,'Nan'],[mif_coord,'Nan'],[prec_coord,'Nan'],[prec_hyper,'Nan'],[recall_coord,'Nan'],[recall_hyper,'Nan'],[tmps2,'Nan']]
    new_data=[acc1,mif1,prec1,recall1,acc2,mif2,prec2,recall2,tmps2]
    df_restit=pd.DataFrame(new_data,index=['acc1','f1_1','prec1','recall1','acc2','f1_2','prec2','recall2','time'])
    tmps2=time.time()-tmps1
    tmps2=tmps2/60
    #print ("Temps d'execution = %f" %tmps2 + "min")
  
    return df_restit.transpose()

