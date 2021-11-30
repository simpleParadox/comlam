import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import glob
import regex as re

def isNaN(string):
    return string != string

def time_correction(func_r1):
    
    func_r1 = func_r1[[func_r1.columns[0],'SilenceSlide.OnsetTime','StimulusSlide.OnsetTime','SentType']]
    
    x_a = []
    for idx,val in enumerate(func_r1['SilenceSlide.OnsetTime']):
        if math.isnan(val):
            x_a.append(func_r1['StimulusSlide.OnsetTime'][idx])
        else:
            x_a.append(func_r1['SilenceSlide.OnsetTime'][idx])


    func_r1['AllOnsets'] = x_a

    actual_durations_a = []

    for idx,val in enumerate(func_r1['AllOnsets']):
        if idx < 154:
            c = func_r1['AllOnsets'][idx+1] - func_r1['AllOnsets'][idx]
        else:
            if(isNaN(func_r1['SentType'][idx])):
                c = 500
            else:
                c = 4000
        actual_durations_a.append(c)  


    func_r1['ActualDurations'] = actual_durations_a


    adjusted_onset_a = [0]

    for i in range(1,len(func_r1)):
        c = adjusted_onset_a[i-1] + func_r1['ActualDurations'][i-1]
        adjusted_onset_a.append(c)



    func_r1['AdjustedOnset'] = adjusted_onset_a


    func_r1['Adds'] = (func_r1['AdjustedOnset'].astype(float))/1000

    
    func_r1['Seconds'] = (func_r1['AllOnsets'].astype(float))/1000


    for i,value in enumerate(func_r1['AdjustedOnset']):
        func_r1['Adds'][i] = float("{0:.3f}".format(((float(value))/1000)))


        
    for i,value in enumerate(func_r1['AllOnsets']):
        func_r1['Seconds'][i] = float("{0:.3f}".format(((float(value))/1000)))



    #func_r1_adj = func_r1[[func_r1.columns[0],"SentType","Adds"]]
    #func_r2_adj = func_r2[[func_r2.columns[0],"SentType","Adds"]]
    
    #func_r2_adj = func_r2_adj.rename(columns={func_r2.columns[0]: func_r1.columns[0]})
    
    #func_adj = pd.concat([func_r1_adj,func_r2_adj],ignore_index=True)
    
    for i,stim in enumerate(func_r1[func_r1.columns[0]]):
        if not isNaN(func_r1['SentType'][i]):
            func_r1[func_r1.columns[0]][i] = stim.replace(".wav","")
    
    return func_r1[[func_r1.columns[0],'SentType','Adds']]

nii = pd.DataFrame()
nii["nifti"] = np.zeros(205)
nii["nifti_from_time"] = np.zeros(205)
nii["nifti_to_time"] = np.zeros(205)

for idx,val in enumerate(nii["nifti"]):
    if idx == 0:
        nii["nifti_from_time"][idx] = 0
    else:
        nii["nifti_from_time"][idx] = nii["nifti_from_time"][idx-1]+ 2

    if idx == 0:
        nii["nifti_to_time"][idx] = 2
    else:
        nii["nifti_to_time"][idx] = nii["nifti_to_time"][idx-1]+ 2
#nifti index (0-647)
nii["nifti"][0] = 0
for i in range(1,len(nii["nifti"])):
    nii["nifti"][i] = nii["nifti"][i-1]+1
    
def align_niftis(df):
    
    df["from_nifti"] = np.zeros(len(df))
    df["to_nifti"] = np.zeros(len(df))

    for i in range(len(df)):
        for j in range(len(nii["nifti"])):
            if nii["nifti_from_time"][j] <= df["Adds"][i]:
                from_nii = nii["nifti"][j]
                #largest_start_time = nii["nifti_from_time"][j]
                #diff = nii["nifti_to_time"][j] - df["Adds"][i]
                #if diff < .5 and diff > 0 and j!= 207 and not isNaN(df['SentType'][idx]):
                    #print(i, diff)
                    #from_nii = nii["nifti"][j+1]
                #else:
                    #from_nii = nii["nifti"][j]
                #print("from nifti",from_nii)

            if i!=154:
                if nii["nifti_to_time"][j] > df["Adds"][i+1]:
                    ##smallest_end_time = nii["nifti_to_time"][j]
                    to_nii = nii["nifti"][j]
                    ##print("from nifti",from_nii,"to nifti",to_nii)
                    break

        df["from_nifti"][i] = int(from_nii)
        df["to_nifti"][i] = int(to_nii)

    for i in range(len(df)):
        if not isNaN(df['SentType'][i]):
            df['from_nifti'][i] = df['from_nifti'][i] + 1
            df['to_nifti'][i] = df['from_nifti'][i] + 1

    #df_a["from_nifti"][-1:] = df_a["to_nifti"][-2:-1]
    #df_a["to_nifti"][-1:] = 647

    df["to_nifti"] = df["to_nifti"].apply(np.int64)
    df["from_nifti"] = df["from_nifti"].apply(np.int64)
    
    return df

def mean_voxel_sub(fp_a):
    np_a = img.get_data(fp_a)
    
    np_a = np.transpose(np_a,(3,0,1,2))
    
    np_a_rs = np_a.reshape(np_a.shape[0],-1)
    
    mean_voxels_a = np.mean(np_a_rs, axis = 0)
    
    for i,row in enumerate(np_a_rs):
        np_a_rs[i] = np_a_rs[i] - mean_voxels_a
        
    return np_a_rs

def split_conditions(df):
    df_LT = df[df["SentType"] == "LT"]
    df_LF = df[df["SentType"] == "LF"]
    df_M  = df[df["SentType"] == "M"]
    df_SM = df[df["SentType"] == "SM"]
    
    df_LT = df_LT.reset_index()
    df_LF = df_LF.reset_index()
    df_M = df_M.reset_index()
    df_SM = df_SM.reset_index()
    
    return df_LT,df_LF,df_M,df_SM

def split_conditions_wo_reset(df):
    df_LT = df[df["SentType"] == "LT"]
    df_LF = df[df["SentType"] == "LF"]
    df_M  = df[df["SentType"] == "M"]
    df_SM = df[df["SentType"] == "SM"]
    
    return df_LT,df_LF,df_M,df_SM

#function to create phrase, category, exemplar

def create_categories(df):
    
    df['phrase'] = np.array(['aa' for _ in range(len(df))])
    df['category'] = np.array(['aa' for _ in range(len(df))])
    df['exemplar'] = np.array(['aa' for _ in range(len(df))])

    for i,sentence in enumerate(df[df.columns[0]]):
        if not isNaN(df['SentType'][i]):
            phrase = sentence.replace('Some', '')
            phrase = phrase.replace('are', '')
            df['phrase'][i] = phrase
            wordlist = df['phrase'][i].split(' ')
            df['category'][i] = wordlist[1]
            df['exemplar'][i] = wordlist[3]
            
    return df

def reorder_func(df,np_func):
    func_ordered = list()

    for i in range(len(df)):
        from_nii = int(df["from_nifti"][i])
        to_nii = int(df["to_nifti"][i])
        
        func = []
        nii_file_index = list(range(from_nii,to_nii+1))

        for idx,val in enumerate(nii_file_index):
            func.append(np_func[val])
        func = np.array(func)
        func_ordered.append(func)
    func_ordered = np.array(func_ordered)
    return func_ordered

def alignment(df, func):
    np_func = mean_voxel_sub(func)
    df = time_correction(df)
    df = align_niftis(df)
    df = create_categories(df)
    
    func = reorder_func(df, np_func)
    func_reordered = [x[0] for x in func]
    df['func'] = func_reordered
    
    return df

def glux_concat(df1, df2):
    
    df1 = df1[[df1.columns[1],'SentType','category','func']]
    df2 = df2[[df2.columns[1],'SentType','category','func']]
    
    df2 = df2.rename(columns={df2.columns[0]:df1.columns[0]})
    
    df = pd.concat([df1, df2], ignore_index = True)
    return df

def split_by_condition(df1, df2):
    df1_LT, df1_LF, df1_M, df1_SM = split_conditions(df1)
    df2_LT, df2_LF, df2_M, df2_SM = split_conditions(df2)
    
    LT = glux_concat(df1_LT,df2_LT)
    LF = glux_concat(df1_LF,df2_LF)
    M = glux_concat(df1_M, df2_M)
    SM = glux_concat(df1_SM, df2_SM)
    return LT, LF, M, SM

def reorder_by_category(df, dn):
    
    df_reord = pd.DataFrame()
    
    for word in dn:
        for i in range(len(df)):
            if df['category'][i] == word:
                entry = df.loc[i]
                df_reord = df_reord.append([entry])
    return df_reord

def wrapper(df1, func1, df2, func2):
    df1 = alignment(df1, func1)
    df2 = alignment(df2, func2)
    lt, lf, m, sm = split_by_condition(df1, df2)
    
    #load dendogram order
    with open('lt_dn.pkl','rb') as f:
        lt_dn = pk.load(f)
    with open('lf_dn.pkl','rb') as f:
        lf_dn = pk.load(f)
    with open('sm_dn.pkl','rb') as f:
        sm_dn = pk.load(f)
    with open('m_dn.pkl','rb') as f:
        m_dn = pk.load(f)
        
    LT_reord = reorder_by_category(lt, lt_dn)
    LF_reord = reorder_by_category(lf, lf_dn)
    M_reord = reorder_by_category(m, m_dn)
    SM_reord = reorder_by_category(sm, sm_dn)
    
    LT = [x for x in LT_reord['func']]
    LT = np.array(LT)
    
    LF = [x for x in LF_reord['func']]
    LF = np.array(LF)
    
    M = [x for x in M_reord['func']]
    M = np.array(M)
    
    SM = [x for x in SM_reord['func']]
    SM = np.array(SM)
    
    all_func = np.concatenate((LT, LF, M, SM))

    #labels of each category
    labels_LT = np.array(LT_reord[LT_reord.columns[0]])
    labels_LF = np.array(LF_reord[LF_reord.columns[0]])
    labels_M = np.array(M_reord[M_reord.columns[0]])
    labels_SM = np.array(SM_reord[SM_reord.columns[0]])
    
    return all_func, LT, LF, M, SM, labels_LT, labels_LF, labels_M, labels_SM

root = "/home/dteodore/projects/def-afyshe-ab/dteodore/Glux/"
participants = ["P050", "P055", "P056", "P059", "P066"]

for part in participants:
    print(part)
    func_r1 = pd.read_csv(root + part + "/" + part + "_R1.csv")
    func_r2 = pd.read_csv(root + part + "/" + part + "_R2.csv")

    nii_r2 = root + part + "/" + "filtered_func_dataB.nii"
    nii_r1 = root + part + "/" + "filtered_func_dataA.nii"

    #fMRI functional data aligned condition wise according to labels

    all_func, LT, LF, M, SM, labels_LT, labels_LF, labels_M, labels_SM = wrapper(func_r1, nii_r1, func_r2, nii_r2)

    open_file = open(root + part + '/' + 'aligned_func_' + part + '.pkl', "wb")
    pk.dump(all_func, open_file)
    open_file.close()

    open_file = open(root + part + '/' + 'aligned_func_LT' + part + '.pkl', "wb")
    pk.dump(LT, open_file)
    open_file.close()

    open_file = open(root + part + '/' + 'aligned_func_LF' + part + '.pkl', "wb")
    pk.dump(LF, open_file)
    open_file.close()

    open_file = open(root + part + '/' + 'aligned_func_M' + part + '.pkl', "wb")
    pk.dump(M, open_file)
    open_file.close()

    open_file = open(root + part + '/' + 'aligned_func_SM' + part + '.pkl', "wb")
    pk.dump(SM, open_file)
    open_file.close()
    

sent_types = [labels_LT, labels_LF, labels_M, labels_SM]
sent_name = ['LT', 'LF', 'M', 'SM']
for i in range(len(sent_types)):
    f = open(sent_name[i] + '.txt', 'w')
    for sent in sent_types[i]:
        f.write(sent + '\n')
    f.close()