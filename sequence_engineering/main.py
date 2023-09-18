import numpy as np
import pandas as pd
import random
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
from operator import itemgetter 
from matplotlib import pyplot
from collections import Counter
import tqdm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from absl import flags, app
import os
import json



def main(argv):
    FLAGS = flags.FLAGS
    del argv
    
    def make_reverse(aso_seq):
        seqs_list=[]
        for ii in range(len(aso_seq)):
            vec = aso_seq[ii][::-1]
            target_seqs=[]
            for index in range(len(vec)):
                if vec[index] =='G':
                    target_seqs.append('C')
                elif vec[index] =='C':
                    target_seqs.append('G')
                elif vec[index] =='T':
                    target_seqs.append('A')
                elif vec[index] =='A':
                    target_seqs.append('T')
            target_seqs="".join(target_seqs)
            seqs_list.append(target_seqs)
        return seqs_list


    def objective_func(target_gibbs, others_gibbs, target_folds, vec_a, ind):  

        obj_ftn =  vec_a[1] * target_gibbs + vec_a[2] * others_gibbs +  vec_a[3] * (1-(0.7-target_folds)**2) +  vec_a[0]

        rank = np.where(np.argsort(obj_ftn, axis=None)[::-1]==ind)[0]

        return obj_ftn,rank


    def check_rank(seq,vec_a):

        search_len=gibbs_target.shape[0]
        ranks=[]
        for uu in range(len(seq)):
            score,rank=objective_func(gibbs_target[:search_len],gibbs_off_target_top_k[:search_len],folds_target[:search_len],vec_a,seq[uu])    
            ranks.append(rank[0])    
        return ranks, score


    def print_topk(lists,num,delta=3):
        seqs=[]
        for ii in range(num):
            seqs.append(lists[0])

            baseline=np.arange(lists[0]-delta+1,lists[0]+delta)

            lists = np.array([ele for ele in lists if ele not in baseline])

        seqs = np.array(seqs)
        return seqs
    
    sc_path= 'patent_experiments/'
    target_DB = pd.read_csv(sc_path+'aso_features_patent.csv') # Experiments and our features ('gibbs','off_gibbs','secondary_structure')

    n_length = FLAGS.seq_len 
    topk=FLAGS.topk 
    Input_path = './features'
    if not os.path.exists(Input_path): os.mkdir(Input_path)

    gibbs_target_0 = np.load(Input_path+'/ido_target_gibbs_0.npy')
    folds_target_0 = np.load(Input_path+'/ido_target_gibbs_folds_0.npy')
    gibbs_off_target_0 = np.load(Input_path+'/ido_off_target_gibbs_0.npy')

    gibbs_target_1 = np.load(Input_path+'/ido_target_gibbs_1.npy')
    folds_target_1 = np.load(Input_path+'/ido_target_gibbs_folds_1.npy')
    gibbs_off_target_1 = np.load(Input_path+'/ido_off_target_gibbs_1.npy')

    gibbs_target_2 = np.load(Input_path+'/ido_target_gibbs_2.npy')
    folds_target_2 = np.load(Input_path+'/ido_target_gibbs_folds_2.npy')
    gibbs_off_target_2 = np.load(Input_path+'/ido_off_target_gibbs_2.npy')

    gibbs_target_3 = np.load(Input_path+'/ido_target_gibbs_3.npy')
    folds_target_3 = np.load(Input_path+'/ido_target_gibbs_folds_3.npy')
    gibbs_off_target_3 = np.load(Input_path+'/ido_off_target_gibbs_3.npy')

    gibbs_target_4 = np.load(Input_path+'/ido_target_gibbs_4.npy')
    folds_target_4 = np.load(Input_path+'/ido_target_gibbs_folds_4.npy')
    gibbs_off_target_4 = np.load(Input_path+'/ido_off_target_gibbs_4.npy')

    gibbs_target_5 = np.load(Input_path+'/ido_target_gibbs_5.npy')
    folds_target_5 = np.load(Input_path+'/ido_target_gibbs_folds_5.npy')
    gibbs_off_target_5 = np.load(Input_path+'/ido_off_target_gibbs_5.npy')


    gibbs_target = np.load(Input_path+'/ido_target_gibbs_length19.npy')

    folds_target=np.concatenate([folds_target_0, folds_target_1, folds_target_2, folds_target_3,folds_target_4,folds_target_5])
    gibbs_off_target=np.concatenate([gibbs_off_target_0,gibbs_off_target_1, gibbs_off_target_2, gibbs_off_target_3,gibbs_off_target_4,gibbs_off_target_5],axis=0)
    gibbs_off_target.sort(axis=1)

    df_ref =  pd.read_csv('mrna_dataset/ido1_information.csv')
    ido=str(list(df_ref[df_ref['Genbank_accession']=='NM_002164']['sequence'].values))[2:-2] ## ido gene sequence
    ido_len = len(ido)

    checkpoint_dir = './outputs/{}'.format(FLAGS.out_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if FLAGS.mode == 'train':
        elements=['gibbs','off_gibbs','secondary_structure'] 

        X = target_DB[elements]
        Y = target_DB['Inhibition rate'].values
        mask = ~target_DB['Inhibition rate'].isna()

        X = X[mask]

        Y = Y[mask]
        Y = 1-Y

        X_train=X
        y_train=Y
        X_test=X
        y_test=Y
        
        reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_train_pred = reg.predict(X_train)
        coef = reg.coef_/reg.coef_[0]

        r_squared = r2_score(y_test, y_pred)
        
        print('optimal setting: a*=[{:.4f},{:.4f},{:.4f},{:.4f}]^T'.format(reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2]))
        a_star=[reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2]]
        
        fold_name = FLAGS.rnastructure

        with open("features/fold.json", "r") as json_file:
            loaded_data = json.load(json_file)

        folding = loaded_data[fold_name]

        fold_v1=[]
        for idx in range(len(gibbs_target)):
            fold_v1.append(folding[idx:idx+n_length])

        folds_target=[]
        for ii in range(len(fold_v1)):
            folds_target.append(fold_v1[ii].count('.')/len(fold_v1[ii]))

        folds_target = np.array(folds_target)
        gibbs_off_target_top_k =np.mean(gibbs_off_target[:,:FLAGS.topk],axis=1)

        candidate=np.arange(gibbs_target.shape[0])
        full_rank, score = check_rank(candidate,a_star) #a*
        rank_sorted = np.array(full_rank).argsort()     
        
        ours = print_topk(rank_sorted,FLAGS.num_candidates)
        
        df_save = pd.DataFrame([],columns=['rank','ASO 5-3','binding_site'])
        for ii in range(len(ours)):
            df_save.loc[ii]=[ii+1,make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]]
            if ii == 0:
                print('{}st ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))
            elif ii==1:
                print('{}nd ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))  
            elif ii==2:
                print('{}rd ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))   
            else:
                print('{}th ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))  
        
        df_save.to_csv(checkpoint_dir + '/top_sequences.csv', index=False)

    elif FLAGS.mode == 'eval':

        input_str = FLAGS.a_star
        float_array_elements = input_str.split()
        a_star = [float(element) for element in float_array_elements]

        
        if len(float_array_elements) != 4:
            raise ValueError("Input array must have exactly 4 elements.")

        fold_name = FLAGS.rnastructure

        with open("features/fold.json", "r") as json_file:
            loaded_data = json.load(json_file)

        folding = loaded_data[fold_name]
        
        fold_v1=[]
        for idx in range(len(gibbs_target)):
            fold_v1.append(folding[idx:idx+n_length])

        folds_target=[]
        for ii in range(len(fold_v1)):
            folds_target.append(fold_v1[ii].count('.')/len(fold_v1[ii]))

        folds_target = np.array(folds_target)
        gibbs_off_target_top_k =np.mean(gibbs_off_target[:,:FLAGS.topk],axis=1)

        candidate=np.arange(gibbs_target.shape[0])
        full_rank, score = check_rank(candidate,a_star) #a*
        rank_sorted = np.array(full_rank).argsort()     
        
        ours = print_topk(rank_sorted,FLAGS.num_candidates)
        
        df_save = pd.DataFrame([],columns=['rank','ASO 5-3','binding_site'])
        for ii in range(len(ours)):
            df_save.loc[ii]=[ii+1,make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]]
            if ii == 0:
                print('{}st ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))
            elif ii==1:
                print('{}nd ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))  
            elif ii==2:
                print('{}rd ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))   
            else:
                print('{}th ASO 5-3:{}, binding_site:{}'.format((ii+1),make_reverse([ido[ours[ii]:ours[ii]+FLAGS.seq_len].upper()])[0],ours[ii]))  
        
        df_save.to_csv(checkpoint_dir + '/top_sequences.csv', index=False)

if __name__ == '__main__':
    flags.DEFINE_string('mode', 'train', '[train or eval]')
    flags.DEFINE_string('target', 'ido1', 'target gene name')
    flags.DEFINE_string('rnastructure', 'mfold', 'rna strucuture estimation name')
    flags.DEFINE_string("a_star", "1 0 0 0", help="Input vector as a space-separated string")
    flags.DEFINE_string('out_dir', 'ido1_topk', 'Load the checkpoint files in this directory')
    flags.DEFINE_integer('seq_len', 19, 'length of ASO candidates.')
    flags.DEFINE_integer('num_candidates', 6, 'max node size.')
    flags.DEFINE_integer('topk', 50, 'number of possible off target genes')

    app.run(main)






