import sys
import pandas as pd
import psycopg2
import sklearn
import seaborn as sns
import datetime
from sklearn import preprocessing,cross_validation,svm,metrics,tree,decomposition
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier,Perceptron,OrthogonalMatchingPursuit,RandomizedLogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt 
sns.set_style("white")

# ###load data

# set up sqlalchemy engine

def load_data(test_end_year):
    engine = create_engine('postgresql://10.10.2.10/appliedda')
    query = """SELECT * FROM c6.partial_evaluate WHERE (oldSpell_end>='2010-05-31'
        AND oldSpell_end<='{test_end_year}-12-31');""".format(test_end_year=test_end_year)
    for chunk in pd.read_sql(query,engine,chunksize=5000):
        df=df.append(chunk)
    df=pd.read_sql(query,engine)
    print(df.shape)
    return df

def create_dummies(localdf):

# list of variables that need to be turn to dummies: 'quarter_t', 'edlevel','workexp','district',
#'race','sex','rootrace','foreignbn'
#quarter_t
    localdf['t_q1']=(localdf['quarter_t']==1)
    localdf['t_q2']=(localdf['quarter_t']==2)
    localdf['t_q3']=(localdf['quarter_t']==3)
    #edlevel
    localdf['edu_hs'] = (localdf['edlevel']==1)
    localdf['edu_hsgrad'] = (localdf['edlevel']==2)
    localdf['edu_coll'] = (localdf['edlevel']==3)
    localdf['edu_collgrad'] = (localdf['edlevel']==4)
    #workexp
    localdf['work_prof'] = (localdf['workexp'] == 2)
    localdf['work_othermgr'] = (localdf['workexp'] == 3)
    localdf['work_clerical'] = (localdf['workexp'] == 4)
    localdf['work_sales'] = (localdf['workexp'] == 5)
    localdf['work_crafts'] = (localdf['workexp'] == 6)
    localdf['work_oper'] = (localdf['workexp'] == 7)
    localdf['work_service'] = (localdf['workexp'] == 8)
    localdf['work_labor'] = (localdf['workexp'] == 9)
    #district
    localdf['dist_cookcty'] = (localdf['district'] ==1)
    localdf['dist_downstate'] = (localdf['district'] ==0)
    #race from assistance case: 1,2,3
    localdf['race_1'] = (localdf['race'] == 1)
    localdf['race_2'] = (localdf['race'] == 2)
    #sex
    localdf['male'] = (localdf['sex'] == 1)
    localdf['female'] = (localdf['sex'] == 2)
    #rootrace
    localdf['hh_white'] = (localdf['rootrace'] == 1)
    localdf['hh_black'] = (localdf['rootrace'] == 2)
    localdf['hh_native'] = (localdf['rootrace'] == 3)
    localdf['hh_hispanic'] = (localdf['rootrace'] == 6)
    localdf['hh_asian'] = (localdf['rootrace'] == 7)
    #foreignbn: 0,1,2,3,4,5
    localdf['foreignbn_1'] = (localdf['foreignbn'] == 1)
    localdf['foreignbn_2'] = (localdf['foreignbn'] == 2)
    localdf['foreignbn_3'] = (localdf['foreignbn'] == 3)
    localdf['foreignbn_4'] = (localdf['foreignbn'] == 4)
    localdf['foreignbn_5'] = (localdf['foreignbn'] == 5)
    
    return localdf


def define_label_and_features(label):
    # label and feature global
    sel_label=str(label)
    sel_features=['foodstamp','tanf','spell_cancel','num_emp_tp4','wage_sum_tp4','wage_high_tp4','num_emp_tp3',
                 'wage_sum_tp3','wage_high_tp3','num_emp_tp2','wage_sum_tp2','wage_high_tp2','num_emp_tp1','wage_sum_tp1',
                 'wage_high_tp1','num_emp_tm1','wage_sum_tm1','wage_high_tm1','num_emp_tm2','wage_sum_tm2',
                 'wage_high_tm2','num_emp_tm3','wage_sum_tm3','wage_high_tm3','num_emp_tm4','wage_sum_tm4',
                 'wage_high_tm4','wage_sum_tp1t4','wage_sum_tm1t4','spell_length','n_prespells','max_spell_length',
    'min_spell_length','avg_spell_length','total_foodstamp_utlnow','total_tanf_utlnow','marstat','homeless','hh_counts',
    't_q1','t_q2','t_q3','edu_hs','edu_hsgrad','edu_coll','edu_collgrad','work_prof','work_othermgr','work_clerical',
    'work_sales','work_crafts','work_oper','work_service','work_labor','dist_cookcty','dist_downstate','race_1',
    'race_2','male','female','hh_white','hh_black','hh_native','hh_hispanic','hh_asian','foreignbn_1','foreignbn_2',
    'foreignbn_3','foreignbn_4','foreignbn_5']
    sel_features_spell=['foodstamp','tanf','spell_cancel','spell_length','n_prespells','max_spell_length',
                        'min_spell_length','avg_spell_length','total_foodstamp_utlnow','total_tanf_utlnow']
    sel_features_wage=['num_emp_tm1','wage_sum_tm1','wage_high_tm1',
                  'num_emp_tm2','wage_sum_tm2','wage_high_tm2',
                 'num_emp_tm3','wage_sum_tm3','wage_high_tm3',
                 'num_emp_tm4','wage_sum_tm4','wage_high_tm4','wage_sum_tm1t4']
    sel_features_demo=['age','marstat','homeless','hh_counts',
    't_q1','t_q2','t_q3','edu_hs','edu_hsgrad','edu_coll','edu_collgrad','work_prof','work_othermgr','work_clerical',
    'work_sales','work_crafts','work_oper','work_service','work_labor','dist_cookcty','dist_downstate','race_1',
    'race_2','male','female','hh_white','hh_black','hh_native','hh_hispanic','hh_asian','foreignbn_1','foreignbn_2',
    'foreignbn_3','foreignbn_4','foreignbn_5']
    
    return sel_label, sel_features

def test_train_split_process_scale(test_end_year,df,sel_label,sel_features):
    #create training and test set
    print("create df_training and df_testing")
    year_minus_one=test_end_year-1
    print("year is {}".format(test_end_year))
    print("year - 1 is {}".format(year_minus_one))
    
    df['oldspell_end'] = pd.to_datetime(df['oldspell_end'])
    
    df_training=df[(df['oldspell_end']>=datetime.date(2010,5,31)) & (df['oldspell_end'] <=
                                                          datetime.date(year_minus_one,12,31))]
    df_testing=df[(df['oldspell_end']>=datetime.date(2010,5,31)) & (df['oldspell_end'] <=
                                                          datetime.date(test_end_year,12,31))]
  
    print("df_training shape is {}".format(df_training.shape))
    print("df_testing shape is {}".format(df_testing.shape))
        
    print('Imputed with mean')
    df_training['age'].fillna(df_training['age'].mean(), inplace = True) 
    df_testing['age'].fillna(df_testing['age'].mean(), inplace=True)
    print("baseline: ")
    print(df_training[sel_label].value_counts(normalize=True)) 
    
    print("create np.array")
    y_train = np.array(df_training[sel_label].values,dtype='int')
    X_train = np.array(df_training[sel_features].values,dtype='float32')
    y_test= np.array(df_testing[sel_label].values,dtype='int')
    X_test = np.array(df_testing[sel_features].values,dtype='float32')
    
    #y_train=df_training[sel_label].values
    #X_train=df_training[sel_features].values
    #y_test=df_testing[sel_label].values
    #X_test=df_testing[sel_features].values
    
    print('Scaling training and testing sets')
    
    scaler=StandardScaler().fit(X_train)
    StandardScaler(with_mean=False)
    scaled_X_train=scaler.transform(X_train)
    scaled_X_test=scaler.transform(X_test)
    print ("mean of un-scaled features for X_train:{}".format(np.mean(X_train)))
    print ("mean of un-scaled features for X_test:{}".format(np.mean(X_test)))
    print ("mean of normalized features for X_train:{}".format(np.mean(scaled_X_train)))
    print ("mean of normalized features for X_test:{}".format(np.mean(scaled_X_test)))
    X_train=scaled_X_train
    X_test=scaled_X_test
    
    #scaled_X_train=empty
    #scaled_X_test=empty
    #df=empty
    
   
    return X_train,X_test,y_train,y_test

# Accuracy is the ratio of the correct predictions (both positive and negative) to all predictions. 
# $$ Accuracy = \frac{TP+TN}{TP+TN+FP+FN} $$

# Two additional metrics that are often used are **precision** and **recall**. 
# 
# Precision measures the accuracy of the classifier when it predicts an example to be positive. It is the ratio of correctly predicted positive examples to examples predicted to be positive. 
# 
# $$ Precision = \frac{TP}{TP+FP}$$
# 
# Recall measures the accuracy of the classifier to find positive examples in the data. 
# 
# $$ Recall = \frac{TP}{TP+FN} $$
# 
# By selecting different thresholds we can vary and tune the precision and recall of a given classifier. A conservative classifier (threshold 0.99) will classify a case as 1 only when it is *very sure*, leading to high precision. On the other end of the spectrum, a low threshold (e.g. 0.01) will lead to higher recall.

def plot_precision_recall(y_true,y_score):
    """
    Plot a precision recall curve
    
    Parameters
    ----------
    y_true: ls
        ground truth labels
    y_score: ls
        score output from model
    """
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true,y_score)
    plt.plot(recall_curve, precision_curve)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    auc_val = auc(recall_curve,precision_curve)
    print('AUC-PR: {0:1f}'.format(auc_val))
    plt.show()
    plt.clf()


def plot_precision_recall_n(y_true, y_prob, model_name,fname):
    """
    y_true: ls 
        ls of ground truth labels
    y_prob: ls
        ls of predic proba from model
    model_name: str
        str of model name (e.g, LR_123)
    fname: str
        name of file to write to
    """
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax1.set_ylim(0,1.05)
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax2.set_ylim(0,1.05)
    
    name = model_name
    plt.title(name)
    plt.savefig(fname)


def precision_at_k(y_true, y_scores,k):
    
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores ])
    return precision_score(y_true, y_pred)


def define_clfs_params(grid_size):
    clfs={'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
       'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB':AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm='SAMME',n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM':svm.SVC(kernel='linear',probability='true',random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6,n_estimators=10),
        'NB': GaussianNB(),
        'DT':DecisionTreeClassifier(),
        'SGD':SGDClassifier(loss='hinge',penalty='l2'),
        'KNN':KNeighborsClassifier(n_neighbors=3)
        }
    small_grid={
        'RF':{'n_estimators':[10,100],'max_depth':[5,50],'max_features':['sqrt','log2'],'min_samples_split':[2,10]},
        'LR':{'penalty':['l1'],'C':[0.00001,0.001,0.1,1,10]},
        'SGD':{'loss':['hinge','log','perceptron'],'penalty':['l2','l1','elasticnet']},
        'ET':{'n_estimators':[10,100],'criterion':['gini','entropy'],'max_depth':[5,50],'max_features':['sqrt','log2'],
             'min_samples_split':[2,10]},
        'AB':{'algorithm':['SAMME','SAMME.R'],'n_estimators':[1,10,100,1000,10000]},
        'GB':{'n_estimators':[10,100],'learning_rate':[0.001,0.1,0.5],'subsample':[0.1,0.5,1.0],'max_depth':[5,50]},
        'NB':{},
        'DT':{'criterion':['gini','entropy'],'max_depth':[1,5,10,20,50,100],'max_features':['sqrt','log2'],
              'min_samples_split':[2,5,10]},
        'SVM':{'C':[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN':{'n_neighbors':[1,5,10,25,50,100],'weights':['uniform','distance'],
               'algorithm':['auto','ball_tree','kd_tree']}
    }
    if (grid_size=='small'):
        return clfs,small_grid


def joint_sort_descending(l1,l2):
    #l1,l2 have to be numpy arrays
    idx=np.argsort(l1)[::-1]
    return l1[idx],l2[idx]

def generate_binary_at_k(y_scores,k):
    cutoff_index=int(len(y_scores)*(k/100.0))
    test_predictions_binary=[1 if x<cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores,k):
    
    y_scores,y_true=joint_sort_descending(np.array(y_scores),np.array(y_true))
    preds_at_k=generate_binary_at_k(y_scores,k)
    precision=precision_score(y_true,preds_at_k)
    return precision


def clf_loop(models_to_run, clfs,grid,X_train,X_test,y_train,y_test,test_end_year,label):

    #results_df=pd.DataFrame(columns=('timestamp','model_type','clf','parameters','auc-roc','baseline','p_at_0.5','p_at_1','p_at_5','p_at_10','p_at_10'))
    index = 0
    for index,clf in enumerate([clfs[x] for x in models_to_run]): 
        print 'clf:', clf
        print models_to_run[index]
        clf_nm = models_to_run[index]
        parameter_values=grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                print clf
                y_pred_probs=clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
        	print('Model fitted')        
#you can also store the model, feature importances, and prediction scores
                #we're only string the metrics for now
                y_pred_probs_sorted,y_test_sorted=zip(*sorted(zip(y_pred_probs,y_test),reverse=True))
		print('scoring') 
                results_df=pd.DataFrame(columns=('timestamp','model_type','clf','parameters','auc-roc',
                                                  'baseline','p_at_0.5','p_at_1','p_at_5','p_at_10','p_at_10'))

                results_df.loc[len(results_df)]=[str(datetime.datetime.now()),
						 models_to_run[index],
						 clf,
						 p,
                                                 roc_auc_score(y_test,y_pred_probs),
                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,100),
                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,0.5),
                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                filename = "results" + "_"+str(test_end_year)+"_"+label+'.csv'
                results_df.to_csv(filename, index=True, mode='a')
                
                if NOTEBOOK==1:
                    plot_precision_recall_n(y_test,
                                            y_pred_probs,
                                            clf,
                                            fname=clf_nm+'_'+str(index)+'_'+label+'_'+str(test_end_year)+'.png')
                    index +=1
            except IndexError,e:
                print 'Error:',e
                continue
    return results_df


def main():
    test_end_year = int(sys.argv[1])
    print(test_end_year)
    label= sys.argv[2]
    print label
    #start_year = 2010
    
    df=load_data(test_end_year)
    df=create_dummies(df)
    sel_label,sel_features=define_label_and_features(label)
    X_train,X_test,y_train,y_test=test_train_split_process_scale(test_end_year,df,sel_label,sel_features)
    
    grid_size='small'
    clfs,grid=define_clfs_params(grid_size)
    models_to_run=['LR']
    #'RF','DT','KNN','ET','AB','GB','NB'
    #df=pd.read_csv("user/nnnnn.")
    #features=['','']
    #X=df[features]
    #y=df.return_1yr
    
    
    
    results_df=clf_loop(models_to_run,clfs,grid,X_train,X_test,y_train,y_test,test_end_year,label)
    if NOTEBOOK==1:
        results_df
    #filename = "results" + "_"+year+"_"+label
    #results_df.to_csv(filename, index=True)
    
    #master_results_df = pd.read_csv('results.csv')
    #master_results_df = master_results_df.append(results_df)
    #master_results_df.to_csv('results.csv')
    

NOTEBOOK=0

##if _name_=='__main__':
main()




