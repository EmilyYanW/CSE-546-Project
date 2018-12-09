
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# # Load Data

# In[2]:

filename = 'ConsUS-201810-emv-zip-98004.csv'

def load_data(filename):
    
    df = pd.read_csv(filename, header= None, low_memory = False)
    header = pd.read_csv('ConsUS-201810-header.txt', sep="\t", header=None)
    header  = list(header.loc[0]) + ['email']
    df.columns = header
    
    df1 = df.copy()
    print('number of rows: ', len(df))
    print('missing home purchase date: ', 
          (df1['HomePurchaseDate'] == df1['HomePurchaseDate']).mean())
    
    return df1
    
    


# In[3]:

df1 = load_data(filename)


# In[4]:

# process date information
# return xxxx year format
import math

def return_floor(x):
    if x == x:
        return math.floor(x/10000)
    else:
        return x


# In[5]:

def target_list(df2):
    
    df_target = df2[(df2['SellDate'] ==2018)]
    
    return df_target


# In[6]:

def deal_with_date(df1, target_year):
    
    print('start total rows: ', len(df1))
    df2 = df1[(df1['HomePurchaseDate'] ==df1['HomePurchaseDate'] )]
    print('Delete rows missing Home Purchase Date: ',
          (df1['HomePurchaseDate'] != df1['HomePurchaseDate']).sum(), 'now: ', len(df2) )
    
    # Convert Home Purchase Date to XXXX year format
    df2['HomePurchaseDate'] = df2['HomePurchaseDate'] .apply(lambda x: return_floor(x))
    
    # Convert DOB to XXXX year format
    df2['DOB']  = df2['DOB'].apply(lambda x: return_floor(x))
    
    # purchase year from now
    df2['PurchaseYearFromNow'] = 2018 - df2['HomePurchaseDate']
    
    # How old the house is from now
    df2['SinceBuiltNow'] = 2018 - df2['HomeYearBuilt']
    
    # Calculate sold date with home purchase date and length of residence
    df2['SellDate'] = df2['HomePurchaseDate'] +  df2['LengthOfResidence']
    
    # Distribution of Purchase Date
    plt.figure(figsize = (8,6))
    plt.hist(df2['HomePurchaseDate'], bins = 50)
    plt.xticks(np.arange(min(df2['HomePurchaseDate']), 2021, 2), rotation = 60)
    plt.title('Distribution over HomePurchase Date', fontsize = 15)

    

    # Delete weired sell data 
    df3 = df2[df2['SellDate'] <= 2018]
    df4 = df3[df3['LengthOfResidence'] > 0]
    
    print('delete percent of weird sell date >=2018 : ', (df2['SellDate'] >= 2018).mean(),
         'now: ', len(df3))
    
    # Distribution of Length of Residence
    plt.figure(figsize = (8,6))
    plt.hist(df3['LengthOfResidence'], bins = 10)
    plt.xlabel('Length Of Residence Years')
    plt.title('Distribution over Length Of Resdience', fontsize = 15)
    plt.show()
    
    # Distribution of age of owner

    df4['Age'].hist()
    plt.xlabel('Age')
    plt.title('Distribution over Age of Homeowner', fontsize = 15)
    plt.show()
    # Scatter of Purchase from now and length of residence

    plt.scatter(df4['PurchaseYearFromNow'], df4['LengthOfResidence'])
    plt.title('Purchase from now and Length of Residence', fontsize = 15)
    plt.xlabel('Purchase from now')
    plt.ylabel('Length Of Residence')
    
    
    return df4
 


# In[148]:

df4 = deal_with_date(df1, 6)


# # Prepare model data

# In[8]:

def prepare_model_date(df4, target_year, percent):
    
    # select columns that have columns missing percentage <= percent
    
    params = (df4.isnull().sum()> percent*len(df4))
    params = params[params == False]
    df_model = df4[params.index]

    
    # Drop identifying columns
    df_model = df_model.drop(['ST','FirstName','MiddleName',
                              'LastName','Phone','FileDate'], axis = 1)
    
    df_model[['Zip','Zip4']] = df_model[['Zip','Zip4']].astype('str')
    df_model = df_model.drop(['Address','Zip4'], axis = 1)
    
    cvs = df_model.columns[(df_model.dtypes == 'object')]
    
    print('categorical columns: ')
    print(cvs)
    
    df_model1 = df_model.copy()
    
    df_model1 = df_model1.drop(['DOB','HomeRefinanceDate','HomeYearBuilt'], axis = 1)
    for cv in cvs:
        one_hot = pd.get_dummies(df_model1[cv], prefix = cv)
        df_model1 = pd.concat([df_model1, one_hot], axis = 1)
        df_model1 = df_model1.drop([cv], axis = 1)
    
    df_model1[['HomePurchaseDate', 'HomePurchasePrice',
     'CensusMedianHomeValue','LengthOfResidence','SellDate']] = df4[['HomePurchaseDate', 
                                                                     'HomePurchasePrice',
     'CensusMedianHomeValue','LengthOfResidence','SellDate']]
    
    print('number of rows: ', len(df_model1))
    
    colname = 'Target_'+str(target_year)
    df_model1[colname ] = df_model1['LengthOfResidence'] < target_year
    
    print('percent of sold in target year houses: ', (df_model1[colname] == True).mean())
    
    # drop target columns
    
    return df_model1, colname


# In[201]:

df_model2, colname = prepare_model_date(df4, 6, 0.6)


# In[10]:

pd.set_option('max_colwidth', 500)


# In[147]:

len(df_model2)


# ## Train the model

# In[202]:

def train_model(df_model2, colname, param, num_round):
    
    print('number of rows: ', len(df_model2))
    
    print('average house price: ', df_model2['HomePurchasePrice'].mean())
    
    df_model3 = df_model2[df_model2['HomePurchasePrice'] <= 1000]
    print('rows deleted for price > 1000: remaining ', len(df_model3), 
          df_model3['HomePurchasePrice'].mean())
    
    
    df_target = df_model3[df_model3['SellDate'] == 2018]
    df_model3 = df_model3[df_model3['SellDate'] < 2018]
    
    print('prepare train-val-test split')
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df_model3.drop(colname, axis = 1), 
                                                    df_model3[colname], test_size=0.3)
    
    # Validation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
    
    # Train the model
    X_train1 = X_train.drop(['LengthOfResidence','SellDate','HomePurchaseDate',
                               'PurchaseYearFromNow', 'MortgagePurchaseDate'], axis = 1)
    
    X_val1 = X_val.drop(['LengthOfResidence','SellDate','HomePurchaseDate',
                               'PurchaseYearFromNow','MortgagePurchaseDate'], axis = 1)
    X_test1 = X_test.drop(['LengthOfResidence','SellDate','HomePurchaseDate',
                               'PurchaseYearFromNow','MortgagePurchaseDate'], axis = 1)
    
    df_target1= df_target.drop(['LengthOfResidence','SellDate','HomePurchaseDate',
                               'PurchaseYearFromNow','MortgagePurchaseDate'], axis = 1)
    
    print('train the model')
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train1, label=y_train)
    dval = xgb.DMatrix(X_val1, label = y_val)
    
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    
    bst = xgb.train(param, dtrain, num_round, evallist)
#     print(bst.feature_importances_)
# #     importance = xgb.importance( model = bst)
    importance = bst.get_fscore()
    print('analyze results')
    # Analyze Resutls
    
    # Test set
    dtest = xgb.DMatrix(X_test1)
    ypred = bst.predict(dtest)
    ypred_binary = ypred > 0.5
    
    
    # Target Set
    
    dtarget = xgb.DMatrix(df_target1.drop(colname, axis = 1), 
                          label = df_target1[colname])
    ypred_target = bst.predict(dtarget)
    df_target['ypred'] = ypred_target
    
    from sklearn.metrics import precision_recall_fscore_support
    print(precision_recall_fscore_support(y_test, ypred_binary, average='binary'))
    
    
    ## Accuracy

    Acc = (ypred_binary  == y_test).sum() / len(y_test)
    print('Accuracy: ', Acc)
    
    # Random Guessing
    
    Guess = (y_test == False).sum()/len(y_test)
    print('Random Guess: ', Guess)
    
    # ROC
    
    from sklearn.metrics import roc_curve
    fpr_rf, tpr_rf, thresh = roc_curve(y_test, ypred)


    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr_rf, tpr_rf)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    X_test['ypred'] =ypred
    X_test['y_label'] = y_test
    X_test['ypred_binary'] = ypred_binary

    X_test2 = X_test[['HomePurchasePrice','SellDate','LengthOfResidence',
                      'HomePurchaseDate','y_label','ypred_binary','ypred']]

    # Conditional Distribution

    fig = plt.figure(figsize = (8,6))
    plt.hist(X_test[X_test['y_label'] ==1 ]['ypred'], bins = 100)
    plt.hist(X_test[X_test['y_label'] ==0 ]['ypred'], bins = 100)
    plt.title('Conditional Distribution of Score', fontsize = 15)
    plt.xlabel('Score - Preditced Probability', fontsize = 15)
    plt.show()
    
    
#     Training
    
    ypred_train = bst.predict(dtrain)

    ypred_binary_trian = ypred_train> 0.5

    correct_trian = ypred_binary_trian == y_train
    
    X_train['ypred_train'] = ypred_train
    X_train['y_train'] =  y_train
    X_train['ypred_binary_trian'] = ypred_train> 0.5
    X_train['correct'] = ypred_binary_trian == y_train
    
    X_train2 = X_train[['HomePurchasePrice','SellDate','LengthOfResidence',
                      'HomePurchaseDate','y_train','ypred_binary_trian','ypred_train']]
    
    return X_test2, X_train2, df_target, X_train1, importance, df_model3,fpr_rf, tpr_rf, thresh
#     return   Acc,precision_recall_fscore_support(y_test, ypred_binary, average='binary')[0],precision_recall_fscore_support(y_test, ypred_binary, average='binary')[1]


# In[62]:

Max_depths = np.arange(4,8,1)
Nthreads = np.arange(3,8)
Objectives = ['binary:logistic', 'binary:logitraw','binary:hinge']
Eval_metrics = ['error','aucpr','auc']
Num_rounds = np.arange(10,15,1)


# In[203]:

params = {'eval_metric': 'auc',
 'max_depth':6,
 'nthread': 4,
          'eta':0.2,
          'silent':1,
 'objecive':'binary:logistic'}


# In[204]:

X_test2 , X_train2, df_target, X_train1, importance, df_model3,fpr_rf, tpr_rf, thresh = train_model(df_model2, 
                                                                             colname, params, 25)


# In[215]:

plt.scatter(df_model3['HomePurchasePrice'], df_model3['LengthOfResidence'])
plt.show()


# In[205]:

df_target[['Address','FirstName','MiddleName','Zip','Zip4','DOB',
                              'LastName','Phone','email']] = df1[['Address','FirstName','MiddleName',
                              'Zip','Zip4','DOB','LastName','Phone','email']]


# In[206]:

df_target['PredBinary'] = (df_target['ypred'] > 0.5)


# In[208]:

(df_target['PredBinary'] == df_target['Target_6']).mean()


# In[162]:

df_target['PredSellDate'] = 5-df_target['LengthOfResidence']


# In[183]:

df_target =df_target.sort_values('ypred', ascending = False).drop_duplicates('Address')


# In[184]:

len(df_target)


# In[185]:

Df_target_final = df_target[df_target['PredSellDate'] <= 3]


# In[186]:

Df_target_final_small = Df_target_final[['Address','FirstName','MiddleName','Zip4','DOB',
                              'LastName','Phone','HomePurchasePrice','HouseholdSize',
                                                   'HomePurchaseDate','LengthOfResidence','Target_6','PredSellDate','ypred']]


# In[187]:

Df_target_final_100 = Df_target_final[:100]


# In[188]:

Df_target_final_100[['Address','FirstName','MiddleName','Zip4','DOB',
                              'LastName','Phone','HomePurchasePrice','HouseholdSize',
                                                   'HomePurchaseDate','LengthOfResidence','Target_6','PredSellDate','ypred']]


# In[174]:

(Df_target_final_small[:100]['PredSellDate'] >= 0).mean()


# In[175]:

(Df_target_final_small[100:200]['PredSellDate']>=0).mean()


# In[176]:

(Df_target_final_small[200:300]['PredSellDate'] >= 0).mean()


# In[177]:

(Df_target_final_small[300:400]['PredSellDate'] >= 0).mean()


# In[178]:

(Df_target_final_small[400:500]['PredSellDate'] >= 0).mean()


# In[336]:

df_model2[params.index].to_csv('train_model_98004.csv')
df_target[params.index].to_csv('target_98004.csv')


# In[379]:

len(df_model2)


# In[382]:

df_model3 = df_model2[params.index]


# In[383]:

df_model3.columns


# In[312]:

import operator
sorted_x = sorted(importance.items(), key=operator.itemgetter(1))


# In[313]:

sorted_x 


# In[364]:

params = X_train1.nunique()<=2
params = params[params == False]
X_train1 = X_train1[params.index]


# In[377]:

len(X_train1)


# In[380]:

df_model2['Age'].hist(bins = 30, label = 'train')
df_target['Age'].hist(bins = 30, label = 'target', alpha = 0.6)
plt.legend()


# In[381]:

df_model2['HomePurchasePrice'].hist(bins = 30, label = 'train')
df_target['HomePurchasePrice'].hist(bins = 30, label = 'target', alpha = 0.6)


# In[316]:

df_target[['Address','FirstName','MiddleName','Zip','Zip4','DOB',
                              'LastName','Phone','email']] = df1[['Address','FirstName','MiddleName',
                              'Zip','Zip4','DOB','LastName','Phone','email']]


# In[317]:

df_target_small = df_target[['Address','FirstName','DOB','MiddleName','LastName',
                             'Phone','Zip','Zip4','email',
           'HomePurchasePrice', 'CensusMedianHomeValue','CensusMedianHouseholdIncome',
          'HomePurchaseDate','LengthOfResidence','ypred']]


# # Validation 

# In[406]:

df_target_small['LengthOfResidence'].hist(bins = 15)


# In[394]:

(df_target_small['LengthOfResidence'] >4).mean()


# In[403]:

(df_target_small.sort_values(by = 'ypred', ascending = False)[:100]['LengthOfResidence'] >4).mean()


# In[400]:

(df_target_small.sort_values(by = 'ypred', ascending = False))[:100]


# In[ ]:

(df_target_final['LengthOfResidence'] > 4).mean()


# In[387]:

len(df_target_small)


# In[388]:

len(df_target_score )


# In[389]:

(df_target_score['ypred'] > 0.5).mean()


# In[318]:

df_target_score = pd.DataFrame(df_target_small.groupby('Address')['ypred'].mean()).reset_index()


# In[390]:

df_target_final = df_target_small.sort_values(by = 'ypred', ascending = False).drop_duplicates('Address')


# In[320]:

df_target_final.to_csv('target_list_score.csv')


# In[322]:

df_target_final.head(100)['ypred'].min()


# In[392]:

(df_target_final['LengthOfResidence'] > 4).mean()


# In[325]:

df_target_final.head(100).sort_values('HomePurchasePrice', ascending = False)


# In[263]:

df_target_final['ypred'].hist()


# In[192]:

X_test2['correct'] = X_test2['y_label'] == X_test2['ypred_binary']


# In[194]:

len(X_test2)


# In[198]:

X_test2[(X_test2['y_label'] ==True) & (X_test2['correct'] == False)]


# In[170]:

len(X_train2)


# In[260]:

# Test Train split 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_model2.drop('Target_4', axis = 1), df_model2['Target_4'], test_size=0.3)


# In[261]:

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)


# In[262]:

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)


# In[263]:

dval = xgb.DMatrix(X_val, label = y_val)


# ### CROSS VALIDATION

# In[39]:

Max_depths = np.arange(4,8,1)
Nthreads = np.arange(3,8)
Objectives = ['binary:logistic', 'binary:logitraw','binary:hinge']
Eval_metrics = ['error','aucpr','auc']
Num_rounds = np.arange(10,15,1)


# In[40]:

params = []

for i in np.arange(0, len(Max_depths)):
    for j in np.arange(0, len(Nthreads)):
        for k in np.arange(0, len(Objectives)):
            for p in np.arange(0, len(Eval_metrics)):
                for r in np.arange(0, len(Num_rounds)):
                    
                    max_depth = Max_depths[i]
                    nthread = Nthreads[j]
                    objective = Objectives[k]
                    eval_metric = Eval_metrics[p]
                    num_rounds = Num_rounds[r]
                    
                    param = {
                        'max_depth': max_depth,
                        'objecive': objective,
                        'eval_metric': eval_metric,
                        'nthread': nthread,
                        'num_rounds': num_rounds
                    }
                    
                    params.append(param)


# In[71]:

indices = []
for obj in Objectives:
    
    for ev in Eval_metrics:
        print()
        print(obj)
        print(ev)
        index = []
        for j,param in enumerate(params):
            
            if (param['eval_metric'] == ev)  & (param['objecive'] == obj):
                
                index.append(j)
        indices.append(index)


# In[74]:

for j in np.arange(0,len(indices)):
    print()
    print(np.max([Precisions[i] for i in indices[j]]))
    print(np.max([Recalls[i] for i in indices[j]]))
    print(np.max([ACCs[i] for i in indices[j]]))
    


# In[64]:

np.max([Precisions[i] for i in indices[0]])


# In[ ]:

np.max([Precisions[i] for i in indices[0]])


# In[49]:

params[305]


# In[43]:

Precisions[305]


# In[44]:

Recalls[305]


# In[45]:

ACCs[305]


# In[34]:

Max_depths = np.arange(4,7,1)
Objectives = ['binary:logistic', 'binary:hinge']
Eval_metrics = ['error','aucpr','auc','logloss']
Num_rounds = np.arange(12,14,1)


# In[24]:

ACCs = []
Precisions = []
Recalls = []

for k in np.arange(0, len(Objectives)):
    
    for p in np.arange(0, len(Eval_metrics)):
        
        for i in np.arange(0, len(Max_depths)):
    
            for j in np.arange(0, len(Nthreads)):
                
                for r in np.arange(0, len(Num_rounds)):
                    
                
                    max_depth = Max_depths[i]
                    nthread = Nthreads[j]
                    objective = Objectives[k]
                    eval_metric = Eval_metrics[p]
                    num_rounds = Num_rounds[r]

                    param = {
                        'max_depth': max_depth,
                        'eta':1,
                        'silent':1,
                        'objecive': objective,
                        'eval_metric': eval_metric
                    }

                    results = train_model(df_model2, colname, param, num_rounds)
                    ACCs.append(results[0])
                    Precisions.append(results[1])
                    Recalls.append(results[2])


# In[37]:

np.argmax(np.array(Precisions))


# In[26]:

len(ACCs)


# In[264]:

param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'


# In[265]:

evallist = [(dval, 'eval'), (dtrain, 'train')]


# In[266]:

num_round = 15
bst = xgb.train(param, dtrain, num_round, evallist)


# In[267]:

dtest = xgb.DMatrix(X_test, label = y_test)


# In[268]:

ypred = bst.predict(dtest)


# In[269]:

ypred_train = bst.predict(dtrain)


# In[270]:

ypred_train = bst.predict(dtrain)

ypred_binary_trian = ypred_train> 0.5

correct_trian = ypred_binary_trian == y_train


# In[271]:

correct_trian = ypred_binary_trian == y_train


# In[272]:

ypred_binary = ypred > 0.5


# In[273]:

from sklearn.metrics import precision_recall_fscore_support


# In[274]:

precision_recall_fscore_support(y_test, ypred_binary, average='binary')


# In[275]:

## Accuracy

Acc = (ypred_binary  == y_test).sum() / len(y_test)


# In[276]:

Acc


# In[277]:

Guess = (y_train == False).sum()/len(y_train)


# In[278]:

Guess


# In[279]:

## Random Guessing

Guess = (y_test == False).sum()/len(y_test)


# In[280]:

Guess

