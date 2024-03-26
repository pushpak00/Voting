import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

dtc = DecisionTreeClassifier(random_state=2022)
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2022, kernel='linear')
pipe_svm = Pipeline([('STD', scaler),('SVM',svm)])
da = LinearDiscriminantAnalysis()
voting = VotingClassifier([('TREE',dtc),
                           ('SVM_P',pipe_svm),('LDA',da)],
                          voting='soft')

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

## Separately evaluating classifiers
dtc.fit(X_train, y_train)
y_pred_prob = dtc.predict_proba(X_test)[:,1]
roc_dtc = roc_auc_score(y_test, y_pred_prob)

pipe_svm.fit(X_train, y_train)
y_pred_prob = pipe_svm.predict_proba(X_test)[:,1]
roc_svm = roc_auc_score(y_test, y_pred_prob)

da.fit(X_train, y_train)
y_pred_prob = da.predict_proba(X_test)[:,1]
roc_da = roc_auc_score(y_test, y_pred_prob)

print((roc_dtc,roc_svm,roc_da))
### Weighted
voting = VotingClassifier([('TREE',dtc),
                           ('SVM_P',pipe_svm),('LDA',da)],
                          voting='soft', 
                          weights=[roc_dtc, roc_svm, roc_da])

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

################### Bankruptcy #######################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
from sklearn.linear_model import LogisticRegression

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)
scaler = StandardScaler()
svm = SVC(probability=True, random_state=2022)
pipe = Pipeline([('STD',scaler),('SVM',svm)])
pipe.fit(X_train, y_train)
y_pred_prob = pipe.predict_proba(X_test)[:,1]
roc_svm = roc_auc_score(y_test, y_pred_prob)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)[:,1]
roc_lr = roc_auc_score(y_test, y_pred_prob)

dtc = DecisionTreeClassifier(random_state=2022)
dtc.fit(X_train, y_train)
y_pred_prob = dtc.predict_proba(X_test)[:,1]
roc_dtc = roc_auc_score(y_test, y_pred_prob)

# w/o weights
voting = VotingClassifier([('SVM_P',pipe),('LR',lr),('TREE',dtc)],
                          voting='soft')
voting.fit(X_train, y_train)
y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

# with weights
voting = VotingClassifier([('SVM_P',pipe),('LR',lr),('TREE',dtc)],
                          voting='soft', 
                          weights=[roc_svm,roc_lr,roc_dtc])
voting.fit(X_train, y_train)
y_pred_prob = voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

################### Grid Search CV ######################

voting = VotingClassifier([('SVM_P',pipe),('LR',lr),('TREE',dtc)],
                          voting='soft')
print(voting.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)

params = {'SVM_P__SVM__C':np.linspace(0.001,5,5),
          'SVM_P__SVM__gamma':np.linspace(0.001,5,5),
          'LR__C':np.linspace(0.001,5,5),
          'TREE__max_depth':[None, 3, 5],
          'TREE__min_samples_split':[2,5,10],
          'TREE__min_samples_leaf':[1,5]}


gcv = GridSearchCV(voting, param_grid=params, verbose=3, 
                   cv=kfold, scoring='roc_auc')

gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)






