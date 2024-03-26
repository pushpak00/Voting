import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Image Segmentation")

image_seg = pd.read_csv("Image_Segmention.csv")
X = image_seg.drop('Class', axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
print(le.classes_)

lr = LogisticRegression()
nb = GaussianNB()
da = LinearDiscriminantAnalysis()
scaler = StandardScaler()
svm = SVC(kernel='linear', probability=True, random_state=2022)
pipe = Pipeline([('STD', scaler),('SVM',svm)])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
voting = VotingClassifier([('LR',lr),('NB',nb),
                           ('LDA',da),('SVML',pipe)], voting='soft')
print(voting.get_params())

params = {'LR__C':np.linspace(0.001,5,5),'SVML__SVM__C':np.linspace(0.001,5,5)}

gcv = GridSearchCV(voting, param_grid=params, cv=kfold,
                   verbose=3, scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)





