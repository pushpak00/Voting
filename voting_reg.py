import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=2022)

ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state=2022)
voting = VotingRegressor([('RIDGE',ridge), ('LASSO',lasso),
                          ('ELASTIC',elastic),('TREE',dtr)])
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))

## Evaluating Regressors Separately
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)

lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)

elastic = ElasticNet()
elastic.fit(X_train, y_train)
y_pred = elastic.predict(X_test)
r2_elastic = r2_score(y_test, y_pred)

dtr = DecisionTreeRegressor(random_state=2022)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)
# Weigted
voting = VotingRegressor([('RIDGE',ridge), ('LASSO',lasso),
                          ('ELASTIC',elastic),('TREE',dtr)],
                         weights=[r2_ridge, r2_lasso, r2_elastic, r2_dtr])
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))


############ Grid Search ##########################
kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state=2022)
voting = VotingRegressor([('RIDGE',ridge), ('LASSO',lasso),
                          ('ELASTIC',elastic),('TREE',dtr)])
print(voting.get_params())
params = {'RIDGE__alpha':np.linspace(0.001, 5, 5),
          'LASSO__alpha':np.linspace(0.001, 5, 5),
          'ELASTIC__alpha':np.linspace(0.001, 5, 5),
          'ELASTIC__l1_ratio':np.linspace(0,1,5),
          'TREE__max_depth':[None,3],
          'TREE__min_samples_split':[2,5,10],
          'TREE__min_samples_leaf':[1,5,10]}

gcv = GridSearchCV(voting, param_grid=params,
                   cv=kfold, verbose=3, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

