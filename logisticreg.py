import numpy as np
from numpy import mean
from numpy import std
import sklearn
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import winsound

combinedMusic = np.load('combinedMusicv2.npy') #all entries with music, shape 1500*79 x 30
combinedOther = np.load('combinedOtherv2.npy')

combinedMusic, combinedOther, combinedAll_data= statistics.PCA_func(combinedMusic, combinedOther)

combinedAll_target = np.zeros(combinedAll_data.shape[0])

for i in range(0, combinedMusic.shape[0]):
    combinedAll_target[i] = 1;
print(combinedAll_target.shape[0])
print(np.count_nonzero(combinedAll_target == 0))
print(np.count_nonzero(combinedAll_target == 1))

x_train, x_test, y_train, y_test = train_test_split(combinedAll_data, combinedAll_target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)
# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print('accurancy: ', score)

## ---- logistic regression with k fold ---------

# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, combinedAll_data, combinedAll_target, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# -------------------------------------------

# ----------Random forest -----------------
print('-------RANDOM FOREST =----')
# Number of trees in random forest
n_estimators = np.linspace(100, 3000, int((3000-100)/200) + 1, dtype=int)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]# Minimum number of samples required to split a node
# min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)]
min_samples_split = [1, 2, 5, 10, 15, 20, 30]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Criterion
criterion=['gini', 'entropy']
random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}
rf_base = RandomForestClassifier()
#rf_random = RandomizedSearchCV(estimator = rf_base,
#                               param_distributions = random_grid,
#                               n_iter = 10, cv = 2,
#                               verbose=2,
#                               random_state=42, n_jobs = 2)
#rf_random.fit(x_train, y_train)

rf_base.fit(x_train, y_train);
#print(rf_random.best_params_)
print(rf_base.score(x_train, y_train))
print(rf_base.score(x_test, y_test))

winsound.Beep(440, 2000)

## Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 10, random_state = 42)# Train the model on training data
#rf.fit(x_train, y_train);

## Use the forest's predict method on the test data
#predictions = rf.predict(x_test)

#print(rf.score(x_test, y_test))
#print(rf.score(x_train, y_train))

