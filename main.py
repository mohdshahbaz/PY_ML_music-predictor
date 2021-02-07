import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

# loading music data from csv format
music_data = pd.read_csv('music.csv')

# input data set
X = music_data.drop(columns=['genre'])

# output data set
y = music_data['genre']

# splitting output & input data into two sets of training & testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialize decision tree classifier
model = DecisionTreeClassifier()

# gives training data to the decision tree classifier to create trained model
model.fit(X_train, y_train)

# load trained model from memory
# model = joblib.load('music-recommender.joblib')

# creates .dot file for visualizing trained model's binary tree
# tree.export_graphviz(model, out_file='music-recommender.dot',
#                     feature_names=['age', 'gender'],
#                     class_names=sorted(y.unique()),
#                     label='all',
#                     rounded=True,
#                     filled=True
#                     )

# predicts genre based on gender=1(male) & age 21
predictions = model.predict([[28,0]])

# calculate trained model accuracy score
# score = accuracy_score(y_test, predictions)
# score

# outputs prediction
print(predictions)