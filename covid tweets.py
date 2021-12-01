import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go


# to customize style
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


warnings.filterwarnings('ignore')
# reading in the data tweets
data = pd.read_csv("tweets_tagged.csv", encoding="ISO-8859-1")
# dropping Sr No column because they are simply indexes
data = data.drop(["Sr No"], axis=1)


# convert the string and remove english stop words
vectorizer = CountVectorizer(binary=True, stop_words="english")
X = vectorizer.fit_transform(data["tweet"])
# newData contains only the tweets
newData = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())


# getting all attributes except the label column, 'sr no, tweet'
attributes = [col for col in data.columns if col != "label"] # remove sr no column??

# columns, testing data, label answers, label answers
train_x, test_x, train_y, test_y = train_test_split(newData, data["label"], test_size=0.3, random_state=123)

# converting to dataset
new_train_x = pd.DataFrame(train_x, columns=newData.columns)
new_test_x = pd.DataFrame(test_x, columns=newData.columns)
new_train_y = pd.DataFrame(train_y, columns=["label"])
new_test_y = pd.DataFrame(test_y, columns=["label"])


# shape
print("Shapes")
print(new_train_x.shape)
print(new_test_x.shape)
print(new_train_y.shape)
print(new_test_y.shape)
print()
# amount of certain labels
print("Label counts (1,2,3)")
print(data["label"].value_counts())


# Decision tree
clf = tree.DecisionTreeClassifier()
# train model
clf = clf.fit(train_x, train_y)
# make prediction
pred_y = clf.predict(test_x)

# getting different numbers each time it runs
# sampled no, micro no, none no, weighted no, macro yes
print("\nmacro")
print("Decision Tree Results")
print("------------------------")
print("f1:" + str(f1_score(pred_y, test_y, average="macro")))
print("accuracy:" + str(accuracy_score(pred_y, test_y)))
print("precision:" + str(precision_score(pred_y, test_y, average="macro")))
print("recall:" + str(recall_score(pred_y, test_y, average="macro")))

# linear svc
clf = LinearSVC()
clf = clf.fit(new_train_x, train_y)
pred_y = clf.predict(new_test_x)
print("\nLinearSVC Results")
print("------------------------")
print("f1:" + str(f1_score(pred_y, test_y, average="macro")))
print("accuracy:" + str(accuracy_score(pred_y, test_y)))
print("precision:" + str(precision_score(pred_y, test_y, average="macro")))
print("recall:" + str(recall_score(pred_y, test_y, average="macro")))


# multinomial naive bayes
clf = MultinomialNB()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
print("\nMultinomialNB Results")
print("------------------------")
print("f1:" + str(f1_score(pred_y, test_y, average="macro")))
print("accuracy:" + str(accuracy_score(pred_y, test_y)))
print("precision:" + str(precision_score(pred_y, test_y, average="macro")))
print("recall:" + str(recall_score(pred_y, test_y, average="macro")))


# Regression
clf = LogisticRegression(solver="lbfgs", max_iter=1000)
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
print("\nLinear Regression Results")
print("------------------------")
print("f1:" + str(f1_score(pred_y, test_y, average="macro")))
print("accuracy:" + str(accuracy_score(pred_y, test_y)))
print("precision:" + str(precision_score(pred_y, test_y, average="macro")))
print("recall:" + str(recall_score(pred_y, test_y, average="macro")))


# cross validation
x = newData
y = data['label']
scores = cross_val_score(clf, x, y, cv=10)
print("\nCross Validation")
print("------------------------")
print("Accuracy: %0.2f" % scores.mean())


# map layout
# app.layout =html.Div([html.Label('Covid-19 Sentimental Tweets'),
#               dcc.Dropdown(id='myDropdown',options=[{'label':'Decision Tree', 'value':'trig'},
#                                                     {'label':'LinearSVC', 'value':'Crime'}]
#                            , value='Crime', style={"width":"50%"}, clearable=False),
#
#     dcc.Graph(
#         id='crime-graph', style={'width':'180vh', 'height':'90vh'},
#
#     )
# ])
# # call back
# @app.callback(
#     Output(component_id='crime-graph', component_property='figure'),
#     [Input(component_id='myDropdown', component_property='value')]
# )
# def graphType(myDropdown):
#     hol=df
#     iday=data
#     if myDropdown == 'Crime':
#         blop = px  #bar(hol, x=myDropdown, y='Amount', color='Crime')
#         return (blop)
#     else:
#         trig = px.choropleth(iday, geojson=stateFile, locations=stateName, featureidkey='properties.NAME',
#                          color=totalCrimes,
#                          color_continuous_scale="viridis", range_color=(0, 900000),
#                          scope='usa', labels={'State Total': 'Total Crimes Committed'})
#
#         trig.update_layout(title='Total Crimes per State 2019', geo=dict(scope='usa', projection=dict(type='albers usa'),
#                                                                      showlakes=True, lakecolor='rgb(204, 224, 255)'))
#         return (trig)
#
# if __name__ == '__main__':
#     # this lets us have auto refresh to the graph when data is changed
#     app.run_server(debug=True)
