import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

# reading in the data tweets
data = pd.read_csv("tweets_tagged.csv", encoding="ISO-8859-1")

# getting all attributes except the label column, 'sr no, tweet'
attributes = [col for col in data.columns if col != "label"] # remove sr no column??

# columns, testing data, label answers, label answers
train_x, test_x, train_y, test_y = train_test_split(data[attributes], data["label"], test_size=0.3, random_state=123)

print("Label counts (1,2,3)")
print(data["label"].value_counts())

