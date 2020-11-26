import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

play_tennis = pd.read_csv("PlayTennis.csv")

# This command prints out the dataset we have
print(play_tennis.head())

numberEncoder = LabelEncoder()
play_tennis['Outlook'] = numberEncoder.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = numberEncoder.fit_transform(
    play_tennis['Temperature'])
play_tennis['Humidity'] = numberEncoder.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = numberEncoder.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = numberEncoder.fit_transform(
    play_tennis['Play Tennis'])

# show table after changes

print(play_tennis.head())

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'Play Tennis'

features_train, features_test, target_train, target_test = train_test_split(
    play_tennis[features], play_tennis[target], test_size=0.33, random_state=54)

model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)

print(f'{accuracy}')

print(f'Predicting [Rain,mild,high,weak] = {model.predict([[1,2,0,1]])}')
