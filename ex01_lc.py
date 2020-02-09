import numpy as np
from scipy import sparse
import pandas as pd
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


number_of_jobs =5
# creates a data frame from a json file
def create_tweets_df(file_name):
    tweets = []
    for line in open(file_name, 'r',encoding='utf-8-sig'):
        tweets.append(json.loads(line))
    return pd.DataFrame(tweets, columns=['tweetIDs','text'])

# creates a data frame from a csv file
def create_df_from_csv(file_name, column_names):
    csvDf = pd.read_csv(file_name, sep = '\t', header=None,names = column_names)
    return csvDf

# merges two data frames
def merge_tweets_and_labels(tweets, labels):
    # merge tweets_id and train/test labels data set to get the training and test data
    merged_df = pd.merge(left = tweets, right=labels,left_on ='tweetIDs',right_on='tweetIDs', how='inner')
    return merged_df

def train_MNB_default(train, validation):
    # Pipeline for MNB
    text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('nb_clf', MultinomialNB())
            ])
    text_clf.fit(train.text, train.label)
    scores = cross_val_score(text_clf, train.text, train.label, scoring='accuracy', cv=10)
    print('Cross Validation Score with CV Factor 10')
    print(scores)
    predicted = text_clf.predict(validation.text)
    #print(predicted)
    print('\nMNB classfier accuracy on validation set: ')
    print(np.mean(predicted == validation.label))
    return text_clf

def train_SGD_default(train, validation):
    # Pipeline for SGD
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('nb_clf', SGDClassifier(max_iter=50,  tol=0.001))
        ])

    text_clf.fit(train.text, train.label)
    scores = cross_val_score(text_clf, train.text, train.label, scoring='accuracy', cv=10)
    print('Cross Validation Score with CV Factor 10')
    print(scores)
    predicted = text_clf.predict(validation.text)
    print('\nSGD classfier accuracy on validation set: ')
    print(np.mean(predicted == validation.label))
    return text_clf

def train_MNB_with_GridSearch(train, validation,test):

    text_mnb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mnb', MultinomialNB())
    ])

    param_grid = {'vect__ngram_range': [(1, 1),(1,2), (1,3)],
                  'mnb__alpha' : [0.001,0.01,0.1],
                  'mnb__fit_prior' : [True, False],
             }

    print("Applying gridsearch on MNBClassifier")
    gs_mnb = GridSearchCV(text_mnb, param_grid, cv=5, n_jobs=number_of_jobs, verbose=1)
    gs_mnb.fit(train.text, train.label)

    mnb_df = pd.DataFrame.from_dict(gs_mnb.cv_results_)
    mnb_df.sort_values(by=["rank_test_score"])

    print('\n\n')
    print('Model Best Parameters :')
    print(gs_mnb.best_params_)
    print('score on training set= ', gs_mnb.score(train.text, train.label))
    print('Model Best Score = ',gs_mnb.best_score_)
    test_predict = gs_mnb.predict(test.text)
    acc = accuracy_score(test.label, test_predict)
    print('accuracy on test set= ', acc)
    validation_predict = gs_mnb.predict(validation.text)
    acc = accuracy_score(validation.label, validation_predict)
    print('accuracy on validation set= ', acc)
    cm = confusion_matrix(test.label, test_predict)
    print('\n Confusion Matrix\n')
    print(cm)
    print('\n Heatmap\n')
    sns.heatmap(cm, center=True)
    plt.show()

def train_SGDClassifier_with_GridSearch(train, validation, test):

    text_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('sgd', SGDClassifier(tol=0.001))
    ])

    param_grid = {'vect__ngram_range': [(1, 1)],
             'sgd__loss': ['hinge', 'squared_hinge','modified_huber'],
             'sgd__alpha' : [0.001,0.01,0.1],
             'sgd__penalty' : ["l2", "l1", "none"],
             'sgd__max_iter' : [50, 100, 200],
             }

    print("Applying gridsearch on SGDClassifier")
    gs_sgd = GridSearchCV(text_sgd, param_grid, cv=5, n_jobs=number_of_jobs, verbose=1)
    gs_sgd.fit(train.text, train.label)

    sgd_df = pd.DataFrame.from_dict(gs_sgd.cv_results_)
    sgd_df.sort_values(by=["rank_test_score"])
    gs_sgd.predict(validation.text)

    print('\n\n')
    print('Model Best Parameters :')
    print(gs_sgd.best_params_)

    print('score on training set= ', gs_sgd.score(train.text, train.label))
    print('Model Best Score = ',gs_sgd.best_score_)
    test_predict = gs_sgd.predict(test.text)
    acc = accuracy_score(test.label, test_predict)
    print('accuracy on test set= ', acc)
    validation_predict = gs_sgd.predict(validation.text)
    acc = accuracy_score(validation.label, validation_predict)
    print('accuracy on validation set= ', acc)
    cm = confusion_matrix(test.label, test_predict)
    print('\n Confusion Matrix\n')
    print(cm)
    print('\n Heatmap\n')
    sns.heatmap(cm, center=True)
    plt.show()

def data_preprocessing():
    # Data Preprocessing
    tweets_df = create_tweets_df('tweets.json')
    train_labels = create_df_from_csv('labels-train+dev.tsv', ['label','tweetIDs'])
    test_labels = create_df_from_csv('labels-test.tsv',['label', 'tweetIDs'])

    # adjust data type of tweetIDs to match the labels
    tweets_df['tweetIDs']=tweets_df['tweetIDs'].astype('int64')
    # merge tweets and labels to get the training df
    train = merge_tweets_and_labels(tweets_df, train_labels)
    test = merge_tweets_and_labels(tweets_df, test_labels)
    return train,test

def data_cleaning(train,test):
    ### Data Cleaning
    train.drop('tweetIDs', axis = 1, inplace=True)
    test.drop('tweetIDs', axis = 1, inplace=True)
    ###Training Set split into Trainng and Validation Set
    train, validation= train_test_split(train, test_size=0.1, random_state=42)
    return train, validation, test

def data_plotting(dataSet, title_str):
    plt.figure(figsize=(20,25))
    dataSet.groupby('label').label.count().plot.bar(ylim=0)
    plt.title(title_str)
    plt.show()

train, test = data_preprocessing()

train, validation, test = data_cleaning(train, test)

# plot training and validation data
data_plotting(train, 'Training Data Set Label Frequency')
#data_plotting(validation,'Validation Data Set Label Frequency')

#Remove labels for which there are not enough samples
train=train.groupby("label").filter(lambda x: len(x) >= 10)
# Remove labels from Validation and Test Set which are not present in Training Set
validation = (validation[validation.label.isin(train.label)])
test = (test[test.label.isin(train.label)])

print('Part 1 Pipelined models\n')

print('\nTrain model using MNB Classifier \n')
mnb_clf = train_MNB_default(train, validation)

print('\nMNB classfier accuracy on test set')
print(np.mean(mnb_clf.predict(test.text) == test.label))

print('\n\nTrain model using SGD Classifier\n')
sgd_clf= train_SGD_default(train,validation)

print('SGD classifier accuracy on test set')
print(np.mean(sgd_clf.predict(test.text) == test.label))

print('\n\nPart 2 Using Grid Search to try various hypertparameters')
print('and compare the performance of SGDClassifier and Multinomial Naive Bayes\n\n')

train_MNB_with_GridSearch(train, validation,test)
train_SGDClassifier_with_GridSearch(train, validation, test)
