import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


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

def data_plotting(dataSet):
    plt.figure(figsize=(20,25))
    dataSet.groupby('label').label.count().plot.bar(ylim=0)
    plt.show()


def train_MLPClassifier_default(train, validation):
    # Pipeline for SGD
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('nb_clf', MLPClassifier())
        ])

    text_clf.fit(train.text, train.label)
    scores = cross_val_score(text_clf, train.text, train.label, scoring='accuracy', cv=10)
    print('Cross Validation Score with CV Factor 10')
    print(scores)
    predicted = text_clf.predict(validation.text)
    print('\nSGD classfier accuracy on validation set: ')
    print(np.mean(predicted == validation.label))
    return text_clf

def train_MLPClassifier_with_GridSearch(train, validation, test):
    # Define the mlp pipeline to be used
    text_mlp = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mlp', MLPClassifier())
    ])

    ### Parameters for GridSearch
    # Solvers:
    #   lbfgs:
    parameters = {
#            'vect__max_df': (0.5, 0.75, 1.0),
            # 'vect__max_features': (None, 5000, 10000, 50000),
#            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            'mlp__solver': ('lbfgs','sgd'), #,'liblinear','sgd'
            'mlp__max_iter': (100, 150), #, 15, 20,
            'mlp__alpha': (0.01, 0.001),#, 0.0001, 0.00001, 0.000001
#            'mlp__penalty': ('l1', 'l2', 'elasticnet'),
            'mlp__hidden_layer_sizes': np.arange(1, 5),
#            'mlp__random_state':[0,1,2]
            }

    print("Applying gridsearch on MLPClassifier")
    gs_mlp = GridSearchCV(text_mlp, parameters, cv=10, n_jobs=5, verbose=1)
    gs_mlp.fit(train.text, train.label)

    sgd_df = pd.DataFrame.from_dict(gs_mlp.cv_results_)
    sgd_df.sort_values(by=["rank_test_score"])
    gs_mlp.predict(validation.text)

    print('\n\n')
    print('Model Best Parameters :')
    print(gs_mlp.best_params_)

    print('score on training set= ', gs_mlp.score(train.text, train.label))
    print('Model Best Score = ',gs_mlp.best_score_)
    test_predict = gs_mlp.predict(test.text)
    acc = accuracy_score(test.label, test_predict)
    print('accuracy on test set= ', acc)
    validation_predict = gs_mlp.predict(validation.text)
    acc = accuracy_score(validation.label, validation_predict)
    print('accuracy on validation set= ', acc)
    cm = confusion_matrix(test.label, test_predict)
    print('\n Confusion Matrix\n')
    print(cm)
    print('\n Heatmap\n')
    sns.heatmap(cm, center=True)
    plt.show()

train, test = data_preprocessing()
train, validation, test = data_cleaning(train, test)

# plot training and validation data
#data_plotting(train)
#data_plotting(validation)

#Remove labels for which there are not enough samples
train=train.groupby("label").filter(lambda x: len(x) >= 10)
# Remove labels from Validation and Test Set which are not present in Training Set
validation = (validation[validation.label.isin(train.label)])
test = (test[test.label.isin(train.label)])

print('Part 1 Using Grid Search to try various hyperparameters with MLP Classifier\n')
#mlp_clf=train_MLPClassifier_default(train,validation)
#print('\nMLP classfier accuracy on test set')
#print(np.mean(mlp_clf.predict(test.text) == test.label))
train_MLPClassifier_with_GridSearch(train,validation,test)
