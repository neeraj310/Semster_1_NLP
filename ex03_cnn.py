
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Embedding
from tensorflow.keras.layers import Dropout, Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import itertools

max_features = 6000
maxlen = 268
default_batch_size = 100
embedding_dims = 50
default_filters_size = 400
default_kernel_size = 3
dropout_default = 0.3
hidden_dims = 250
epochs = 1
default_stride = 1
cross_validation_size =2 
output_size = 14 # maximum nu of output labels
verbose = False

DEBUG_PRINT = True # Switch for plotting debug info

# Define parameter grids for various models
param_grid_model_1  = {'filters': [100,200,400],
                       'kernel_size': [3,5,10],
                       'dropout':[0.3,0.4,0.5],
                       'optimizer': ['Adam', 'Nadam']
                      }

param_grid_model_2 = { 'activation': ['relu','elu'],
                       'learn_rate' : [0.001,0.01,0.1],
                       'batch_size' : [25,50,100],
                     }

param_grid_model_3 = { 'kernel_size': [3,5],
                       'dropout' :    [0.3, 0.5]
                     }

param_grid_model_4 = { 'strides': [1,2,3],
                      'kernel_size' : [3, 5, 10],
                      'optimizer': ['Nadam', 'Adam'],
                      'epochs'   : [5, 10]
                     }

param_grid_model_5 = { 'filters' : [100, 150],
                       'kernel_size': [3,5],
                       'optimizer' :  ['Nadam','adam'],
                       'strides' :    [1,2,3]
                     }

class DATA_PREPROCESSOR():
    """
    Data Preprocessing Class
    """
    def __init__(self, file_name1,file_name2,file_name3):
        """
        Define paths of 3 data files to be read
        """
        my_absolute_dirpath = os.path.abspath(os.path.dirname(file_name1))
        self.file_name1 = os.path.join(my_absolute_dirpath + '/' + file_name1)
        print(self.file_name1)

        self.file_name2 = os.path.join(my_absolute_dirpath + '/' +file_name2)
        print(self.file_name2)

        self.file_name3 = os.path.join(my_absolute_dirpath + '/' +file_name3)
        print( self.file_name3)

    def readDataSet(self):
        '''
        Read input corpus files
        '''
        filename = self.file_name1
        print(filename)

        tweets = []
        for line in open(filename, 'r',encoding='utf-8-sig'):
             tweets.append(json.loads(line))

        tweets_df = pd.DataFrame(tweets, columns=['tweetIDs','text'])

        filename = self.file_name2

        train_labels = pd.read_csv(filename, sep = '\t', header=None,names = ['label','tweetIDs'])

        filename = self.file_name3
        test_labels =  pd.read_csv(filename, sep = '\t', header=None,names = ['label','tweetIDs'])

        return(tweets_df,train_labels, test_labels)

    def merge_tweets_and_labels(self,tweets, labels):
        """
        merge tweets_id and train/test labels data set to get the training and test data
        """
        merged_df = pd.merge(left = tweets, right=labels,left_on ='tweetIDs',right_on='tweetIDs', how='inner')
        return merged_df

    def handle_class_imblance(self,train):
        """
        Replace labels of all samples in training set with frequency less than 200
        """
        imbalanced_languages = train.groupby("label").filter(lambda x: len(x) <= 200)
        imbalanced_labels = train.label.isin(imbalanced_languages.label)
        # replace imblance labels with und(undefined)
        train.loc[imbalanced_labels, 'label'] = 'und'
        return train

    def data_cleaning(self,train,test):

        """
        Data Cleaning
        """
        train.drop('tweetIDs', axis = 1, inplace=True)
        test.drop('tweetIDs', axis = 1, inplace=True)
        # Remove white spaces from the labels
        train.label = train.label.str.strip()
        test.label = test.label.str.strip()

        train = self.handle_class_imblance(train)
        #Training Set split into Trainng and Validation Set
        train, validation = train_test_split(train, test_size=0.1, random_state=42)

        # Remove labels from Validation and Test Set which are not present in Training Set
        validation = (validation[validation.label.isin(train.label)])
        test = (test[test.label.isin(train.label)])
        return train, validation, test

    def tokenizer(self,train, test,validation):
        """
        Tokenize the tweets into character level vocab
        Find the size of max tweet
        And use size of longest tweet for padding size to make the input vector of
        same length
        We have hardcoded the max_length to 268 after printing out the longest tweet length
        """
        tokenizer = Tokenizer(char_level = True)
        tokenizer.fit_on_texts(train.text)
        x_train = tokenizer.texts_to_sequences(train.text)
        x_test =  tokenizer.texts_to_sequences(test.text)
        x_validation =  tokenizer.texts_to_sequences(validation.text)
        """
        Enable padding to make the input vector length independent of tweet length
        """
        x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post', value=0)
        x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post', value=0)
        x_validation = pad_sequences(x_validation, maxlen=maxlen, padding='post', truncating='post', value=0)

        """
        Convert the label names to one hot vector encoding
        """
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train.label)
        np.unique(y_train)
        num_class = max(y_train+1)
        y_train = to_categorical(y_train,num_class)
        y_test= label_encoder.transform(test.label)
        y_test = to_categorical(y_test,num_class)
        y_validation= label_encoder.transform(validation.label)
        y_validation = to_categorical(y_validation,num_class)
        return (x_train, x_test, x_validation, y_train, y_test,y_validation)

    def preprocess_data(self):
        '''
        Preprocess the input corpus. Following steps are involed in data cleaning

        '''
        tweets_df,train_labels, test_labels = self.readDataSet()
         # adjust data type of tweetIDs to match the labels
        tweets_df['tweetIDs']=tweets_df['tweetIDs'].astype('int64')
        # merge tweets and labels to get the training df
        train = self.merge_tweets_and_labels(tweets_df, train_labels)
        test = self.merge_tweets_and_labels(tweets_df, test_labels)
        train, validation, test = self.data_cleaning(train, test)

        return train, validation, test

class PLOTTER():
    def __init__(self, is_print_enabled=False):
        self.is_print_enabled = is_print_enabled

    def data_plotting(self, dataSet, title_str):
        """
        Plot the trainign data after clean up
        Plotting training data gives a good intution about the training set
        """
        if self.is_print_enabled:
            plt.figure(figsize=(20,25))
            dataSet.groupby('label').label.count().plot.bar(ylim=0)
            plt.title(title_str)
            plt.show()

    def plot_history(self,history):
        """
        Plot Accuracy and loss plot of the Gridsearch best model
        """
        if self.is_print_enabled:
            # Plot training & validation accuracy values
            training_accuracy = history.history['accuracy']
            test_accuracy = history.history['val_accuracy']
            training_loss = history.history['loss']
            test_loss = history.history['val_loss']
            epoch_count = range(1, len(training_loss) + 1)

            plt.figure(figsize=(7,7))
            plt.plot(epoch_count, training_accuracy, 'r--')
            plt.plot(epoch_count, test_accuracy, 'b-')
            plt.legend(['Training accuracy', 'Test accuracy'])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model accuracy')
            plt.show();

            plt.figure(figsize=(7,7))
            plt.plot(epoch_count, training_loss, 'r--')
            plt.plot(epoch_count, test_loss, 'b-')
            plt.legend(['Training Loss',    'Test Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model loss')
            plt.show();

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if self.is_print_enabled:
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                title='Normalized confusion matrix'
            else:
                title='Confusion matrix'
            plt.figure(figsize=(8,8))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

def model_1(filters, kernel_size, dropout, optimizer):
    # then we can go ahead and set the parameter space

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
    model.add(Conv1D((filters),
                 (kernel_size),
                 padding='valid',
                 activation='relu',
                 strides=(default_stride)))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout),)
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    #optimizer = 'adam'
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary()
    return model

def model_2(activation,learn_rate):
    # then we can go ahead and set the parameter space
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
    model.add(Conv1D((default_filters_size),
                 (default_kernel_size),
                 padding='valid',
                 activation=activation,
                 strides=(default_stride)))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_default),)
    model.add(Activation('relu'))
     # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    #optimizer = 'adam'
    optimizer = SGD(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary()
    return model

def model_3(kernel_size, dropout):
    # then we can go ahead and set the parameter space

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
    model.add(Conv1D((default_filters_size),
                 (kernel_size),
                 padding='valid',
                 activation='relu',
                 strides=(default_stride)))
    # we use max pooling:
    model.add(GlobalAveragePooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout),)
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    optimizer = 'Nadam'
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary()
    return model

def model_4(strides,kernel_size,optimizer):
    # then we can go ahead and set the parameter space

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
    model.add(Conv1D((default_filters_size),
                 (kernel_size),
                 padding='valid',
                 activation='relu',
                 strides=strides))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_default),)
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    #optimizer = 'adam'
    #optimizer = SGD(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary()
    return model

def model_5(filters, kernel_size, optimizer,strides):
    # then we can go ahead and set the parameter space

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
    model.add(Conv1D((filters),
                 (kernel_size),
                 padding='valid',
                 activation='relu',
                 strides=strides))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_default),)
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    #optimizer = 'adam'
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary()
    return model

class MODEL_RUNNER():
    def __init__(self,model_fn, params, plotter,verbose = False):
        self.model_fn  = model_fn
        self.params = params
        self.batch_size = default_batch_size
        self.verbose_level = verbose
        self.plotter = plotter

    def run_model(self,data_tuple, train):
        """
        Execute the model based on the function being passed
        data_tuple : Data Tuple containing test, training and validation data
        train = training corpus
        """
        x_train,x_validation, x_test, y_train,y_validation, y_test = data_tuple

        model = KerasClassifier(self.model_fn, batch_size=self.batch_size, \
                                verbose =  self.verbose_level)
        """
        Start Gridsearch according to the param list being passed while
        creating the model instance
        """
        validator = GridSearchCV(model,
                                 param_grid=self.params,
                                 verbose   =  self.verbose_level,
                                 cv= cross_validation_size)
        validator.fit(x_train, y_train, verbose =  self.verbose_level)

        """
        Plot Confusion Matrix, and Accuracy-Loss Output per epoch
        """
        self.gridsearch_output(validator,data_tuple,train)

    def probabilty_to_classencoding(self, y_pred):
        """
        Output of sigmoid is probability distribution,
        But we need the one hot encoding vector based on the class of
        highest probabilty.
        This function converts output of sigmoid to output class one hot encoding form
        """
        y_class = np.array( y_pred )
        idx = np.argmax(y_class, axis=-1)
        y_class = np.zeros(y_class.shape )
        y_class[ np.arange(y_class.shape[0]), idx] = 1
        return y_class

    def binary_class_to_label(self, binary_class):
        """
        Return the class value based on the one hot encoding vector
        """
        return(np.argmax(binary_class, axis=1))

    def gridsearch_output(self,validator, data_tuple,train):
        """
        Print Gridsearch best model score and its parameters
        Dump Confusion matrix
        Dump Loss and Accuracy plot
        """
        print('The parameters of the best model are: ')
        print(validator.best_params_)
        print('Grid Result Best Score is ')
        print(validator.best_score_)

        best_model = validator.best_estimator_.model
        x_train,x_validation, x_test, y_train,y_validation, y_test = data_tuple
        print('Accuracy on validation Test')
        loss, accuracy = best_model.evaluate(x_validation, y_validation)
        print('loss = {} : accuraacy : {}'.format(loss, accuracy))

        print('Accuracy on Test Set')
        metric_names = best_model.metrics_names
        metric_values = best_model.evaluate(x_test, y_test)
        for metric, value in zip(metric_names, metric_values):
            print(metric, ': ', value)

        y_true, y_pred = y_test, best_model.predict(x_test)
        """
        conver predicted prbablity values to one hot encoding
        """
        y_pred = self.probabilty_to_classencoding(y_pred)

        print('Confusion Matrix')
        cm = confusion_matrix(self.binary_class_to_label(y_true), \
                              self.binary_class_to_label(y_pred))
        print(cm)

        print(classification_report(self.binary_class_to_label(y_true),
                                    self.binary_class_to_label(y_pred)))
        self.plotter.plot_confusion_matrix(cm,classes=train.label.unique())

        history = best_model.fit(x_train, y_train,
        batch_size=self.batch_size,
        validation_data = (x_validation, y_validation),
        epochs=epochs,verbose = 1)
        """
        Plot Loss and Accuracy function per epoch
        """
        self.plotter.plot_history(history)

def main():
    """
    main function
    Runs 5 model one by one
    DEBUG_PRINT : Make this flag true if user wants to see various plots and CM
    """

    preprocessor = DATA_PREPROCESSOR('tweets.json','labels-train+dev.tsv','labels-test.tsv')
    train, validation, test, = preprocessor.preprocess_data()

    x_train, x_test,x_validation, y_train, y_test, y_validation = \
                                         preprocessor.tokenizer(train, test,validation)
                                         

    x_train = x_train[1:100,:]
    y_train = y_train[1:100, :]

    x_validation = x_train[1:100,:]
    y_validation = y_train[1:100, :]                                         

    data_tuple = ( x_train,x_validation, x_test, y_train,y_validation, y_test)
    """
    plot training and validation data
    """
    plotter = PLOTTER(DEBUG_PRINT)
    plotter.data_plotting(train, 'Training Data Set Label Frequency')

    print('Running Model 1')
    model1_runner = MODEL_RUNNER(model_1, param_grid_model_1, plotter,verbose)
    start_time = time.time()
    model1_runner.run_model(data_tuple, train)
    print("--- time for model 1 is %s seconds ---" % (time.time() - start_time))

    print('Running Model 2')
    start_time = time.time()
    model2_runner = MODEL_RUNNER(model_2, param_grid_model_2, plotter,verbose)
    model2_runner.run_model(data_tuple, train)
    print("--- time for model 2 is %s seconds ---" % (time.time() - start_time))

    print('Running Model 3')
    start_time = time.time()
    model3_runner = MODEL_RUNNER(model_3, param_grid_model_3,plotter,verbose)
    model3_runner.run_model(data_tuple, train)
    print("--- time for model 3 is %s seconds ---" % (time.time() - start_time))

    print('Running Model 5')
    start_time = time.time()
    model5_runner = MODEL_RUNNER(model_5, param_grid_model_5,plotter, verbose)
    model5_runner.run_model(data_tuple, train)
    print("--- time for model 5 is %s seconds ---" % (time.time() - start_time))

    # Running model 4 after model 5 to enable early stopping for model 4
    print('Running Model 4')
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)
    start_time = time.time()
    model4_runner = MODEL_RUNNER(model_4, param_grid_model_4,plotter, verbose)
    model4_runner.run_model(data_tuple, train)
    print("--- time for model 4 is %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__": main()
