# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:22:31 2018

@author: world
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility



import tensorflow as tf 


from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
#import numpy as np
#from PIL import Image
import glob
#import tensorflow as tf
#f= open('D:/test/test/127.jpg','r+b')
sess = tf.Session()
X_train=[]
y_train=[]
X_test=[]
y_test=[]
txt_paths =glob.glob( "F:/nlp/bbc/bbcsport/train/athletics/*.txt")
txt_paths1 =glob.glob( "F:/nlp/bbc/bbcsport/train/cricket/*.txt")
txt_paths2 =glob.glob( "F:/nlp/bbc/bbcsport/train/football/*.txt")
txt_paths3 =glob.glob( "F:/nlp/bbc/bbcsport/train/rugby/*.txt")
txt_paths4 =glob.glob( "F:/nlp/bbc/bbcsport/train/tennis/*.txt")

txt_pathstest1 =glob.glob( "F:/nlp/bbc/bbcsport/test/athletics/*.txt")
txt_pathstest2 =glob.glob( "F:/nlp/bbc/bbcsport/test/cricket/*.txt")
txt_pathstest3=glob.glob( "F:/nlp/bbc/bbcsport/test/football/*.txt")
txt_pathstest4 =glob.glob( "F:/nlp/bbc/bbcsport/test/rugby/*.txt")
txt_pathstest5=glob.glob( "F:/nlp/bbc/bbcsport/test/tennis/*.txt")


def matrixs (path,x):
    text=""
    for file_path in path:
        #file = tf.read_file(file_path)
       
        with open(file_path, "r") as text_file:
         file=text_file.read()
         tokenizer = RegexpTokenizer(r'\w+')
         file2=tokenizer.tokenize(file)
         for i in file2:
             text = text +" " +i
     
         X_train.append(text)
         y_train.append(x) 
#.eval(session=sess)       
def matrixstest (path,x):
    text=""
    for file_path in path:
        #file = tf.read_file(file_path)
       
        with open(file_path, "r") as text_file:
         file=text_file.read()
         tokenizer = RegexpTokenizer(r'\w+')
         file2=tokenizer.tokenize(file)
         for i in file2:
             text = text +" " +i
   
        X_test.append(text)
        y_test.append(x) 

matrixs(txt_paths,"athletics")
matrixs(txt_paths1,"cricket")
matrixs(txt_paths2,"football")
matrixs(txt_paths3,"rugby")
matrixs(txt_paths4,"tennis")
##


matrixstest(txt_pathstest1,"athletics")
matrixstest(txt_pathstest2,"cricket")
matrixstest(txt_pathstest3,"football")
matrixstest(txt_pathstest4,"rugby")
matrixstest(txt_pathstest5,"tennis")

#X_train =np.array(X_train)
#y_train= np.array(y_train)
#
#
#X_test=np.array(X_test)
#y_test =np.array(y_test)


#print(X_train[2],y_train[2])
#print(X_test.shape,y_test.shape)




#---------------------NB------------------------

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, y_train)



predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))

#--------------------------svm---------

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='perceptron', penalty='l2',
                                           alpha=1e-3, n_iter=10, random_state=42))])
text_clf_svm=text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)
print(np.mean(predicted_svm == y_test))



