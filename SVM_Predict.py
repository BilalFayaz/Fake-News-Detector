import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer    # initialize it with stop words  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Referenced from this site
# https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47



def get_from_csv():

    df = pd.read_csv('news.csv', low_memory=False)
    df = df.dropna()

    diagnosis_map = {'REAL':1, 'FAKE':-1}
    df['label'] = df['label'].map(diagnosis_map)
    labels=df.label

    # print(labels.head())
    
    x_train = df['text']
    y_train = labels

    y_train = np.array(y_train)

    return x_train,y_train


def Vectorize(x_train,x_test):

    # DataFlair - Initialize a TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
    #DataFlair - Fit and transform train set, transform test set

    tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test=tfidf_vectorizer.transform(x_test)

    tfidf_train = tfidf_train.toarray()
    tfidf_test = tfidf_test.toarray()

    return tfidf_train,tfidf_test

class SVM():
    def __init__(self, learning_rate = 1/10, lambda_parameter = 0.0001, n_iters = 10):
        self.LR = learning_rate
        self.LP = lambda_parameter
        self.N = n_iters        # no. of iterations
        self.w = None           # Weights
        self.b = None           # Bias


    def fit(self,X,y):   # Learning Data

        print('LEARNING')

        Y = y
        n_samples, n_features = X.shape  # Gets tuple of Dimensions of X

        # n_features has the dimension of X which has the vectorised words
        # self.w has an array of weights full of zeros at first
        # bias is set to 0 
        self.w = np.zeros(n_features)      
        self.b = 0                          

        # Linear Model
        # Gradient Descent
        for _ in range(self.N):

            for idx, x_i in enumerate(X):

                # f(x) = (np.dot(x_i,self.w) - self.b)
                
                # Calculating Cost function 
                # Hinge Loss
                
                # if Y * f(x) >= 1 Then our loss is zero
                # loss function helps maximize the margin in hinge loss.
                condition = Y[idx] * (np.dot(x_i,self.w) - self.b) >= 1
        
                if condition:
                    # if class is correctly predicted 
                    # Updating Gradients with only regularization parameter
                    # regularization parameter to balance the margin maximization and loss.
                    self.w -= self.LR * (2 * self.LP * self.w)

                else:

                    # if class is incorrectly predicted 
                    # use loss to update Gradient with regularization parameter and loss where loss = np.dot(x_i, Y[idx])

                    self.w -= self.LR * (2 * self.LP * self.w - np.dot(x_i, Y[idx]))

                    #updating bias
                    self.b -= self.LR * Y[idx]                      

    def predict(self,X):                # Predicting Data
        
        print('PREDICTING')

        Y_pred = np.sign(np.dot(X,self.w) - self.b)
       
        if Y_pred == -1:
            return 'FAKE'
        else:
            return 'REAL'


a = SVM()

# opening text file to read for prediction
def read_text():

    try:

        file = open('new.txt')
        x_test = [file.read()]
        file.close()
    except:
        file = open('new.txt', encoding='utf-8')
        x_test = [file.read()]
        file.close()

    return x_test

# X_test = read_text()

# X_train, Y_train = get_from_csv()
# X_train, X_test = Vectorize(X_train, X_test) 

# a.fit(X_train,Y_train)
# print(a.predict(X_test))
