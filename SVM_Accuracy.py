import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer    # initialize it with stop words  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('news.csv', low_memory=False)
df = df.dropna()


diagnosis_map = {'REAL':1, 'FAKE':-1}
# diagnosis_map = {1:1, 0:-1}
df['label'] = df['label'].map(diagnosis_map)
labels=df.label


x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

y_train = np.array(y_train)
y_test = np.array(y_test)

# DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set


tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


tfidf_train = tfidf_train.toarray()
tfidf_test = tfidf_test.toarray()

class SVM():
    def __init__(self, learning_rate = 1/100, lambda_parameter = 0.0001, n_iters = 100):
        self.LR = learning_rate
        self.LP = lambda_parameter
        self.N = n_iters
        self.w = None
        self.b = None


    def fit(self,X,y):

        print('LEARNING')

        Y = y
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        # gradient descent
        for _ in range(self.N):

            for idx, x_i in enumerate(X):

                condition = Y[idx] * (np.dot(x_i,self.w) - self.b) >= 1
        
                if condition:
                    self.w -= self.LR * (2 * self.LP * self.w)

                else:
                    self.w -= self.LR * (2 * self.LP * self.w - np.dot(x_i, Y[idx]))
                    self.b -= self.LR * Y[idx]

    def predict(self,X,Y_test):
        
        print('PREDICTING')
        Y_pred= np.sign(np.dot(X,self.w) - self.b)
        
        score=accuracy_score(Y_test,Y_pred)
        return f'Accuracy: {round(score*100,2)}%'
        

a = SVM()

a.fit(tfidf_train,y_train)
print(a.predict(tfidf_test,y_test))