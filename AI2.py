import csv
from os import close
from tkinter import *
from tkinter.ttk import *
from typing import List
from tkinter import filedialog
import time
import SVM_Predict as svm
from sklearn.feature_extraction.text import TfidfVectorizer 

root = Tk()
root.title("News Reader")
root.geometry("800x800")

File = open("news.csv", encoding = 'utf8')
Reader = csv.reader(File)
Data = list(Reader)
del(Data[0])


T= Text(root, height=30, width=100, font='TkDefaultFont 8', wrap=WORD)
T.grid(row=1, column=1)
T.insert(END, " Some text") 
'''listbox1 = Listbox(root, listvariable = var)
listbox1.config( width = 100, height = 10, background = "azure", foreground = "red", font = ('TkDefaultFont 8') )
listbox1.grid(row = 1 , column =1)'''

label =  Label(root, anchor=CENTER, text = "Fake News Detection Using AI ", font = "Castellar 15")  
label.grid(row=0, column=1)

a = svm.SVM()

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


def learn(event=None):
    
    xtrain,ytrain = svm.get_from_csv()
    xtrain = tfidf_train=tfidf_vectorizer.fit_transform(xtrain)
    xtrain = xtrain.toarray() 
    a.fit(xtrain,ytrain)

    print('Learned')
    learn_label = Label(root, text="Learned", font=('TkDefaultFont', 8 )).grid(row=5, columnspan=3, pady=10)
    
def predit(event=None):

    xtest = svm.read_text()
    
    xtest = tfidf_test=tfidf_vectorizer.transform(xtest)
    xtest = xtest.toarray()
    prediction = a.predict(xtest)
    print(prediction)
    P = Label(root, text="Prediction:"+prediction).grid(row=6, columnspan=3, pady=10)
    #P.grid(row=5, column=3)
    

def update(event=None):
    
    inputValue= T.get("1.0","end-1c")

    print('Updated')
    with open('new.txt', 'w', encoding='utf-8') as f:
        for i in range(1):
            f.write(inputValue)
    
    pb1 = Progressbar(root, 
        orient=HORIZONTAL, 
        length=300, 
        mode='determinate'
        )
    pb1.grid(row=4, columnspan=3, pady=20)
    for i in range(5):
        root.update_idletasks()
        pb1['value'] += 20
        time.sleep(1)
    pb1.destroy()
    Label(root, text='File Written Successfully!', foreground='green').grid(row=4, columnspan=3, pady=10)
     
    #index = listbox1.curselection()[0]
    #print(index)
    #return None

button1 = Button(root, text = "Update", command = update)
button1.grid(row=1, column=9, padx=20)

button2 = Button(root, text='Exit', command=root.destroy)
button2.grid(row=4, column=9,padx = 20 )

button3 = Button(root, text='Learn', command=learn)
button3.grid(row=2, column=9, padx= 20)

button4 = Button(root, text='Predict', command= predit)
button4.grid(row=3, column=9, padx=20)



root.mainloop()