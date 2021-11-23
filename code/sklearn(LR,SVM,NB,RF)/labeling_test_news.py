import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

if __name__=="__main__":
    try:
        trainData = pd.read_csv("labeledNews.tsv",delimiter="\t")
        testData = pd.read_csv("unlabeledNews.tsv",delimiter="\t")
    except:
        print("Please first make 'labeledNews.tsv' by execution of 'make_tsv_file.py' program.")
        sys.exit()

    x_train = trainData.text.copy()
    y_train = trainData.reality.copy()

    x_test  = testData.text.copy()
    
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train.values.astype('str'))
    x_test  = vectorizer.transform(x_test.values.astype('str'))    
    
    SupportVectorMachines = svm.SVC()
    SupportVectorMachines.fit(x_train,y_train)
    predictions = SupportVectorMachines.predict(x_test)
    #print(predictions)
       
    for i in range(200):
        #print("newscontent_"+str(3001+i)+"\t"+str(predictions[i]))
        if predictions[i]==1:
            print("newscontent_"+str(3001+i)+"\tReal")
        else:
            print("newscontent_"+str(3001+i)+"\tFake")

    



    
            

   