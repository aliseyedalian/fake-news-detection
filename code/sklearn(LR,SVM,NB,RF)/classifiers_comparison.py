import pandas as pd
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB,ComplementNB
from sklearn.ensemble import RandomForestClassifier

def test():
    x_test = data.text[0:2002].copy()
    y_test = data.reality[0:2002].copy()
    vectorizer = TfidfVectorizer()
    x_test = vectorizer.fit_transform(x_test.values.astype('str'))         
    SupportVectorMachines = svm.SVC()
    SupportVectorMachines.fit(x_test,y_test)
    predictions = SupportVectorMachines.predict(x_test)
    print(len(predictions))
    accuracy = accuracy_score(y_test,predictions)
    precision = precision_score(y_test,predictions)
    recall = recall_score(y_test,predictions)
    f1 = f1_score(y_test,predictions)
    print(accuracy,precision,recall,f1)



''' #5-fold-cross-validation:
#real: 0-200 , 201-400 , 401-600 , 601-800 , 801-1000
#fake: 0-200 , 201-400 , 401-600 , 601-800 , 801-1000
#i=1:   test    train     train     train     train
#i=2:   train    test     train     train     train
#i=3:   train   train     test      train     train
#i=4:   train   train     train     test      train
#i=5:   train   train     train     train     test
'''
def train_test_5FoldCrossValidation(newsText,y,i): 
    if i==1: 
        x_test = newsText[0:201].copy().append(newsText[1001:1202].copy())
        y_test = y[0:201].copy().append(y[1001:1202].copy())
        x_train = newsText[201:1001].copy().append(newsText[1202:2002].copy())
        y_train = y[201:1001].copy().append(y[1202:2002].copy())
    if i==2: 
        x_test = newsText[201:401].copy().append(newsText[1202:1402].copy())
        y_test = y[201:401].copy().append(y[1202:1402].copy())
        x_train_part1 = newsText[0:201].copy().append(newsText[1001:1202].copy())
        y_train_part1 = y[0:201].copy().append(y[1001:1202].copy())
        x_train_part2 = newsText[401:1001].copy().append(newsText[1402:2002].copy())
        y_train_part2 = y[401:1001].copy().append(y[1402:2002].copy())
        x_train = x_train_part1.append(x_train_part2)
        y_train = y_train_part1.append(y_train_part2)
    if i==3:
        x_test = newsText[401:601].copy().append(newsText[1402:1602].copy())
        y_test = y[401:601].copy().append(y[1402:1602].copy())
        x_train_part1 = newsText[0:401].copy().append(newsText[1001:1402].copy())
        y_train_part1 = y[0:401].copy().append(y[1001:1402].copy())
        x_train_part2 = newsText[601:1001].copy().append(newsText[1602:2002].copy())
        y_train_part2 = y[601:1001].copy().append(y[1602:2002].copy())
        x_train = x_train_part1.append(x_train_part2)
        y_train = y_train_part1.append(y_train_part2)
    if i==4:
        x_test = newsText[601:801].copy().append(newsText[1602:1802].copy())
        y_test = y[601:801].copy().append(y[1602:1802].copy())
        x_train_part1 = newsText[0:601].copy().append(newsText[1001:1602].copy())
        y_train_part1 = y[0:601].copy().append(y[1001:1602].copy())
        x_train_part2 = newsText[801:1001].copy().append(newsText[1802:2002].copy())
        y_train_part2 = y[801:1001].copy().append(y[1802:2002].copy())
        x_train = x_train_part1.append(x_train_part2)
        y_train = y_train_part1.append(y_train_part2)
    if i==5:
        x_test = newsText[801:1001].copy().append(newsText[1802:2002].copy())
        y_test = y[801:1001].copy().append(y[1802:2002].copy())
        x_train = newsText[0:801].copy().append(newsText[1001:1802].copy())
        y_train = y[0:801].copy().append(y[1001:1802].copy())
            
    return x_train , x_test , y_train , y_test


def RunLogisticRegression():
    print("Logistic Regression:")
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    #5-fold-cross-validation:
    for i in range(1,6): 
        x_train,x_test,y_train,y_test = train_test_5FoldCrossValidation(data.text, data.reality, i)
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        #vectorizer = HashingVectorizer()
        x_train = vectorizer.fit_transform(x_train.values.astype('str'))
        x_test  = vectorizer.transform(x_test.values.astype('str'))
        
        logreg = LogisticRegression()
        logreg.fit(x_train,y_train)
        predictions = logreg.predict(x_test)

        sum_accuracy += accuracy_score(y_test,predictions)
        sum_precision += precision_score(y_test,predictions)
        sum_recall += recall_score(y_test,predictions)
        sum_f1 += f1_score(y_test,predictions)        
    print("\taccuracy = ",sum_accuracy/5)
    print("\tf1-score = ",sum_f1/5)
    print("\tprecision = ",sum_precision /5)
    print("\trecall = ",sum_recall/5)

def RunSupportVectorMachines():
    print("\nSVM:")
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    #5-fold-cross-validation:
    for i in range(1,6): 
        x_train,x_test,y_train,y_test = train_test_5FoldCrossValidation(data.text, data.reality, i)
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train.values.astype('str'))
        x_test  = vectorizer.transform(x_test.values.astype('str'))        
        SupportVectorMachines = svm.SVC()
        SupportVectorMachines.fit(x_train,y_train)
        predictions = SupportVectorMachines.predict(x_test)

        sum_accuracy += accuracy_score(y_test,predictions)
        sum_precision += precision_score(y_test,predictions)
        sum_recall += recall_score(y_test,predictions)
        sum_f1 += f1_score(y_test,predictions)        
    print("\taccuracy = ",sum_accuracy/5)
    print("\tf1-score = ",sum_f1/5)
    print("\tprecision = ",sum_precision /5)
    print("\trecall = ",sum_recall/5)

def RunMultinomialNaiveBayes():
    print("\nMultinomial Naive Bayes:")
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    #5-fold-cross-validation:
    for i in range(1,6): 
        x_train,x_test,y_train,y_test = train_test_5FoldCrossValidation(data.text, data.reality, i)
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train.values.astype('str'))
        x_test  = vectorizer.transform(x_test.values.astype('str'))        
        nb = MultinomialNB()
        nb.fit(x_train,y_train)
        predictions = nb.predict(x_test)

        sum_accuracy += accuracy_score(y_test,predictions)
        sum_precision += precision_score(y_test,predictions)
        sum_recall += recall_score(y_test,predictions)
        sum_f1 += f1_score(y_test,predictions)       
    print("\taccuracy = ",sum_accuracy/5)
    print("\tf1-score = ",sum_f1/5)
    print("\tprecision = ",sum_precision /5)
    print("\trecall = ",sum_recall/5)

def RunComplementNaiveBayes():
    print("\nComplement Naive Bayes:")
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    #5-fold-cross-validation:
    for i in range(1,6): 
        x_train,x_test,y_train,y_test = train_test_5FoldCrossValidation(data.text, data.reality, i)
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train.values.astype('str'))
        x_test  = vectorizer.transform(x_test.values.astype('str'))        
        nb = ComplementNB()
        nb.fit(x_train,y_train)
        predictions = nb.predict(x_test)

        sum_accuracy += accuracy_score(y_test,predictions)
        sum_precision += precision_score(y_test,predictions)
        sum_recall += recall_score(y_test,predictions)
        sum_f1 += f1_score(y_test,predictions)      
    print("\taccuracy = ",sum_accuracy/5)
    print("\tf1-score = ",sum_f1/5)
    print("\tprecision = ",sum_precision /5)
    print("\trecall = ",sum_recall/5)

def RunGaussianNaiveBayes():
    print("\nGaussian Naive Bayes:")
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    #5-fold-cross-validation:
    for i in range(1,6): 
        x_train,x_test,y_train,y_test = train_test_5FoldCrossValidation(data.text, data.reality, i)
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train.values.astype('str'))
        x_test  = vectorizer.transform(x_test.values.astype('str'))        
        nb = GaussianNB()
        nb.fit(x_train.toarray(),y_train)
        predictions = nb.predict(x_test.toarray())

        sum_accuracy += accuracy_score(y_test,predictions)
        sum_precision += precision_score(y_test,predictions)
        sum_recall += recall_score(y_test,predictions)
        sum_f1 += f1_score(y_test,predictions)      
    print("\taccuracy = ",sum_accuracy/5)
    print("\tf1-score = ",sum_f1/5)
    print("\tprecision = ",sum_precision /5)
    print("\trecall = ",sum_recall/5)

def RunRandomForest():
    print("\nRandom Forest:")
    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    #5-fold-cross-validation:
    for i in range(1,6): 
        x_train,x_test,y_train,y_test = train_test_5FoldCrossValidation(data.text, data.reality, i)
        #vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        x_train = vectorizer.fit_transform(x_train.values.astype('str'))
        x_test  = vectorizer.transform(x_test.values.astype('str'))        
        rf = RandomForestClassifier()
        rf.fit(x_train,y_train)
        predictions = rf.predict(x_test)

        sum_accuracy += accuracy_score(y_test,predictions)
        sum_precision += precision_score(y_test,predictions)
        sum_recall += recall_score(y_test,predictions)
        sum_f1 += f1_score(y_test,predictions)        
    print("\taccuracy = ",sum_accuracy/5)
    print("\tf1-score = ",sum_f1/5)
    print("\tprecision = ",sum_precision /5)
    print("\trecall = ",sum_recall/5)



if __name__=="__main__":
    try:
        data = pd.read_csv("labeledNews.tsv",delimiter="\t")
    except:
        print("Please first make tsv files by execution of 'make_tsv_files.py' program.")
        sys.exit()
    
    #test()

    RunLogisticRegression()
    RunSupportVectorMachines()
    RunMultinomialNaiveBayes()
    RunComplementNaiveBayes()
    RunGaussianNaiveBayes()
    RunRandomForest()


    
            

   