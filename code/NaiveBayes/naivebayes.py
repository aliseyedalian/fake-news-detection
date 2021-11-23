import json
import numpy as np
import operator


def C(word,doc): #count word in document
    return doc.count(word)

def P_thetaFake(realTrain,fakeTrain):
    return len(fakeTrain)/(len(realTrain)+len(fakeTrain))
    
def P_thetaReal(realTrain,fakeTrain):
    return len(realTrain)/(len(realTrain)+len(fakeTrain))
    
def P_w_thetaFake(word,fakeTrain):
    mu = 0.9
    numerator = 0
    denominator = 0
    for newsId in fakeTrain:
        numerator += C(word,fakeTrain[newsId])  
        denominator += len(fakeTrain[newsId])
    numerator += mu * P_w_bg(word)
    denominator += mu
    return numerator/denominator

def P_w_thetaReal(word,realTrain):
    mu = 0.9
    numerator = 0
    denominator = 0
    for newsId in realTrain:
        numerator += C(word,realTrain[newsId])  
        denominator += len(realTrain[newsId])
    numerator += mu * P_w_bg(word)
    denominator += mu
    return numerator/denominator

def P_w_bg(term):#background 
    return termFrequency[term]/CollectionSize

def naiveBayes(test,realTrain,fakeTrain):
    p_thetaReal = P_thetaReal(realTrain,fakeTrain)
    p_thetaFake = P_thetaFake(realTrain,fakeTrain)
    result = dict()
    for newsId in test: 
        doc = test[newsId]
        score = np.log(p_thetaReal/p_thetaFake)     
        for word in selectedWords: 
            score += C(word,doc) * np.log(P_w_thetaReal(word,realTrain)/P_w_thetaFake(word,fakeTrain))
        if score > 0 :
            result[newsId] = "Real"
        else:
            result[newsId] = "Fake"
        print(newsId+'\t'+result[newsId])
    return result

def featureSelection_tf():
    #select words with most frequency in collection
    number = 150 #number of selected words
    sortedTermFrequency = {k: v for k, v in sorted(termFrequency.items(),reverse=True, key=lambda item: item[1])}
    selectedWords = {k: sortedTermFrequency[k] for k in list(sortedTermFrequency)[:number]} #first n items in dict
    return selectedWords

def featureSelection_df():
    #select words with most df 
    number = 150 #number of selected words
    docFrequency = dict()
    for word in wordDocs:
        df = len(wordDocs[word])
        docFrequency[word] = df
    sortedDocFrequency = {k: v for k, v in sorted(docFrequency.items(),reverse=True, key=lambda item: item[1])}
    selectedWords = {k: sortedDocFrequency[k] for k in list(sortedDocFrequency)[:number]} #first n items in dict
    return selectedWords

def featureSelection_tfidf():
    #select words with most amount of tfidf 
    number = 150 #number of selected words
    wordTFIDF = dict()
    for word in wordDocs:
        tf = termFrequency[word]
        idf = 1/len(wordDocs[word])
        wordTFIDF[word] = tf*idf
    sortedTFIDF = {k: v for k, v in sorted(wordTFIDF.items(),reverse=True, key=lambda item: item[1])}
    selectedWords = {k: sortedTFIDF[k] for k in list(sortedTFIDF)[:number]} #first n items in dict
    return selectedWords

def featureSelection_tfdf():
    #select words with most amount of tfidf 
    number = 150 #number of selected words
    wordTFDF = dict()
    for word in wordDocs:
        tf = termFrequency[word]
        df = len(wordDocs[word])
        wordTFDF[word] = np.log(tf*df)
    sortedTFDF = {k: v for k, v in sorted(wordTFDF.items(),reverse=True, key=lambda item: item[1])}
    selectedWords = {k: sortedTFDF[k] for k in list(sortedTFDF)[:number]} #first n items in dict
    return selectedWords

def featureSelection_giniindex():
    #select words with most gini index
    number = 400 #number of selected words
    min_df = 25
    wordGini = dict()
    for word in wordDocs:    
        df = len(wordDocs[word])
        if(df < min_df): #ignore this word and go to next word
            continue 
        countRealDocs = 0
        countFakeDocs = 0        
        for docId in wordDocs[word]:
            if str(docId) in realNews:
                countRealDocs += 1
            if str(docId) in fakeNews:
                countFakeDocs += 1
        #print("countFakeDocs",countFakeDocs)
        #print("countRealDocs",countRealDocs)
        if countFakeDocs==countRealDocs:
            continue
        p_real_word = countRealDocs/df
        p_fake_word = countFakeDocs/df        
        wordGini[word] = p_real_word**2 + p_fake_word**2   
    featureWords = list(dict(sorted(wordGini.items(), key=operator.itemgetter(1),reverse=True)).keys())[:number]       
    return featureWords


'''
#5-fold-cross-validation:
#real: 0-200 , 201-400 , 401-600 , 601-800 , 801-1000
#fake: 0-200 , 201-400 , 401-600 , 601-800 , 801-1000
#i=0:   test    train     train     train     train
#i=1:   train    test     train     train     train
#i=2:   train   train     test      train     train
#i=3:   train   train     train     test      train
#i=4:   train   train     train     train     test
'''
def k_fold_cross_validation():
    for i in range(5): 
        if i==0:         
            test = {**dict(list(realNews.items())[0:201]),**dict(list(fakeNews.items())[0:201])}
            realTrain = dict(list(realNews.items())[201:1001])
            fakeTrain = dict(list(fakeNews.items())[201:1001])
            result0 = naiveBayes(test,realTrain,fakeTrain)   
        if i==1:
            test = {**dict(list(realNews.items())[201:401]),**dict(list(fakeNews.items())[201:401])}
            realTrain = {**dict(list(realNews.items())[0:201]),**dict(list(realNews.items())[401:1001])}
            fakeTrain = {**dict(list(fakeNews.items())[0:201]),**dict(list(fakeNews.items())[401:1001])}
            result1 = naiveBayes(test,realTrain,fakeTrain)   
        if i==2: 
            test = {**dict(list(realNews.items())[401:601]),**dict(list(fakeNews.items())[401:601])}
            realTrain = {**dict(list(realNews.items())[0:401]),**dict(list(realNews.items())[601:1001])}
            fakeTrain = {**dict(list(fakeNews.items())[0:401]),**dict(list(fakeNews.items())[601:1001])}
            result2 = naiveBayes(test,realTrain,fakeTrain)   
        if i==3:
            test = {**dict(list(realNews.items())[601:801]),**dict(list(fakeNews.items())[601:801])}
            realTrain = {**dict(list(realNews.items())[0:601]),**dict(list(realNews.items())[801:1001])}
            fakeTrain = {**dict(list(fakeNews.items())[0:601]),**dict(list(fakeNews.items())[801:1001])}
            result3 = naiveBayes(test,realTrain,fakeTrain) 
        if i==4:
            test = {**dict(list(realNews.items())[801:1001]),**dict(list(fakeNews.items())[801:1001])}
            realTrain = dict(list(realNews.items())[0:801])
            fakeTrain = dict(list(fakeNews.items())[0:801])
            result4 = naiveBayes(test,realTrain,fakeTrain) 
    result = {**result0 , **result1 ,**result2 , **result3, **result4}  #concatenating result dictionaries
    return result


if __name__ == "__main__":
    with open("index/realNews.json") as realnews_jsonfile:
        realNews = json.load(realnews_jsonfile)
        realnews_jsonfile.close()
    with open("index/fakeNews.json") as fakenews_jsonfile:
        fakeNews = json.load(fakenews_jsonfile)
        fakenews_jsonfile.close()
    with open("index/termFrequency.json") as WF_jsonfile:
        termFrequency = json.load(WF_jsonfile)      
        WF_jsonfile.close()
    with open("index/wordDocs.json") as WD_jsonfile:
        wordDocs = json.load(WD_jsonfile)
        WD_jsonfile.close()


    CollectionSize = sum(termFrequency.values())


    #selectedWords = featureSelection_tf()
    #selectedWords = featureSelection_df()
    #selectedWords = featureSelection_tfidf()
    #selectedWords = featureSelection_tfdf()
    selectedWords = featureSelection_giniindex()
    
    k_fold_cross_validation()

    
        
    

            