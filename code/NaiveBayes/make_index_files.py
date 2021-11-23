import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from io import StringIO
from html.parser import HTMLParser
import string
#nltk.download('punkt')
#nltk.download('stopwords')



class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
 
def concatenat(list1,list2):
    for item in list2:
        list1.append(item)
    return list1

def add2termFrequency(word,count):                    
    if word not in termFrequency:
        termFrequency[word] = 0
    termFrequency[word] += count

def add2wordDocs(word,docId):
    if word not in wordDocs:
        wordDocs[word] = list()
    wordDocs[word].append(docId)

def textPurify(impureText):
    pure = [word for word in nltk.word_tokenize(strip_tags(impureText.lower())) if word not in stopwords.words('english')]
    pure = [PorterStemmer().stem(word) for word in pure]
    for w in pure:
        if w in string.punctuation:
            pure.remove(w)
    return pure 

def preprocessNews():
    print("preprocessing real news ...")
    for news_id in range(0,1001):
        path = "../../DataSet/Real/newscontent_" + str(news_id)
        with open(path) as newsfile:
            doc = json.loads(newsfile.readline()) 
            docText = concatenat(textPurify(doc['text']),textPurify(doc['title'])) #text+title of news
            realNews[news_id] = docText
            for word in set(docText):
                countWord = docText.count(word)
                add2termFrequency(word,countWord)
                add2wordDocs(word,news_id)
            newsfile.close()
    
    print("preprocessing fake news ...")
    for news_id in range(2000,3001):
        path = "../../DataSet/Fake/newscontent_" + str(news_id)
        with open(path) as newsfile:
            doc = json.loads(newsfile.readline()) 
            docText = concatenat(textPurify(doc['text']),textPurify(doc['title'])) #text+title of news
            fakeNews[news_id] = docText
            for word in set(docText):
                countWord = docText.count(word)                
                add2termFrequency(word,countWord)
                add2wordDocs(word,news_id)
            newsfile.close()
    



if __name__ == "__main__":
    realNews = dict()       #newsId --> newstext
    fakeNews = dict()       #newsId --> newstext
    termFrequency = dict()  #word -->  word frequency in collection
    wordDocs = dict()       #word -->  list of documents contain this word
    
    preprocessNews()

    #store results:
    print("result stored in json files.")
    with open("index/realNews.json",'w') as json_file:
        json.dump(realNews,json_file)
        json_file.close()

    with open("index/fakeNews.json",'w') as json_file:
        json.dump(fakeNews,json_file)
        json_file.close()

    with open("index/termFrequency.json",'w') as json_file:
        json.dump(termFrequency,json_file)
        json_file.close()

    with open("index/wordDocs.json",'w') as json_file:
        json.dump(wordDocs,json_file)
        json_file.close()


    

    

            

     
        


        
    




