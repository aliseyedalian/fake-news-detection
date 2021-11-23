import json
import nltk
import sys
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

def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)

def textPurify(impureText):
    pure = [word for word in nltk.word_tokenize(strip_tags(impureText.lower())) if word not in stopwords.words('english')]
    pure = [PorterStemmer().stem(word) for word in pure]
    for w in pure:
        if w in string.punctuation:
            pure.remove(w)            
    return pure 

def processLabeledNews():
    print("processing real news ...")
    realNews = dict()
    fakeNews = dict()
    for news_id in range(0,1001):
        path = "../../DataSet/Real/newscontent_" + str(news_id)
        with open(path) as newsfile:
            doc = json.loads(newsfile.readline()) 
            docTextlist = concatenat(textPurify(doc['text']),textPurify(doc['title'])) #text+title of news
            docText = convert_list_to_string(docTextlist)
            realNews[news_id] = docText
            newsfile.close()
    
    print("processing fake news ...")
    for news_id in range(2000,3001):
        path = "../../DataSet/Fake/newscontent_" + str(news_id)
        with open(path) as newsfile:
            doc = json.loads(newsfile.readline()) 
            docTextlist = concatenat(textPurify(doc['text']),textPurify(doc['title'])) #text+title of news
            docText = convert_list_to_string(docTextlist)
            fakeNews[news_id] = docText            
            newsfile.close()
    #saving:
    with open("labeledNews.tsv", "w") as record_file:
        record_file.write("id\treality\ttext\n")
        for news_id in realNews:
            record_file.write('"%s"\t%s\t"%s"\n' % (news_id,1,realNews[news_id]))
        for news_id in fakeNews:
            record_file.write('"%s"\t%s\t"%s"\n' % (news_id,0,fakeNews[news_id]))

def processUnlabeledNews():
    print("processing unlabeled news ...")
    unlabeledNews = dict()
    for news_id in range(3001,3201):
        path = "../../DataSet/Test/newscontent_" + str(news_id)
        with open(path) as newsfile:
            doc = json.loads(newsfile.readline()) 
            docTextlist = concatenat(textPurify(doc['text']),textPurify(doc['title'])) #text+title of news
            docText = convert_list_to_string(docTextlist)
            unlabeledNews[news_id] = docText            
            newsfile.close()
    #saving:
    with open("unlabeledNews.tsv", "w") as record_file:
        record_file.write("id\ttext\n")
        for news_id in unlabeledNews:
            record_file.write('"%s"\t"%s"\n' % (news_id,unlabeledNews[news_id]))
        record_file.close()


if __name__ == "__main__":
    processLabeledNews()
    processUnlabeledNews()


    

    

            

     
        


        
    




