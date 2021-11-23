import sys
from collections import OrderedDict


def Accuracy(systemOpinion,humanOpinion):
    totall = len(systemOpinion) #2002
    correctDecision = 0
    for id in systemOpinion:
        if systemOpinion[id]==humanOpinion[id]:
            correctDecision += 1     
    return correctDecision/totall

def precisionRecallF1score(systemOpinion,humanOpinion):
    TruePositives  = 0 #TP : human and system say Real
    TrueNegatives  = 0 #TN : human and system say Fake
    FalsePositives = 0 #FP : human says Fake but system wrongly says Real
    FalseNegatives = 0 #FN : human says Real but system wrongly says Fake
    
    for id in systemOpinion:
        if systemOpinion[id]=="Real" and humanOpinion[id]=="Real":
            TruePositives += 1
        elif systemOpinion[id]=="Fake" and humanOpinion[id]=="Fake":
            TrueNegatives += 1
        elif systemOpinion[id]=="Real" and humanOpinion[id]=="Fake":
            FalsePositives += 1
        elif systemOpinion[id]=="Fake" and humanOpinion[id]=="Real":
            FalseNegatives += 1

    precision = TruePositives / (TruePositives + FalsePositives)
    recall = TruePositives / (TruePositives + FalseNegatives)
    f1score = 2*precision*recall / (precision+recall)

    return [precision,recall,f1score]








if __name__ == "__main__":
    errorMessage = "python evaluation.py <..input/file/path.txt>"
    if(len(sys.argv) != 2):
        print(errorMessage)
        sys.exit()
    else:
        inputFilePath = sys.argv[1]
    try:
        inputFile = open(inputFilePath) 
        Lines = inputFile.readlines()
        inputFile.close()
    except:
        print(errorMessage)

    systemOpinion = dict()
    for line in Lines:
        try:
            news_id = int(line.split()[0])
        except ValueError:
            continue
        real_or_fake = line.split()[1]
        systemOpinion[news_id] = real_or_fake
    systemOpinion = dict(OrderedDict(sorted(systemOpinion.items())))

    humanOpinion = dict()
    for id in range(1001): # 0 to 1000 are Real news
        humanOpinion[id] = "Real"
    for id in range(2000,3001): # 2000 to 3000 are Fake news
        humanOpinion[id] = "Fake"

    accuracy = Accuracy(systemOpinion,humanOpinion)
    precisionRecallF1score = precisionRecallF1score(systemOpinion,humanOpinion)
    precision = precisionRecallF1score[0]
    recall = precisionRecallF1score[1]
    f1score= precisionRecallF1score[2]
    
    print("accuracy ",accuracy)
    print('f1-score ',f1score)
    print('precision',precision)
    print('recall   ',recall)
    