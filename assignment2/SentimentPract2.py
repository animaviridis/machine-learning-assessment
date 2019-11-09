#!/usr/bin/env python
import re
import random
import math
import collections
import itertools

import io_functions
import bayes

PRINT_ERRORS = 0



# ---------------------------End Training ----------------------------------

# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total = 0
    correct = 0
    totalpos = 0
    totalneg = 0
    totalpospred = 0
    totalnegpred = 0
    correctpos = 0
    correctneg = 0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score = 0
        for word in Words:
            if word in sentimentDictionary:
               score += sentimentDictionary[word]
 
        total += 1
        if sentiment == "positive":
            totalpos += 1
            if score >= threshold:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                correct += 0
                totalnegpred += 1
        else:
            totalneg += 1
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                correct += 0
                totalpospred += 1
 
    acc = correct/float(total)
    print(dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")
    precision_pos = correctpos/float(totalpospred)
    recall_pos = correctpos/float(totalpos)
    precision_neg = correctneg/float(totalnegpred)
    recall_neg = correctneg/float(totalneg)
    f_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos)
    f_neg = 2*precision_neg*recall_neg/(precision_neg+recall_neg)

    print(dataName + " Precision (Pos)=%0.2f" % precision_pos + " (%d" % correctpos + "/%d" % totalpospred + ")")
    print(dataName + " Recall (Pos)=%0.2f" % recall_pos + " (%d" % correctpos + "/%d" % totalpos + ")")
    print(dataName + " F-measure (Pos)=%0.2f" % f_pos)

    print(dataName + " Precision (Neg)=%0.2f" % precision_neg + " (%d" % correctneg + "/%d" % totalnegpred + ")")
    print(dataName + " Recall (Neg)=%0.2f" % recall_neg + " (%d" % correctneg + "/%d" % totalneg + ")")
    print(dataName + " F-measure (Neg)=%0.2f" % f_neg + "\n")


# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word] < 0.0000001:
            predictPower[word] = 1000000000
        else:
            predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            

    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print("NEGATIVE:")
    print(head)
    print("\nPOSITIVE:")
    print(tail)


# ---------- Main Script --------------------------

# initialise datasets and dictionaries
sentimentDictionary, sentencesTrain, sentencesTest, sentencesNokia = io_functions.read_files()

pWordPos = {}  # p(W|Positive)
pWordNeg = {}  # p(W|Negative)
pWord = {}     # p(W) 

# build conditional probabilities using training data
bayes.trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
# print("Naive Bayes")
# bayes.testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
bayes.testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
# bayes.testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)


# run sentiment dictionary based classifier on datasets
# testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
# testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)
# testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, -3)


# print most useful words
# mostUseful(pWordPos, pWordNeg, pWord, 50)



