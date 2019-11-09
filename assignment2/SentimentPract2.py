#!/usr/bin/env python

import aux_functions as aux
import bayes
import rule_based

# initialise datasets and dictionaries
sentimentDictionary, sentencesTrain, sentencesTest, sentencesNokia = aux.read_files()

# build conditional probabilities using training data
pWordPos, pWordNeg, pWord = bayes.trainBayes(sentencesTrain)

# run naive bayes classifier on datasets
# print("Naive Bayes")
# bayes.testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
bayes.testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
# bayes.testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)


# run sentiment dictionary based classifier on datasets
# rule_based.testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, -4)
# rule_based.testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, -4)
# rule_based.testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, -3)


# print most useful words
# mostUseful(pWordPos, pWordNeg, pWord, 50)



