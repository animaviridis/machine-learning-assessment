import re


PRINT_ERRORS = 0


def trainBayes(sentences_train):
    """calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data"""

    p_word_pos = {}  # p(W|Positive)
    p_word_neg = {}  # p(W|Negative)
    p_word = {}  # p(W)

    freq_positive = {}
    freq_negative = {}
    dictionary = {}
    pos_words_tot = 0
    neg_words_tot = 0
    all_words_tot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentences_train.items():
        wordList = re.findall(r"[\w']+", sentence)

        # TODO:
        # Populate bigramList (initialised below) by concatenating adjacent words in the sentence.
        # You might want to seperate the words by _ for readability, so bigrams such as:
        # You_might, might_want, want_to, to_seperate....

        bigramList = wordList.copy()  # initialise bigramList
        for x in range(len(wordList) - 1):
            bigramList.append(wordList[x] + "_" + wordList[x + 1])

        # -------------Finish populating bigramList ------------------#

        # TODO:
        #  when you have populated bigramList,
        #  uncomment out the line below and ,
        #  and comment out the unigram line to make use of bigramList rather than wordList

        for word in bigramList:  # calculate over bigrams
            # for word in wordList:  # calculate over unigrams
            all_words_tot += 1  # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment == "positive":
                pos_words_tot += 1  # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freq_positive):
                    freq_positive[word] = 1
                else:
                    freq_positive[word] += 1
            else:
                neg_words_tot += 1  # keeps count of total words in negative class

                #  keep count of each word in positive context
                if not (word in freq_negative):
                    freq_negative[word] = 1
                else:
                    freq_negative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freq_negative):
            freq_negative[word] = 1
        if not (word in freq_positive):
            freq_positive[word] = 1

        # Calculate p(word|positive)
        p_word_pos[word] = freq_positive[word] / float(pos_words_tot)

        # Calculate p(word|negative)
        p_word_neg[word] = freq_negative[word] / float(neg_words_tot)

        # Calculate p(word)
        p_word[word] = (freq_positive[word] + freq_negative[word]) / float(all_words_tot)

    return p_word_pos, p_word_neg, p_word


def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos):
    """INPUTS:
     sentencesTest is a dictionary with sentences associated with sentiment
     dataName is a string (used only for printing output)
     pWordPos is dictionary storing p(word|positive) for each word
        i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
     pWordNeg is dictionary storing p(word|negative) for each word
     pWord is dictionary storing p(word)
     pPos is a real number containing the fraction of positive reviews in the dataset
    """

    pNeg = 1 - pPos

    # These variables will store results (you do not need them)
    total = 0
    correct = 0
    totalpos = 0
    totalpospred = 0
    totalneg = 0
    totalnegpred = 0
    correctpos = 0
    correctneg = 0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)  # collect all words

        # TODO: Exactly what you did in the training function:
        # Populate bigramList by concatenating adjacent words in wordList.

        bigramList = wordList.copy()  # initialise bigramList
        for x in range(len(wordList) - 1):
            bigramList.append(wordList[x] + "_" + wordList[x + 1])

        # ------------------finished populating bigramList--------------
        pPosW = pPos
        pNegW = pNeg

        for word in bigramList:  # calculate over bigrams
            # for word in wordList:  # calculate over unigrams
            if word in pWord:
                if pWord[word] > 0.00000001:
                    pPosW *= pWordPos[word]
                    pNegW *= pWordNeg[word]

        prob = 0
        if pPosW + pNegW > 0:
            prob = pPosW / float(pPosW + pNegW)

        total += 1
        if sentiment == "positive":
            totalpos += 1
            if prob > 0.5:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                correct += 0
                totalnegpred += 1
                if PRINT_ERRORS:
                    print("ERROR (pos classed as neg %0.2f):" % prob + sentence)
        else:
            totalneg += 1
            if prob <= 0.5:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                correct += 0
                totalpospred += 1
                if PRINT_ERRORS:
                    print("ERROR (neg classed as pos %0.2f):" % prob + sentence)

    acc = correct / float(total)
    print(dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")

    precision_pos = correctpos / float(totalpospred)
    recall_pos = correctpos / float(totalpos)
    precision_neg = correctneg / float(totalnegpred)
    recall_neg = correctneg / float(totalneg)
    f_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
    f_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    print(dataName + " Precision (Pos)=%0.2f" % precision_pos + " (%d" % correctpos + "/%d" % totalpospred + ")")
    print(dataName + " Recall (Pos)=%0.2f" % recall_pos + " (%d" % correctpos + "/%d" % totalpos + ")")
    print(dataName + " F-measure (Pos)=%0.2f" % f_pos)

    print(dataName + " Precision (Neg)=%0.2f" % precision_neg + " (%d" % correctneg + "/%d" % totalnegpred + ")")
    print(dataName + " Recall (Neg)=%0.2f" % recall_neg + " (%d" % correctneg + "/%d" % totalneg + ")")
    print(dataName + " F-measure (Neg)=%0.2f" % f_neg + "\n")
