import re


def testDictionary(sentences_test, data_name, sentiment_dictionary, threshold):
    """This is a simple classifier that uses a sentiment dictionary to classify
    a sentence. For each word in the sentence, if the word is in the positive
    dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1.
    If the final score is above a threshold, it classifies as "Positive",
    otherwise as "Negative"
    """

    total = 0
    correct = 0
    totalpos = 0
    totalneg = 0
    totalpospred = 0
    totalnegpred = 0
    correctpos = 0
    correctneg = 0
    for sentence, sentiment in sentences_test.items():
        Words = re.findall(r"[\w']+", sentence)
        score = 0
        for word in Words:
            if word in sentiment_dictionary:
                score += sentiment_dictionary[word]

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

    acc = correct / float(total)
    print(data_name + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")
    precision_pos = correctpos / float(totalpospred)
    recall_pos = correctpos / float(totalpos)
    precision_neg = correctneg / float(totalnegpred)
    recall_neg = correctneg / float(totalneg)
    f_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
    f_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    print(data_name + " Precision (Pos)=%0.2f" % precision_pos + " (%d" % correctpos + "/%d" % totalpospred + ")")
    print(data_name + " Recall (Pos)=%0.2f" % recall_pos + " (%d" % correctpos + "/%d" % totalpos + ")")
    print(data_name + " F-measure (Pos)=%0.2f" % f_pos)

    print(data_name + " Precision (Neg)=%0.2f" % precision_neg + " (%d" % correctneg + "/%d" % totalnegpred + ")")
    print(data_name + " Recall (Neg)=%0.2f" % recall_neg + " (%d" % correctneg + "/%d" % totalneg + ")")
    print(data_name + " F-measure (Neg)=%0.2f" % f_neg + "\n")
