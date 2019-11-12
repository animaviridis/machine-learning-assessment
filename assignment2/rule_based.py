import re


import aux_functions as aux


def testDictionary(sentences_test, data_name, sentiment_dictionary, threshold, print_errors=False):
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
                totalnegpred += 1
                if print_errors:
                    print(f"ERROR (pos classed as neg, score={score}): {sentence}")
        else:
            totalneg += 1
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1
                if print_errors:
                    print(f"ERROR (neg classed as pos, score={score}): {sentence}")

    acc = correct / float(total)
    print(data_name + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")
    aux.report_metrics(data_name, 'Pos', correctpos, totalpos, totalpospred)
    aux.report_metrics(data_name, 'Neg', correctneg, totalneg, totalnegpred)
