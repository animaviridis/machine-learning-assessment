import random
import re


def read_and_split(fname):
    with open(fname, 'r', encoding="ISO-8859-1") as f:
        content = re.split(r'\n', f.read())
    return content


def read_and_strip(fname):
    with open(fname, 'r', encoding="ISO-8859-1") as f:
        content = [line.strip() for line in f.readlines() if not line.startswith(";") and not line == '\n']
    return content


def make_pos_neg_dict(pos_list, neg_list, dictionary=None, pos=1, neg=-1):
    dictionary = dictionary or {}
    dictionary.update({i: pos for i in pos_list})
    dictionary.update({i: neg for i in neg_list})
    return dictionary


def read_files():
    # reading pre-labeled input and splitting into lines
    pos_sentences = read_and_split('data/rt-polarity-pos.txt')
    neg_sentences = read_and_split('data/rt-polarity-neg.txt')
    pos_sentences_nokia = read_and_split('data/nokia-pos.txt')
    neg_sentences_nokia = read_and_split('data/nokia-neg.txt')

    pos_word_list = read_and_strip('data/positive-words.txt')
    neg_word_list = read_and_strip('data/negative-words.txt')
    sentiment_dictionary = make_pos_neg_dict(pos_word_list, neg_word_list)

    # create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, to see how well
    # the model generalises to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels
    sentences_train = {}
    sentences_test = {}

    for i in pos_sentences:
        if random.randint(1, 10) < 2:
            sentences_test[i] = "positive"
        else:
            sentences_train[i] = "positive"

    for i in neg_sentences:
        if random.randint(1, 10) < 2:
            sentences_test[i] = "negative"
        else:
            sentences_train[i] = "negative"

    # create Nokia Datset:
    sentences_nokia = make_pos_neg_dict(pos_sentences_nokia, neg_sentences_nokia, pos='positive', neg='negative')

    return sentiment_dictionary, sentences_train, sentences_test, sentences_nokia


def mostUseful(pWordPos, pWordNeg, pWord, n):
    """Print out n most useful predictors"""

    predictPower = {}
    for word in pWord:
        if pWordNeg[word] < 0.0000001:
            predictPower[word] = 1000000000
        else:
            predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])

    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower) - n:]
    print("NEGATIVE:")
    print(head)
    print("\nPOSITIVE:")
    print(tail)


def compute_metrics(correct, total, total_pred):
    """Compute precision, recall, and F1 score with add-one smoothing"""

    precision = (correct + 1) / (total_pred + 1)
    recall = (correct + 1) / (total + 1)
    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def report_metrics(data_name, posneg, correct, total, total_pred):
    precision, recall, f_score = compute_metrics(correct, total, total_pred)
    print(f"{data_name} Precision ({posneg})={precision:.2f}"  + " (%d" % correct + "/%d" % total_pred + ")")
    print(f"{data_name} Recall ({posneg})={recall:.2f}" + " (%d" % correct + "/%d" % total + ")")
    print(f"{data_name} F-measure ({posneg})={f_score:.2f}")


def make_n_grams(sentence, n=2, sep='_'):
    word_list = re.findall(r"[\w']+", sentence)  # collect all words
    return [sep.join(word_list[i:i+n]) for i in range(len(word_list)-n+1)]
