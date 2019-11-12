"""New implementation of the rule-based system"""

import re
from sklearn import metrics

import aux_functions as aux


class RuleBasedSentimentAnalyser(object):
    def __init__(self, sentiment_dictionary, threshold=0, print_errors=False):
        self.sentiment_dictionary = sentiment_dictionary
        self.but_words = aux.read_and_split('data/but-words.txt')
        self.negation_words = aux.read_and_split('data/negation-words.txt')
        self.threshold = threshold
        self.print_errors = print_errors

    def evaluate_sentence(self, sentence):
        """Determine whether a sentence is positive or negative (for now, follow the original implementation)"""

        words = re.findall(r"[\w']+", sentence)
        score = 0
        flag = 1

        for word in words:
            if word in self.sentiment_dictionary:
                # update the score according to sentiment associated with the given word
                score += flag * self.sentiment_dictionary[word]
                flag = 1

            elif word in self.but_words:
                # invert and rescale the score for the first part of the sentence
                # the part after a 'but' is likely to carry opposite sentiment, more important to the opinion holder
                score = - 0.5*score
                flag = 1

            elif word in self.negation_words:
                # make the next known word carry opposite sentiments
                flag = -1

        return score, score >= self.threshold

    def evaluate(self, sentences_test: dict, data_name: str):
        sentiments_true = list(map(lambda s: int(s == 'positive'), sentences_test.values()))
        sentiments_pred = len(sentiments_true) * [0]

        def pn(val):
            return 'pos' if val else 'neg'

        for i, sentence in enumerate(sentences_test.keys()):
            s_true = sentiments_true[i]
            score, s_pred = self.evaluate_sentence(sentence)
            sentiments_pred[i] = int(s_pred)

            if self.print_errors:
                if s_pred != bool(s_true):
                    print(f"ERROR ({pn(s_true)} classed as {pn(s_pred)}, score={score}): {sentence}")

        self.report_results(data_name, sentiments_true, sentiments_pred)

    @staticmethod
    def report_results(data_name, y_true, y_pred):
        cm = metrics.confusion_matrix(y_true, y_pred)
        correct = cm.trace()
        total = cm.sum()

        print(f"{data_name} Accuracy (All)={correct/total:.2f} ({correct}/{total})\n")
        for k, label in enumerate(['Pos', 'Neg']):
            i = 1 - k
            aux.report_metrics(data_name, label, cm[i, i], cm[i, :].sum(), cm[:, i].sum())

