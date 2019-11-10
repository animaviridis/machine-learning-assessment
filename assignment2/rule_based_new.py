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
        for word in words:
            if word in self.sentiment_dictionary:
                score += self.sentiment_dictionary[word]

        return score, score >= self.threshold

    def evaluate(self, sentences_test: dict, data_name: str):
        sentiments_true = list(map(lambda s: int(s == 'positive'), sentences_test.values()))
        sentiments_pred = len(sentiments_true) * [0]

        def pn(val):
            return 'pos' if val else 'neg'

        for i, (sentence, sentiment_true) in enumerate(sentences_test.items()):
            score, sentiment_pred = self.evaluate_sentence(sentence)
            sentiments_pred[i] = int(sentiment_pred)

            if self.print_errors:
                if sentiments_pred != sentiment_true:
                    print(f"ERROR ({pn(sentiment_true)} classed as {pn(sentiments_pred)}, score={score}): {sentence}")

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

