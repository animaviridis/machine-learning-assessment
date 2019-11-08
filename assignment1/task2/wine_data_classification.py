import configparser
import numpy as np
import pandas as pd
import logging
import coloredlogs

from sklearn.model_selection import KFold
from sklearn import metrics

from decisiontree import Node


# read the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# configure logger
logger = logging.getLogger(__name__)
coloredlogs.install(level=config['Logging']['level'], logger=logger.parent)

# load dataset
# a) load data headers, strip 3 initial characters (row number + closing bracket + optional whitespace)
data_headers_fname = config['Data']['data_headers']
logger.debug(f"Loading data headers from file: {data_headers_fname}")
headers = np.loadtxt(data_headers_fname, dtype=str, delimiter='\n', converters={0: lambda attr: attr[3:]})
logger.debug(f"Data headers: {headers}")

# b) load the data
data_fname = config['Data']['data_file']
logger.info(f"Loading data from file: {data_fname}")
df_input = pd.read_csv(data_fname, names=headers)

N_SPLITS = 10

logger.info(f"Decision tree learning and testing with {N_SPLITS}-fold cross validation")
splitter = KFold(n_splits=N_SPLITS, shuffle=True)

test_labels_true = []
test_labels_pred = []

for i, (train_idx, test_idx) in enumerate(splitter.split(df_input)):
    logger.info(f"Cross-validation round {i} with {len(train_idx)} train samples and {len(test_idx)} test samples")
    logger.debug(f"Test indices: {test_idx}")

    # Initialise a decision tree
    tree = Node(df_input, target_column=0, indices=train_idx)

    # perform learning
    tree.learn(max_depth=5)
    tree.print_terminal_labels()

    # prune
    tree.prune(min_points=2)
    tree.print_terminal_labels()

    true_i, pred_i = tree.test(df_input.iloc[test_idx])
    test_labels_true.extend(true_i)
    test_labels_pred.extend(pred_i)

cm = metrics.confusion_matrix(test_labels_true, test_labels_pred)
accuracy = cm.trace() / cm.sum()
logger.info(f"Total accuracy: {100*accuracy:.2f}% ({cm.trace()}/{cm.sum()} samples)")
