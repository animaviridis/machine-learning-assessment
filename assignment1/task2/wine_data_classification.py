import configparser
import numpy as np
import pandas as pd
import logging
import coloredlogs

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import aux_functions as aux
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

# Define exemplary tree
tree = Node(df_input, target_column=int(config['Data']['target_column']))
tree.learn(max_depth=5)
tree.prune(min_points=2)

# perform classification using DT
N_SPLITS = 10
tree_params = dict(max_depth=range(3, 10), min_points=range(1, 5))

# all_results, best_result = aux.tune_params(aux.cross_validate_tree, tree_params, func_args=(N_SPLITS, df_input),
#                                            scoring_metrics='f1_score')
# print(all_results)
# print("Best result: ", best_result)


df_y = tree.class_labels
df_x = tree.data[tree.input_attributes]

gs = aux.make_grid_searcher(df_x, df_y, N_SPLITS)

# Random forest
logger.info("Tuning random forest classifier")
rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf_params = dict(max_depth=range(3, 10), min_samples_leaf=range(1,5))
print(gs(rf, rf_params))

# shallow NN
logger.info("Tuning shallow NN classifier")
snn = MLPClassifier()
snn_params = dict(hidden_layer_sizes=[(100,), (200,), (500,)], alpha=[0.001, 0.0001, 0.00001])
print(gs(snn, snn_params))

# deep NN
logger.info("Tuning deep NN classifier")
dnn = MLPClassifier()
dnn_params = dict(hidden_layer_sizes=[100*(10,), 50*(20,), 20*(50,)], alpha=[0.001, 0.0001, 0.00001])
print(gs(dnn, dnn_params))
