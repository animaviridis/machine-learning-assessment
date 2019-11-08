import configparser
import numpy as np
import pandas as pd
import logging
import coloredlogs
import itertools
from tqdm import tqdm

import aux_functions as aux


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
tree_params = dict(max_depth=range(3, 10), min_points=range(1, 5))
tree_results = []

tree_params_keys = tree_params.keys()
tree_params_prod = itertools.product(*tree_params.values())
for params in tqdm(tree_params_prod):
    params_dict = dict(zip(tree_params_keys, params))
    xv = aux.cross_validate_tree(N_SPLITS, df_input)
    params_dict.update(xv)
    tree_results.append(params_dict)

best_result = tree_results[np.argmax([t['f1_score'] for t in tree_results])]
print(best_result)
