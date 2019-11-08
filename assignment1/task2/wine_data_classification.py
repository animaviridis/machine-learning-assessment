import configparser
import numpy as np
import pandas as pd
import logging
import coloredlogs

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


# Initialise a decision tree
tree = Node(df_input, target_column=0)
tree.learn()
tree.print_terminal_labels()
