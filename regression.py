import configparser
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging


# read the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# configure logger
logging.basicConfig(level=config['Logging']['level'])
logger = logging.getLogger(__name__)


# load data dataset
data_fname = config['Data']['data_file']
logger.info(f"Loading data from file {data_fname}")
df_input = pd.read_excel(data_fname)
logger.info("Loading completed")
print(df_input.head())

# split dataset into input and target
target_var = df_input.keys()[int(config['Data']['target_column'])]
df_target = df_input.loc[:, target_var]
df_input.drop(target_var, axis=1, inplace=True)

