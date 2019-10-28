import configparser
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
import matplotlib.pyplot as plt

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
logger.info(f"Loading completed. Data shape: {df_input.shape}")
print(df_input.head())

# split dataset into input and target
target_var = df_input.keys()[int(config['Data']['target_column'])]
df_target = df_input.loc[:, target_var]
df_input.drop(target_var, axis=1, inplace=True)

input_vars = list(df_input.keys())
n = len(input_vars)
fig, axes = plt.subplots(2, n//2 + n%2, figsize=(12, 10))
for i, input_var in enumerate(input_vars):
    ax = axes[i%2, i//2]
    ax.plot(df_input.loc[:, input_var], '.', color='darkblue', alpha=0.9)
    ax.grid(color='lightgray')
    ax.set_xlabel(input_var[:input_var.find('(')])
    if not i//2:
        ax.set_ylabel(target_var[:target_var.find('(')])
fig.tight_layout()
plt.show()


# fit data
fitter = LinearRegression()
logger.debug("Fitting the data...")
fitter.fit(df_input, df_target)
logger.info(f"Fitting completed. Score: {fitter.score(df_input, df_target):.3f}")
