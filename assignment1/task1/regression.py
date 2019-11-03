import configparser
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
import matplotlib.pyplot as plt

# read the configuration file
config = configparser.ConfigParser()
config.read("../../config.ini")

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(level=config['Logging']['level'])

# load data dataset
data_fname = config['RegressionData']['data_file']
logger.info(f"Loading data from file: {data_fname}")
df_input = pd.read_excel(data_fname)

# trim the variable names to the first occurrence of an opening bracket
df_input.rename(columns={key: key[:key.find('(')] for key in df_input.keys()}, inplace=True)
logger.info(f"Loading completed. Data shape: {df_input.shape}")
print(df_input.head())

# split dataset into input and target
target_var = df_input.keys()[int(config['RegressionData']['target_column'])]
df_target = df_input.loc[:, target_var]
df_input.drop(target_var, axis=1, inplace=True)

input_vars = list(df_input.keys())
n = len(input_vars)

nt = n+1
fig, axes = plt.subplots(nt//3 + (1 if nt%3 else 0), 3, figsize=(8, 6))
for i, input_var in enumerate(input_vars):
    ax = axes[i//3, i%3]
    ax.hist(df_input.loc[:, input_var], color='darkblue', lw=1)
    ax.set_xlabel(input_var)
axes[-1, n%3].hist(df_target, color='crimson')
fig.tight_layout()
plt.show()


fig, axes = plt.subplots(2, n//2 + n%2, figsize=(12, 6))
for i, input_var in enumerate(input_vars):
    ax = axes[i%2, i//2]
    ax.plot(df_input.loc[:, input_var], '.', color='darkblue', alpha=0.3)
    ax.grid(color='lightgray')
    ax.set_xlabel(input_var)
    if not i//2:
        ax.set_ylabel(target_var)
fig.tight_layout()
plt.show()


# fit data
fitter = LinearRegression()
logger.debug("Fitting the data...")
fitter.fit(df_input, df_target)
logger.info(f"Fitting completed. Score: {fitter.score(df_input, df_target):.3f}")
