import configparser
import pandas as pd


# read the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

# split dataset into
df_input = pd.read_excel(config['Data']['data_file'])
target_var = df_input.keys()[int(config['Data']['target_column'])]
df_target = df_input.loc[:, target_var]
df_input.drop(target_var, axis=1, inplace=True)
print(df_input.head())
