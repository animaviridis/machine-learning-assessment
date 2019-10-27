import configparser
import pandas as pd


# read the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

df = pd.read_excel(config['Data']['data_file'])
print(df.head())
