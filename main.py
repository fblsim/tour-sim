import pandas as pd
import time

from scraper import get_ratings

from object import Tournament

import matplotlib.pyplot as plt
import seaborn as sns

#%%
#path = 'worldcup2018/'
path = 'euro2021/'

# dataframes for group fixtures
df_groups = pd.read_csv( path + "groups.csv", index_col = None)
df_fixtures = pd.read_csv(path + "df_fixtures.csv")

# dataframe for tree/bracket
tree = pd.read_csv(path + "tree.csv", index_col = None).set_index('id')

# datframe for seeding of third teams
thirdseed = pd.read_csv( path + "3rdseeding.csv", index_col = 0)
thirdseed.columns = thirdseed.columns.astype('int')

# get elo ratings
df_elo = get_ratings()


#%%

name = 'euro2021'
mode = 'euro'


input = {'df_groups': df_groups, 'df_fixtures': df_fixtures, 'tree': tree, 'df_elo': df_elo, 'thirdseed': thirdseed}

T = Tournament(name, mode, input, verbose = False)

start = time.time()
T.simulate(N = 200)
end = time.time()

T.plot_result()
T.plot_matrix()

T.plot_opps('Germany')

P = T.pmatrix
E = T.xscore_matrix






