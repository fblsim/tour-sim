import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange
import string
#from scipy.stats import skellam


plt.style.use("dark_background")

# home effect: is added to xg-difference (of team playing at home)
home_xg = .2

def xs2xg(s):
    # exp. score to xg diff.
    return  (1/1.14145407) * np.log(s/(1-s))

def xg2xs(g):
    # xg diff. to xscore
    return 1/ (1+ np.exp(-1.14145407*g))

def elo_expected_score(elodiff):
    """
    returns: expected score of team 1
    elodiff: given by ELO2-ELO1
    """
    return 1/( 1+10**(elodiff/400))

class Tournament():
    def __init__(self, name, mode, input, verbose = False):
        
        self.name = name
        self.mode = mode
        self.verbose = verbose
        
        self.df_groups = input['df_groups']
        self.df_fixtures = input['df_fixtures']
        self.tree = input['tree']
        
        self.all_teams = list(self.df_groups.values.ravel())
        self.all_teams.sort()
        
        self.df_elo = input['df_elo'].loc[self.all_teams]
        
        self.xscore_matrix()
        
        if mode == 'euro':
            self.thirdseed = input['thirdseed']
        else:
            self.thirdseed = None
            
        return
    
    def xscore_matrix(self):
        a = self.df_elo.ELO.values
        p = len(self.all_teams)
        tmp = np.tile(a, p ).reshape((p,p))
    
        diff = pd.DataFrame( tmp.T- tmp, columns = self.all_teams, index = self.all_teams)
        self.xscore_matrix = diff.apply(elo_expected_score)
        
        return 
    
    def simulate(self, N=100):
        
        pmatrix = pd.DataFrame()
        all_groups = list()
        all_trees = list()

        #for j in np.arange(N):
        for j in trange(N, leave = True):
            
            
            res_wide, res_tree, groups = simulate_tournament(self.df_elo, self.df_groups, self.df_fixtures, self.tree, self.thirdseed, \
                                                           mode = self.mode, verbose = self.verbose)
            
                
            pmatrix = pmatrix.add(res_wide, fill_value = 0)
            all_groups.append(groups)
            all_trees.append(res_tree)
            
        self.all_groups = all_groups
        self.all_trees = all_trees
        self.pmatrix = pmatrix/N
        
        return
    
    def plot_result(self):
        
        if len(self.pmatrix.columns) == 5:
            xticklabels = ["Round of 16", "Round of 8", "Semifinal", "Final", "Winner"]
        else:
            xticklabels = np.arange(len(self.pmatrix.columns))
            
        fig, ax = plt.subplots()
        
        cmap = 'Blues' #'RdYlGn'
        sns.heatmap(self.pmatrix * 100 , annot = True, ax = ax,
                            cmap = cmap, cbar = False, vmax = 100, vmin = -10,
                            xticklabels = xticklabels )
        ax.xaxis.set_ticks_position('top') 
        ax.set_title('Simulated tournament results' , pad = 30)

    def plot_matrix(self):
        fig, ax = plt.subplots()
        fig.suptitle("Expected score matrix of row vs. column")
        
        cmap = 'Blues' #'RdYlGn'
        p = sns.heatmap(np.round(self.xscore_matrix,2).T , annot = True, ax = ax,
                            cmap = cmap, cbar = False, vmax = 1, vmin = 0)
        
        p.set_xticklabels(p.get_xticklabels(), rotation = 30, fontsize = 10)
        p.set_yticklabels(p.get_yticklabels(), rotation = 30, fontsize = 10)
        return
    
    def plot_opps(self, team):

        all_ko = list()
        for j in range(len(self.all_trees)):
            tr = self.all_trees[j]
            
            tmp = tr[tr.team1 == team][['stage', 'team2', 'winner']].values
            all_ko.append(tmp)
        
            tmp = tr[tr.team2 == team][['stage', 'team1', 'winner']].values
            all_ko.append(tmp)
        
        
        all_ko = pd.DataFrame(np.vstack(all_ko), columns = ['stage', 'opponent', 'winner'])
        
        # data manipulation
        all_ko['stage'] = pd.to_numeric(all_ko['stage'])
        all_ko['winner'] = (all_ko['winner']==team)
        
        
        # count by stage and opponent
        opp_win = all_ko.groupby(['stage','opponent'])['winner'].agg(['sum','count'])
        
        opp_win = opp_win.reset_index('opponent')
        opp_win['denom'] = all_ko.groupby(['stage']).size()
        
        # calculate cond. probabilities
        opp_win['prob of encounter'] = opp_win['count']/opp_win['denom']
        opp_win['rel_perc_win'] = opp_win['sum']/opp_win['count']

        # PLOTTING
        fig, axs = plt.subplots(1,4)
        
        titles ={1: 'round of 16', 2: 'quarterfinals', 3:'semis', 4:'final'}
        
        for j in np.arange(start=1, stop=5):
            
            to_plot = opp_win.loc[j, ['opponent', 'prob of encounter']].sort_values('prob of encounter', ascending = False)
            ax = axs.ravel()[j-1]
            
            sns.heatmap(to_plot.set_index('opponent'),  cmap = plt.cm.Blues,\
                        xticklabels = [], annot = True, square = True, cbar = False, ax = ax)
            ax.set_title(titles[j])
            
        return 
#%%

# needs: df_groups, df_fixtures, tree, df_elo
def simulate_tournament(df_elo, df_groups, df_fixtures, tree, thirdseed = None, mode = 'euro', verbose = True):
    
    if mode == 'euro':
        assert thirdseed is not None, "For EURO mode, you need to provide info on how the third teams are seeded"
             
    
    #####################################################################
    #### GROUP STAGE
    #####################################################################
    groups = dict()
    all_teams = list()
    n_groups = len(df_groups.columns)
    
    for j in range(n_groups):
        
        ids = df_groups.columns[j]
        teams = list(df_groups[ids])
        all_teams += teams
        f = df_fixtures.query(f"team1 in {teams}")
        e = df_elo.query(f"index in {teams}")
        
        g = Group(ids, teams, fixtures = f, elo = e)
        
        g.simulate_games()
        g.construct_table()
        
        assert g.size == 4, "Wrong group size"
        groups[ids] = g
    
    all_teams.sort()
    
    if verbose:
        for g in groups.values():
            print(g.table)
    
    
    all_goal_realizations = list()
    for g in groups.values():
        tmp = g.games.loc[:,['goals1', 'goals2']].values
        all_goal_realizations.append(tmp)
    all_goal_realizations = pd.DataFrame(np.vstack(all_goal_realizations), columns = ['goals1', 'goals2'])
    
    #####################################################################
    #### MAP GROUPS TO SEEDS
    #####################################################################
    
    stages = list(tree['stage'].unique())
    
    # main array to bookeep the proceeding teams
    seed = pd.DataFrame(index = np.arange(tree.index.max()+1, dtype = 'int'), columns = ['team'])
    
    # helper array for seeding after group stage
    proceed_per_group = 2
    group_seed = np.arange(proceed_per_group*n_groups).reshape(proceed_per_group, n_groups, order = "F")
    group_seed = pd.DataFrame(group_seed, columns = df_groups.columns)
    
    # set seed after group stage
    for g in groups.values():
        
        for l in range(proceed_per_group):
            sd = group_seed.loc[l,g.id]
            seed.loc[sd]  = g.table.index[l]
            
    # for EURO, we need to seed also the best thirds placed teams
    if mode == 'euro':
        # get all thirds
        thirds = list()
        g_id = list()
        for g in groups.values():
            thirds.append(g.table.iloc[2,:])
            g_id.append(g.id)
        
        # construct table and sort
        thirds = pd.DataFrame(thirds).drop(columns = ['sorting']) 
        thirds['group'] = g_id
        thirds = thirds.sort_values(['points', 'goal diff', 'goals'], ascending = False)
        
        # define key in order to get correct seeding
        key3 = thirds.iloc[:4]['group']
        key3_dict = key3.to_dict(); key3_dict = {v: k for k, v in key3_dict.items()}
        
        key3 = key3.values
        key3.sort()
        key3 = ''.join(list(key3))
        
        for s3, g3 in thirdseed.loc[key3].iteritems():
            seed.loc[s3] = key3_dict[g3]
        
        assert not np.any(seed.loc[:15].isna()), "Some seeds sre still not set"
    #####################################################################
    #### KO STAGE
    #####################################################################
    
    res = tree.copy()
    res_wide = pd.DataFrame(index = all_teams, columns = stages)
    champions = pd.Series(index = all_teams)
    
    
    for s in stages:
    
        s_tree = tree[tree.stage == s]
        
        for j in s_tree.index:
            
            team1 = seed.loc[s_tree.loc[j,'parent1']].item()
            team2 = seed.loc[s_tree.loc[j,'parent2']].item()
            
            if verbose:
                print([team1,team2])
            
            res_wide.loc[[team1,team2], s] = 1
            
            # known result
            if not np.isnan(s_tree.loc[j,'score1']):
                assert s_tree.loc[j,'team1'] == team1, f"Mismatch: {s_tree.loc[j,'team1']}, {team1}"
                assert s_tree.loc[j,'team2'] == team2, f"Mismatch: {s_tree.loc[j,'team2']}, {team2}"
                
                if s_tree.loc[j,'score1'] > s_tree.loc[j,'score2']:
                    w = team1
                elif s_tree.loc[j,'score1'] < s_tree.loc[j,'score2']: 
                    w = team2
                else:
                    raise KeyError("Score of KO game seems to be a tie!")
            # simulate  
            else:
                e = df_elo.loc[[team1,team2]].values
                
                if 'location_country' in s_tree.columns:
                    l = s_tree.loc[j,'location_country']
                else:
                    l = None
                
                m = Match(id = j, teams = [team1,team2], elo = e, location = l)
                m.simulate()
                
                if verbose:
                    print(m.winner, m.p)
                
                w = m.winner
                
            # set seed according to result
            seed.loc[j, 'team'] = w
            
            # store if it was the final
            if s == stages[-1]:
                champions.loc[w] = 1
            
            # store
            res.loc[j, 'team1'] = team1
            res.loc[j, 'team2'] = team2
            res.loc[j, 'winner'] = w
    
    res_wide['champion'] = champions
    res_wide = res_wide.fillna(0)

    return res_wide, res, groups


#%%
def default_group():
    game_list = [['a','b'], ['c', 'd'], ['a', 'c'], ['b', 'd'],['a', 'd'], ['b', 'c'] ]
    df_games = pd.DataFrame(game_list, columns = ['team1', 'team2'])
    
    return df_games


class Match:
    def __init__(self, id, teams, elo, location = None):
        self.id = id
        self.teams = teams
        self.elo = elo
        self.location = location
        
        assert len(self.teams) == len(self.elo) == 2
    
    def simulate(self):
        
        # p = probability of teams[0] winning
        self.p = elo_expected_score(self.elo[1] - self.elo[0])
        
        if self.location is not None:
            if self.location == self.teams[0]:
                self.p = xg2xs(xs2xg(self.p) + home_xg)
            elif self.location == self.teams[1]:
                self.p = xg2xs(xs2xg(self.p) - home_xg)
            
        ind = np.random.binomial(n=1, p=self.p, size=1).squeeze()
        
        if ind == 1:
            self.winner = self.teams[0]
        elif ind == 0:
            self.winner = self.teams[1]
        else:
            raise KeyError("Something went wrong!")
            
        return
        
    
class Group:
    
    def __init__(self, id, teams, fixtures = None, elo = None):
        self.id = id
        self.size = len(teams)
        self.fixtures = fixtures 
        self.elo = elo
        
        keys = list(string.ascii_lowercase)[:len(teams)]
        self.teams = dict(zip(keys, teams))
        
        self.construct_fixtures()
        
        return
    
    def construct_fixtures(self):
        
        if self.fixtures is None:
            self.fixtures = default_group().replace(self.teams)
        else:
            assert len(self.fixtures) == self.size*(self.size-1)/2, f"Insufficient data for Group {self.id}"
        return
    
    def construct_table(self):
        
        
        table = self.constructor(self.games, self.teams.values())
        
        lazy = False
        if lazy:
            self.table = table.sort_values(['points','goal diff', 'goals'], ascending = False)
        else:
            ppoints, counts = np.unique(table.points, return_counts = True)        
            table['sorting'] = 0
        
            for p,c in zip(ppoints,counts):
                   
                if c == 1:
                    continue
                
                elif c == 2:
                    #print(f"Direct comparison in group {self.id}")
                    pair = list(table[table.points == p].index)
                    
                    table = self.direct_comparison(table, pair)
                        
                # 3 teams with same points --> constructs sub-table and sorts
                elif c==3:
                    sub_teams = list(table[table.points == p].index)
                    
                    ixx = self.games.team1.isin(sub_teams) & self.games.team2.isin(sub_teams)
                    sub_games = self.games.loc[ixx]
                    
                    sub_table = self.constructor(sub_games, sub_teams)
                    sub_table['sorting'] = 0
                    
                    
                    pair = list(sub_table.loc[sub_table.rank().duplicated(keep = False)].index)
                    
                    # if all sub_games have same result --> sort from all games in group
                    if len(pair) == 3:
                        sorted_from_all = table.loc[sub_teams].sort_values(['goal diff', 'goals'], ascending = False).index
                        sub_table.loc[sorted_from_all, 'sorting'] =  np.arange(3)[::-1]
                    # if there is still a pair with exact same rank --> direct comparison
                    elif len(pair) == 2:
                        sub_table = self.direct_comparison(sub_table, pair)
                    
                    sub_table = sub_table.sort_values(['points', 'goal diff', 'goals' , 'sorting'], ascending = False)
                    table.loc[sub_table.index, 'sorting'] = np.arange(3)[::-1]
        
            self.table = table.sort_values(['points', 'sorting', 'goal diff', 'goals'], ascending = False)
        
        return 
    
    def direct_comparison(self, table, pair):
        assert len(pair) == 2
        
        ixx = self.games.team1.isin(pair) & self.games.team2.isin(pair) 
                
        if self.games.loc[ixx, 'goals1'].item() > self.games.loc[ixx, 'goals2'].item():
            winner = self.games.loc[ixx, 'team1'].item()
            table.loc[winner, 'sorting'] += 1         
        elif self.games.loc[ixx, 'goals1'].item() < self.games.loc[ixx, 'goals2'].item():
            winner = self.games.loc[ixx, 'team2'].item()
            table.loc[winner, 'sorting'] += 1
        
        return table
        
    @staticmethod
    def constructor(games, team_list):
        cols = ['points', 'goals', 'goals against']
        table = pd.DataFrame(index = team_list, columns = cols)
        table[:] = 0
        
        for j, g in games.iterrows():
            
            #goals
            table.loc[g.team1, 'goals'] = table.loc[g.team1, 'goals'] + g.goals1
            table.loc[g.team1, 'goals against'] = table.loc[g.team1, 'goals against'] + g.goals2
            
            table.loc[g.team2, 'goals'] = table.loc[g.team2, 'goals'] + g.goals2
            table.loc[g.team2, 'goals against'] = table.loc[g.team2, 'goals against'] + g.goals1
            
            # points
            if g.goals1 > g.goals2:
                table.loc[g.team1, 'points'] = table.loc[g.team1, 'points'] + 3
            elif g.goals1 < g.goals2:
                table.loc[g.team2, 'points'] = table.loc[g.team2, 'points'] + 3
            else:
                table.loc[g.team1, 'points'] = table.loc[g.team1, 'points'] + 1
                table.loc[g.team2, 'points'] = table.loc[g.team2, 'points'] + 1
                        
        table['goal diff'] = table['goals'] - table['goals against']
        
        return table
    
    @staticmethod
    def elo_expected_score(elodiff):
        return 1/( 1+10**(elodiff/400))
    
    
    def simulate_games(self):
        
        games = self.fixtures.copy()#.drop(columns = ['score1', 'score2'])
        for f in self.fixtures.index:
            
            if not np.isnan(self.fixtures.loc[f, 'score1']):
                #assert self.fixtures.loc[f, 'score1'].dtype == int
                games.loc[f, 'goals1'] = self.fixtures.loc[f, 'score1'].item()
                games.loc[f, 'goals2'] = self.fixtures.loc[f, 'score2'].item()
            
            else:
                team1 = self.fixtures.loc[f,'team1']
                team2 = self.fixtures.loc[f,'team2']
                
                elo1 = self.elo.loc[team1]
                elo2 = self.elo.loc[team2]
                
                # transform elo to xg diff in order to simulate
                # xscore = exp. score of home team, xg positive --> home team favoured
                xscore = self.elo_expected_score(elo2-elo1)
                xg = xs2xg(xscore).squeeze()
                
                if 'location_country' in games.columns:
                    if games.loc[f, 'location_country'] == team1:
                        xg += home_xg
                    elif games.loc[f, 'location_country'] == team2:
                        xg -= home_xg
               
                
                # b is mu1+mu2 = total expected goals
                b = max(2.2, 1.5*abs(xg))
                
                mu1 = (xg+b)/2
                mu2 = (b-xg)/2
                
                #res = skellam.rvs(mu1, mu2, loc=0, size=1, random_state=None).squeeze()
                
                games.loc[f, 'goals1'] = np.random.poisson(lam = mu1, size = 1).squeeze()
                games.loc[f, 'goals2'] = np.random.poisson(lam = mu2, size = 1).squeeze()
        
        games['goals1'] = games['goals1'].astype('int')
        games['goals2'] = games['goals2'].astype('int')
        
        self.games = games
        
        return
    
    
#%% test Group object

# teams = ['France', 'Germany', 'Portugal', 'Sweden']
# tmp_elo = df_elo[df_elo.index.isin(teams)]
# G = Group("A", teams, elo = tmp_elo)           

# G.simulate_games()
# G.games

# G.construct_table()
# G.table