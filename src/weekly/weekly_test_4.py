import pandas
import pandas as pd
from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt
import random


df = pd.read_csv("C:/Users/MSI laptop/Downloads/Euro_2012_stats_TEAM.csv")

def number_of_participants(input_df):
    return len(df["Team"].unique())

def goals(input_df):
    list_ = ["Team", "Goals"]
    return input_df.loc[:, list_]

def sorted_by_goal(input_df):
    return input_df.sort_values("Goals", ascending = False)

def avg_goal(input_df):
    return input_df["Goals"].mean()

def countries_over_five(input_df):
    return input_df.loc[input_df["Goals"]>=6,:]["Team"]


def countries_starting_with_g(input_df):
    list_ = []
    for i in range(len(input_df)):
        if df["Team"][i].startswith("G") == True:
            list_.append(df["Team"][i])

    return list_



def first_seven_columns(input_df):
    return input_df.iloc[:,0:7]


def every_column_except_last_three(input_df):
    return df.iloc[:, 1:len(input_df.columns)-2]


def generate_quartile(input_df):
    input_df2 = input_df.copy()
    input_df2["Quartile"] = ""
    for i in range(len(input_df2)):
        if 6 <= input_df2["Goals"][i] <= 12:
            input_df2["Quartile"][i] = 1
        else:
            if input_df2["Goals"][i] == 5:
                input_df2["Quartile"][i] = 2
            else:
                if 3 <= input_df2["Goals"][i] <= 4:
                    input_df2["Quartile"][i] = 3
                else:
                    input_df2["Quartile"][i] = 4

    return input_df2


def average_yellow_in_quartiles(input_df):
    dummy_list = ['Yellow Cards']
    return input_df.groupby('Quartile')[dummy_list].mean().reset_index()


def minmax_block_in_quartile(input_df):
    input_df2 = input_df.copy()
    return input_df2.groupby('Quartile').agg({'Blocks': ['min', 'max']})


def scatter_goals_shots(input_df):
    plt.scatter(input_df["Goals"], input_df["Shots on target"])
    plt.xlabel("Goals")
    plt.ylabel("Shots on target")

#---Pareto eloszlás --- (ezután jön az utolsó fv.)
class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape


    def pdf(self, x):
        self.x = x
        if x >= self.scale:
            return self.shape*self.scale**self.shape/((self.x)**(self.shape+1))
        else:
            return 0

    def cdf(self, x):
        self.x = x
        return 1- (self.scale/x)**self.shape

    def ppf(self, p):
        self.p = p
        return self.scale*(1-self.p)**(-1/self.shape)

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        if self.shape  <= 1:
            return math.inf
        else:
            return (self.shape*self.scale)/(self.shape-1)

    def variance(self):
        if self.shape <= 2:
            return math.inf
        else:
            return ((self.scale**2*self.shape)/((self.shape-1)**2*(self.shape-2)))

    def skewness(self):
        if self.shape > 3:
            return 2*(1+self.shape)/(self.shape-3)*((self.shape-2)/self.shape)**0.5
        else:
            return math.inf
    def ex_kurtosis(self):
        if self.shape > 4:
            return (6*(self.shape**3+self.shape**2-6*self.shape-2))/(self.shape*(self.shape-3)*(self.shape-4))
        else:
            return math.inf
    def mvsk(self):

        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]










def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(pareto_distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)
    return trajectories

trajectories = gen_pareto_mean_trajectories(ParetoDistribution(rand = random, shape = 1, scale = 1),10,200)








