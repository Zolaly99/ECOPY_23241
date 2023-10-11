import random
import pandas as pd
import pyerf
import math


class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        self.x = x
        sd = self.scale ** 0.5
        numerator = sd*(2*math.pi)**(0.5)
        power = -0.5*((self.x-self.loc)/sd)**2

        return (1/numerator*math.e**power)

    def cdf(self, x):
        self.x = x
        sd = self.scale**0.5
        parentheses = (self.x-self.loc)/(sd*2**0.5)
        return 0.5*(1+math.erf(parentheses))

    #def mvsk(self):
    #    return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

    def ppf(self, p):
        self.p = p
        sd = self.scale**0.5
        return self.loc + sd*2**0.5*pyerf.erfinv(2*p-1)

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return self.loc

    def variance(self):
        return self.scale

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 0

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]




'''
1. Készíts egy függvényt ami a bemeneti dictionary-ből egy DataFrame-et ad vissza.

függvény bemenete: test_dict
Egy példa a kimenetre: test_df
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: dict_to_dataframe
'''
from matplotlib import pyplot as plt

stats = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

def dict_to_dataframe(test_dict):
    test_df = pd.DataFrame.from_dict(test_dict).copy()
    return test_df

test_df = dict_to_dataframe(stats)

dict_to_dataframe(stats)

'''
2. Készíts egy függvényt ami a bemeneti DataFrame-ből vissza adja csak azt az oszlopot amelynek a neve a bemeneti string-el megegyező.

függvény bemenete: test_df, column_name
Egy példa a kimenetre: test_df
kimeneti típus: pandas.core.series.Series
függvény neve: get_column
'''

def get_column(test_df, column_name):
    return test_df.loc[:, column_name]

x = get_column(test_df, "country")

'''
3. Készíts egy függvényt ami a bemeneti DataFrame-ből kiszámolja az országok népsűrűségét és eltárolja az eredményt egy új oszlopba ('density').
(density = population / area)

függvény bemenete: test_df
Egy példa a kimenetre: test_df
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: population_density
'''

def population_density(test_df):
    test_df['density'] = test_df['population']/test_df['area']

population_density(test_df)

'''
4. Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlopdiagramot (bar plot),
ami vizualizálja az országok népességét.

Az oszlopdiagram címe legyen: 'Population of Countries'
Az x tengely címe legyen: 'Country'
Az y tengely címe legyen: 'Population (millions)'

függvény bemenete: test_df
Egy példa a kimenetre: fig
kimeneti típus: matplotlib.figure.Figure
függvény neve: plot_population
'''


def plot_population(test_df):
    plt.bar(test_df["country"], test_df["population"])
    plt.xlabel("Country")
    plt.ylabel("Population (millions)")


plot_population(test_df)

'''
5. Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja az országok területét. Minden körcikknek legyen egy címe, ami az ország neve.

Az kördiagram címe legyen: 'Area of Countries'

függvény bemenete: test_df
Egy példa a kimenetre: fig
kimeneti típus: matplotlib.figure.Figure
függvény neve: plot_area
'''


def plot_population(test_df):
    plt.pie(test_df["area"], labels=test_df["country"], autopct='%.1f%%')
    plt.title("Area of Countries")


plot_population(test_df)


'''
1., Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

függvény bemente: input_csv
Egy példa a kimenetre: df_data
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''


def csv_to_df(input_csv):
    return pd.read_csv(input_csv)


df = csv_to_df("C:/Users/MSI laptop/Downloads/StudentsPerformance.csv")


'''
2., Készíts egy függvényt, ami egy DataFrame-et vár paraméterként,
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

függvény bemente: input_df
Egy példa a kimenetre: df_data_capitalized
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''


def capitalize_columns(input_df):
    dummy_list = []
    for element in input_df.columns:
        if "e" not in element:
            dummy_list.append(element.capitalize())
        else:
            dummy_list.append(element)

    input_df.columns = dummy_list

capitalize_columns(df)

'''
3., Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

függvény bemente: input_df
Egy példa a kimenetre: 5
kimeneti típus: int
függvény neve: math_passed_count
'''


def math_passed_count(input_df):
    return len(input_df[input_df["math score"]>50])


'''
4., Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

függvény bemente: input_df
Egy példa a kimenetre: df_did_pre_course
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''


def did_pre_course(input_df):
    return input_df[input_df["test preparation course"]=="completed"]


'''
5., Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

függvény bemente: input_df
Egy példa a kimenetre: df_average_scores
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: average_scores
'''


def average_scores(input_df):
    dummy_list = ['math score', 'reading score', 'writing score']
    return input_df.groupby('parental level of education')[dummy_list].mean().reset_index()


'''
6., Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

függvény bemente: input_df
Egy példa a kimenetre: df_data_with_age
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: add_age
'''

def add_age(input_df):
    random.seed(42)
    new_input_df = input_df.copy()
    new_input_df["age"] = ""
    for i in range(len(new_input_df)):
        new_input_df["age"][i]=random.randint(18, 66)
    return new_input_df


'''
7., Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

függvény bemente: input_df
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
kimeneti típus: tuple
függvény neve: female_top_score
'''


def female_top_score(input_df):
    new_input_df = input_df.copy()
    df_ = new_input_df.loc[(new_input_df['gender'] =='female')]
    sum_scores = df_['math score'] + df_['reading score'] + df_['writing score']
    df_best_fem = df_.loc[(df_['math score']+ df_['reading score'] + df_['writing score'] == sum_scores.max())]
    return [list(df_best_fem["math score"])[0], list(df_best_fem["math score"])[0], list(df_best_fem["math score"])[0]]


'''
8., Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: 5
80-90%: 4
66-80%: 3
50-65%: 2
<50%: 1

függvény bemente: input_df
Egy példa a kimenetre: df_data_with_grade
kimeneti típus: pandas.core.frame.DataFrame
függvény neve: add_grade
'''


def add_grade(input_df):
    new_input_df = input_df.copy()
    new_input_df["average"] = (new_input_df["math score"] + new_input_df["reading score"] + new_input_df[
        "writing score"]) / 3
    new_input_df["grade"] = 0
    for i in range(len(input_df)):
        if new_input_df["average"][i] >= 90:
            new_input_df["grade"][i] = 5
        else:
            if new_input_df["average"][i] >= 80:
                new_input_df["grade"][i] = 4
            else:
                if new_input_df["average"][i] >= 66:
                    new_input_df["grade"][i] = 3
                else:
                    if new_input_df["average"][i] >= 50:
                        new_input_df["grade"][i] = 2
                    else:
                        new_input_df["grade"][i] = 1

    del new_input_df['average']

    return new_input_df

'''
9., Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot, ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

függvény bemente: input_df
Egy példa a kimenetre: fig
kimeneti típus: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''


def math_bar_plot(input_df):
    dummy_list = ['math score']
    df_ = input_df.groupby('gender')[dummy_list].mean().reset_index()
    plt.bar(df_["gender"], df_["math score"])
    plt.xlabel("Gender")
    plt.ylabel("Math Score")

    ''' 
    10., Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot, ami vizualizálja az elért írásbeli pontszámokat.

    A histogram címe legyen: 'Distribution of Writing Scores'
    Az x tengely címe legyen: 'Writing Score'
    Az y tengely címe legyen: 'Number of Students'

    függvény bemente: input_df
    Egy példa a kimenetre: fig
    kimeneti típus: matplotlib.figure.Figure
    függvény neve: writing_hist
    '''
def writing_hist(input_df):
    plt.hist(input_df["writing score"])
    plt.xlabel("Writing Score")
    plt.ylabel("Number of Students")


''' 
11., Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot, ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

függvény bemente: input_df
Egy példa a kimenetre: fig
kimeneti típus: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''


def ethnicity_pie_chart(input_df):
    df_ = input_df['race/ethnicity'].value_counts().reset_index()
    plt.pie(df_["race/ethnicity"], labels=df_["index"], autopct='%.1f%%')
    plt.title("Proportion of Students by Race/Ethnicity")



"""
1. Készíts egy függvényt, ami létrehoz egy listát, benne number_of_trajectories db listával. A belső listák létrehozásának logikája a következő:
    A bemeneti paraméterként kapott normal_distribution osztály felhasználásával (NormalDistribution 0,1 paraméterekkel) generálj length_of_trajectory véletlen számot
    A belső lista tartalmazza a generált számok kumulatív átlagát.
    Ismételd meg number_of_trajectories alkalommal (mindegyik belső listába egyszer)

függvény bemenete: normal_distribution, number_of_trajectories, length_of_trajectory
kimeneti típus: List    
függvény neve: generate_mean_trajectories
"""


def generate_mean_trajectories(normal_distribution, number_of_trajectories, length_of_trajectory):
    trajectories = []
    for i in range(number_of_trajectories):
        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(normal_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory)/(i+1))
        trajectories.append(mean_trajectory)
    return trajectories
#from src.utils import distributions as dst

trajectories = generate_mean_trajectories(NormalDistribution(rand = random, loc = 0, scale = 1),10,200)




"""
2., Az előző feladatban létrehozott listák listáját ábrázold vonal ábrával. Minden vonal feleljen meg 1 belső listának (50 vonal legyen az ábrán)

Az ábra címe: Mean trajectories

függvény bemente: input_list
függvény kimenete: fig
kimeneti típus: matplotlib.figure.Figure

"""

for y in trajectories:
      x=range(1, len(y)+1)
      plt.plot(x,y)
plt.show()