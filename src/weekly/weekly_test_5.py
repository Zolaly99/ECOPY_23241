from pathlib import Path
import pandas as pd
import random



df = pd.read_csv("C:/Users/MSI laptop/Downloads/chipotle.tsv", sep='\t')

def change_price_to_float(input_df):
    df2 = input_df.copy()
    df2["item_price"] = input_df["item_price"].str.replace("$", "", regex = False).astype(float)
    return df2

def number_of_observations(input_df):
    return len(input_df)

def items_and_prices(input_df):
    list_ = ["item_name", "item_price"]
    df2 = input_df[list_]
    return df2


def sorted_by_price(input_df):
    list_ = ["item_name", "item_price"]
    df2 = input_df[list_]
    return df2.sort_values("item_price", ascending = False)


def avg_price(input_df):
    return input_df["item_price"].mean()


def unique_items_over_ten_dollars(input_df):
    list_ = ["item_name", "item_price", "choice_description"]
    df2 = input_df[list_].drop_duplicates()
    df3 = df2[df2["item_price"] > 10,:]
    return df3["item_name"]


def items_starting_with_s(input_df):
    list_ = []
    for i in range(len(input_df)):
        if input_df["item_name"][i].startswith("S") == True:
            list_.append(input_df["item_name"][i])
    series = pd.DataFrame(list_).squeeze()
    series.name="item_name"
    return series.unique()


def first_three_columns(input_df):
    return df.iloc[:, 0:3]



def every_column_except_last_two(input_df):
    return df.iloc[:, 0:len(input_df.columns)-2]



def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    input_df2 = input_df.copy()
    input_df3 = input_df2.loc[df[column_to_filter].isin(rows_to_keep),columns_to_keep]
    return input_df3



def generate_quartile(input_df):
    input_df2 = input_df.copy()
    input_df2["Quartile"] = 0
    for i in range(len(input_df2)):
        if input_df2["item_price"][i] >= 30:
            input_df2["Quartile"][i] = "premium"
        else:
            if 20 <= input_df2["item_price"][i] < 30:
                input_df2["Quartile"][i] = "high-cost"
            else:
                if 10 <= input_df2["item_price"][i] < 20:
                    input_df2["Quartile"][i] = "medium-cost"
                else:
                    input_df2["Quartile"][i] = "low-cost"

    return input_df2



def average_price_in_quartiles(input_df):
    df2 = generate_quartile(input_df)
    dummy_list = ['item_price']
    df3 = df2.groupby('Quartile')[dummy_list].mean().reset_index()["item_price"]
    return df3.squeeze()




def minmaxmean_price_in_quartile(input_df):
    input_df2 = input_df.copy()
    input_df3 = input_df2.groupby('Quartile').agg({'item_price': ['min', 'max']}).reset_index()
    return input_df3





def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)

    return trajectories


def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)

    return trajectories




def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)

    return trajectories



def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)

    return trajectories

def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)

    return trajectories






























import scipy
import math
import pyerf

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        self.x = x
        if self.a < self.x < self.b:
            return 1/(self.b-self.a)

        else:
            return 0


    def cdf(self, x):
        self.x = x
        if self.a < self.x < self.b:
            return (self.x-self.a)/(self.b-self.a)
        else:
            if self.x < self.a:
                return 0
            else:
                return 1

    def ppf(self, p):
        self.p = p
        return self.a + self.p*(self.b-self.a)

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return 0.5*(self.a+self.b)

    def variance(self):
        return 1/12*(self.b-self.a)**2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return -6/5

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]






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


"""
1., Hozzan létre egy új osztályt aminek a neve LogisticDistribution. Definiáljon benne a __init__ nevű függvényt, amely bemenetként kap egy véletlenszám generátort, és az eloszlás várható értékét (location) és szórás (scale) paramétereit, amely értékeket adattagokba ment le.
    Osztály név: LogisticDistribution
    függvény név: __init__
    bemenet: self, rand, loc, scale
    link: https://en.wikipedia.org/wiki/Logistic_distribution

2., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely megvalósítja az eloszlás sűrűség függvényét.
    függvény név: pdf
    bemenet: x
    kimeneti típus: float

3., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely megvalósítja az eloszlás kumulatív eloszlás függvényét.
    függvény név: cdf
    bemenet: x
    kimeneti típus: float

4., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely implementálja az eloszlás inverz kumulatív eloszlás függvényét
    függvény név: ppf
    bemenet: p
    kimeneti típus: float

5., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely az osztály létrehozásánál megadott paraméterek mellett, logisztikus eloszlású véletlen számokat generál minden meghívásnál
    függvény név: gen_rand
    bemenet: None
    kimeneti típus: float

6., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény várható értékét. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mean
    bemenet: None
    kimeneti típus: float

7., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény varianciáját. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: variance
    bemenet: None
    kimeneti típus: float

8., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény ferdeségét. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: skewness
    bemenet: None
    kimeneti típus: float

9., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény többlet csúcsosságát. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: ex_kurtosis
    bemenet: None
    kimeneti típus: float

10., Egészítse ki a LogisticDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény első momentumát, a 2. és 3. cetrális momentumát, és a többlet csúcsosságot.. Ha az eloszlásnak nincsenek ilyen értékei, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mvsk
    bemenet: None
    kimeneti típus: List
"""
class LogisticDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def pdf(self, x):
     self.x = x
     power = -(self.x - self.location)/self.scale
     numerator = math.e**power
     denominator = self.scale*(1+math.e**power)**2

     return numerator/denominator

    def cdf(self, x):
        self.x = x
        power = -(self.x - self.location)/self.scale
        denominator = 1+math.e**power

        return 1/denominator

    def ppf(self, p):
        self.p = p
        return self.location + self.scale*math.log(self.p/(1-self.p))

    def gen_rand(self):
        return self.ppf(self.rand.random())


    def mean(self):
        return self.location

    def variance(self):
        return self.scale**2*math.pi**2/3

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 6/5

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]


"""
11., Hozzan létre egy új osztályt aminek a neve ChiSquaredDistribution. Definiáljon benne a __init__ nevű függvényt, amelynek bemenetként kap egy véletlenszám generátort, és egy szabadságfok (dof) paramétert, amely értékeket adattagokba ment le.
    Osztály név: ChiSquaredDistribution
    függvény név: __init__
    bemenet: self, rand, dof
    link: https://en.wikipedia.org/wiki/Chi-squared_distribution
    link: https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_chi2.html

12., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely megvalósítja az eloszlás sűrűség függvényét.
    függvény név: pdf
    bemenet: x
    kimeneti típus: float

13., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely megvalósítja az eloszlás kumulatív eloszlás függvényét.
    függvény név: cdf
    bemenet: x
    kimeneti típus: float

14., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely implementálja az eloszlás inverz kumulatív eloszlás függvényét
    függvény név: ppf
    bemenet: p
    kimeneti típus: float

15., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely az osztály létrehozásánál megadott paraméterek mellett, logisztikus eloszlású véletlen számokat generál minden meghívásnál
    függvény név: gen_rand
    bemenet: None
    kimeneti típus: float

16., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény várható értékét. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mean
    bemenet: None
    kimeneti típus: float

17., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény varianciáját. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: variance
    bemenet: None
    kimeneti típus: float

18., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény ferdeségét. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: skewness
    bemenet: None
    kimeneti típus: float

19., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény többlet csúcsosságát. Ha az eloszlásnak nincsen ilyen értéke, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: ex_kurtosis
    bemenet: None
    kimeneti típus: float

20., Egészítse ki a ChiSquaredDistribution osztályt egy új függvénnyel, amely visszaadja az eloszlás függvény első momentumát, a 2. és 3. cetrális momentumát, és a többlet csúcsosságot.. Ha az eloszlásnak nincsenek ilyen értékei, akkor return helyett hívja meg a raise Exception("Moment undefined") parancsot.
    függvény név: mvsk
    bemenet: None
    kimeneti típus: List
"""

class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        self.x = x
        denominator = 2**(self.dof/2)*(scipy.special.gamma(self.dof/2))
        numerator = self.x**(self.dof/2-1)*math.e**(-self.x/2)

        return numerator/denominator

    def cdf(self, x):
        self.x = x

        return scipy.special.gammainc(self.dof/2, self.x/2)

    def ppf(self, p):
        self.p = p

        return 2*scipy.special.gammaincinv(self.dof/2, self.p)


    def gen_rand(self):
        return self.ppf(self.rand.random())


    def mean(self):
        return self.dof

    def variance(self):
        return 2*self.dof

    def skewness(self):
        return math.sqrt(8/self.dof)


    def ex_kurtosis(self):
        return 12/self.dof


    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]



class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def sign(self, input_value: float) -> float:
        return math.copysign(1, input_value)

    def pdf(self, x):
        self.x = x

        if self.sign(self.x-self.loc) == 1:

            y1 = self.x-self.loc
            power = -(y1/self.scale)
            return 1/(2*self.scale)*math.e**(power)

        if self.sign(self.x-self.loc) == -1:
                  y2 = self.loc - self.x
                  power = -(y2 / self.scale)
                  return 1 / (2 * self.scale) * math.e ** (power)

    def cdf(self, x):
        self.x = x
        if self.x <= self.loc:
            return 0.5*math.e**((self.x-self.loc)/self.scale)
        else:
            return 1-0.5*math.e**(-(self.x-self.loc)/self.scale)

    def ppf(self, p):
        self.p = p
        if self.p <= 0.5:
            return self.loc + self.scale*math.log(2*p)
        else:
            return self.loc - self.scale*math.log(2-2*p)

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        return self.loc

    def variance(self):
        return 2*self.scale**2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 3

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]







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


class CauchyDistribution:

    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        self.x = x
        parentheses = ((self.x-self.loc)/self.scale)**2
        denominator = math.pi*self.scale*(1+parentheses)
        return 1/denominator

    def cdf(self, x):
        self.x = x
        parentheses = (self.x - self.loc)/self.scale
        return 1/math.pi*math.atan(parentheses) + 0.5

    def ppf(self, p):
        self.p = p
        parentheses = math.pi*(self.p-0.5)
        return self.loc + self.scale*math.tan(parentheses)

    def gen_rand(self):
        return self.ppf(self.rand.random())

    def mean(self):
        raise Exception("Moments undefined")

    def variance(self):
        raise Exception("Moments undefined")
    def skewness(self):
        raise Exception("Moments undefined")
    def ex_kurtosis(self):
        raise Exception("Moments undefined")

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]