import scipy

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
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.loc = scale
        self.scale = shape

    def pdf(self, x):
     self.x = x
     power = -(self.x - self.loc)/self.scale
     numerator = math.e**power
     denominator = self.scale*(1+math.e**power)**2

     return numerator/denominator

    def cdf(self, x):
        self.x = x
        power = -(self.x - self.loc)/self.scale
        denominator = 1+math.e**power

        return 1/denominator

    def ppf(self, p):
        self.p = p
        return self.loc + self.scale*math.log(self.p/(1-self.p))

    def gen_rand(self):
        return self.ppf(self.rand.random())


    def mean(self):
        return self.loc

    def variance(self):
        return self.scale**2*math.pi**2/3

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 6/5

    def mvsk(self):
        return [self.mean(), self.median(), self.skewness(), self.ex_kurtosis()]


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
        denominator = 2**(self.dof/2)*scipy.stats.gamma(self.dof/2)
        numerator = self.x**(self.dof/2-1)*math.e**(-self.x/2)

        return numerator / denominator

    def cdf(self, x):
        self.x = x

        return scipy.special.gammainc(self.dof/2, self.x/2)

    def ppf(self, p):
        self.p = p

        return scipy.special.gammainc(self.dof/2, self.p)


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
        return [self.mean(), self.median(), self.skewness(), self.ex_kurtosis()]
