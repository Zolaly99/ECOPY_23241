
import math
import numpy

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













