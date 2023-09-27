class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.asymmetry = asymmetry

    def sign(input_value: float) -> float:
        return copysign(1, input_value)

    def pdf(self, x):
        self.x = x
        parentheses = self.scale/(self.asymmetry+1/self.asymmetry)
        power = -(x-self.loc)*self.scale*self.sign(asymmetry)

        return parentheses * math.e**power







