class UniformDistribution:
    def __init__(self, a, b, rand=random.random()):
        self.rand = rand
        self.a = a
        self.b = b