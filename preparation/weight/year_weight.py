from datetime import datetime
from math import exp
import numpy as np
import numpy


class YearWeight:
    def __init__(self):
        self.current_year = int(datetime.now().year)
        self.decaying_func = 0.05

    def calculate_weight(self, vec, publication_year):
        if type(vec) is not numpy.ndarray:
            vec = np.asarray(vec)
        if publication_year is None or publication_year == '':
            publication_year = 2018
        year_diff = self.current_year - int(publication_year)
        return vec*(1/exp(self.decaying_func*year_diff))


def main():
    vec = np.random.rand(10)
    print(type(vec))
    y_w = YearWeight()
    for i in range(2000, 2019):
        val = y_w.calculate_weight(i, vec)
        print(val)


if __name__ == '__main__':
    main()
