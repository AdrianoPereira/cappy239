import numpy as np
from math import ceil, log2, sqrt
from statistics import stdev, mean


class Pmodel(object):
    def __init__(self, noValues=4096, p=0.52, slope=None): #==2**12
        self.noOrders = ceil(log2(noValues))
        self.noValuesGenerated = 2**self.noOrders
        self.y = 1
        self.result = self.generate_serie(noValues, p, slope)

    def generate_serie(self, noValues, p, slope):
        self.y = 1
        for n in range(0, self.noOrders):
            self.y = self.__next_step_1d(self.y, p)   
        
        if slope:
            fourierCoeff = self.__fractal_spectrum_1d(noValues, slope/2).T
            
            meanVal = mean(self.y)
            stdy = stdev(self.y)
            
            x = np.fft.ifft(self.y - meanVal)
            phase = np.angle(x)
            print('Fourrier: ', phase)
            x = fourierCoeff * np.exp(1j*phase)

            x = np.real(np.fft.fft(x))
            x = x * stdy / stdev(x)
            x = x + meanVal
        else:
            x = self.y

        return np.round(x, decimals=5)

    def get_result(self):
        return self.result

    def __next_step_1d(self, y, p, seed=False):
        size = 1 if type(y) == int else len(y)
        y2 = np.array([0.0 for n in range(size*2)])
        
        # if seed:
        #     np.random.seed(vseed)
        
        sign = np.random.random(size)-0.5
        sign /= abs(sign)

        y2[0:2*size-1:2] = y + sign*(1-2*p)*y
        y2[1:2*size:2] = y - sign*(1-2*p)*y

        return y2

    def __fractal_spectrum_1d(self, noValues, slope):
        ori_vector_size = noValues
        ori_half_size   = ori_vector_size/2
        a = np.array([0.0 for n in range(ori_vector_size+1)])
        ori_half_size = int(ori_half_size+2)

        for t2 in range(1, ori_half_size):
            index = t2-1
            t4 = 2 + ori_vector_size - t2

            if t4 > ori_half_size:
                t4 = t2
            if index <= 0:
                coeff = 0
            else:
                coeff = np.power(index, slope)

            a[t2] = coeff
            a[t4] = coeff

        a[1] = 0

        return a[1:]
