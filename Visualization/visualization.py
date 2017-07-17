#!/usr/bin/env python3

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualization:
    '''
        label_1 = 'km'
        label_2 = 'price'
    '''
    def __init__(self):
        self.data = []
        self.label_1 = None
        self.label_2 = None
        self.theta0 = 0
        self.theta1 = 0

    def get_data(self):
        with open('../Dataset/data.csv', 'r') as f:
            for l in f:
                if self.label_1 == None and self.label_2 == None:
                    self.label_1, self.label_2 = l.strip().split(',')
                else:
                    d = {}
                    d[self.label_1], d[self.label_2] = map(int, l.split(','))
                    self.data.append(d)

    def get_theta(self):
        with open('../LinearFunction/theta_values', 'r') as f:
            for l in f:
                self.theta0, self.theta1 = map(float, l.split())

    def estimate(self, mileage):
        return self.theta0 + self.theta1 * mileage

    def show_data(self):
        x = [x[self.label_1] for x in self.data]
        y = [y[self.label_2] for y in self.data]
        self.theta0 = 9000 #
        self.theta1 = -0.025 #
        x1 = min(x)
        x2 = max(x)
        y1 = self.estimate(x1)
        y2 = self.estimate(x2)
        plt.scatter(x, y)
        plt.plot([x1, x2], [y1, y2], linewidth=2, color='red')
        plt.title('Price of differents cars in function of their mileage')
        plt.xlabel('mileage (km)')
        plt.ylabel('price (euros)')
        plt.show()

def main():
    VZ = Visualization()
    VZ.get_data()
    VZ.get_theta()
    VZ.show_data()

if __name__ == "__main__":
    try:
	    main()
    except Exception as e:
        print('Error : ' + str(e))
