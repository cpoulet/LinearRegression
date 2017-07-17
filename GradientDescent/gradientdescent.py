#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class GradientDescent:
    '''
       .x = 'km'          X
       .y = 'price'       Y
    '''
    def __init__(self):
        self.data = {}
        self.stdata = []
        self.error = []
        self.x = None
        self.y = None
        self.m = 0
        self.LearningRate = 0.005
        self.theta0 = 0 #9000
        self.theta1 = 0 #-0.025

    def generator(self):
        for couple in zip(self.data[self.x], self.data[self.y]):
            yield couple

    def get_data(self):
        with open('../Dataset/data.csv', 'r') as f:
        #with open('../Dataset/test', 'r') as f:
            for l in f:
                if self.x == None and self.y == None:
                    self.x, self.y = l.strip().split(',')
                    self.data[self.x] = []
                    self.data[self.y] = []
                else:
                    x, y = map(int, l.split(','))
                    self.data[self.x].append(x)
                    self.data[self.y].append(y)
        self.m = len(self.data[self.x])
   
    def standardization(self):
        min_x = min(self.data[self.x])
        min_y = min(self.data[self.y])
        max_x = max(self.data[self.x])
        max_y = max(self.data[self.y])
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        self.stdata = [((d[0] - min_x) / diff_x, (d[1] - min_y) / diff_y) for d in self.generator()]
        print(self.stdata)

    def intercept_gradient(self):    #gradient du decalage a l'origine theta0
        return sum([(d[1] - self.estimate(d[0])) * d[0] for d in self.stdata])

    def slope_gradient(self):    #gradient de la pente theta1
        return sum([d[1] - self.estimate(d[0]) for d in self.stdata])

    def gradient_step(self):
        theta0_gradient = self.intercept_gradient() * -1
        theta1_gradient = self.slope_gradient() * -1
        print('tetha0 (b)', self.theta0, 'tetha1 (a)', self.theta1)
        print('grad (b)', theta0_gradient, 'grad (a)', theta1_gradient)
        print(self.sum_square_error())
        self.theta0 = self.theta0 - (self.LearningRate * theta0_gradient)
        self.theta1 = self.theta1 - (self.LearningRate * theta1_gradient)
    
    def train(self):
        for _ in range(25):
            self.error.append(self.sum_square_error())
            self.gradient_step()
        print('Y = ', self.theta1,'X + ', self.theta0)
    
    def stdX(self):
        return [x[0] for x in self.stdata]

    def stdY(self):
        return [x[1] for x in self.stdata]

    def show_data(self):
        X = self.stdX()
        Y = self.stdY()
        plt.subplot(211)
        plt.scatter(X, Y)
        plt.plot([min(X), max(X)], [self.estimate(min(X)), self.estimate(max(X))], linewidth=2, color='red')
        plt.title('Price of differents cars in function of their mileage')
        plt.xlabel('mileage (km)')
        plt.ylabel('price (euros)')
        plt.subplot(212)
        plt.title('Evolution of error during the gradient descent')
        plt.xlabel('steps')
        plt.ylabel('sum of squarred errors')
        plt.plot(self.error)
        plt.tight_layout()
        plt.show()

    def fct_error(self, t0, t1):
        return sum([(d[1] - (d[0] * t1 + t0)) ** 2 for d in self.generator()]) / self.m

    def show_plane(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0, 20000, 500)
        y = np.arange(-0.5, 0.5, 0.005)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.fct_error(b,a) for b,a in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap = 'Reds', edgecolors='black')
        ax.set_xlabel('θ_0 (y-intercept)', fontweight='bold')
        ax.set_ylabel('θ_1 (slope)', fontweight='bold')
        ax.set_zticks([])
        ax.xaxis.labelpad=10
        ax.yaxis.labelpad=10
        plt.title('Sum of square Errors', fontweight='bold')
        #ax.view_init(45, 45)
        fig.tight_layout()
        plt.show()

    def estimate(self, mileage):
        return self.theta0 + self.theta1 * mileage

    def sum_square_error(self):
       return sum([(d[1] - self.estimate(d[0])) ** 2 for d in self.stdata]) / 2

    def save_theta(self):
        with open('../LinearFunction/theta_values', 'w') as f:
            f.write(str(self.theta0) + ' ' + str(self.theta1) + '\n')

def main():
    GD = GradientDescent()
    GD.get_data()
    GD.show_plane()
#    GD.standardization()
#    GD.train()
#    GD.show_data()

if __name__ == "__main__":
    main()
#    try:
#	    main()
#    except Exception as e:
#        print('Error : ' + str(e))
