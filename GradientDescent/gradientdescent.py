#!/usr/bin/env python3

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class GradientDescent:
    '''
       .x = 'km'          X
       .y = 'price'       Y
    '''
    def __init__(self, verbose = False, graph = False):
        self.data = {}
        self.stdata = []
        self.error = []
        self.x = None
        self.y = None
        self.m = 0
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        self.LearningRate = 0.1
        self.theta0 = random.uniform(-100, 100)
        self.theta1 = random.uniform(-100, 100)
        self.v = verbose
        self.g = graph
        if verbose:
            print('[info] verbose mode.')
            if graph:
                print('[info] graphical output mode.')

    def get_data(self):
        print('''Enter the path of the datas (\033[1;37mReturn\033[0;m to use the standard path '/Users/raccoon/Documents/LinearRegression/Dataset/data.csv') :''')
        path = input().strip()
        if not path:
            path = '/Users/raccoon/Documents/LinearRegression/Dataset/data.csv'
        with open(path, 'r') as f:
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
        if self.v:
            print('[info] The data has been loaded.')
        if self.g:
            self._show_plane()
   
    def _generator(self):
        for couple in zip(self.data[self.x], self.data[self.y]):
            yield couple

    def scaling(self):
        self.min_x = min(self.data[self.x])
        self.min_y = min(self.data[self.y])
        self.max_x = max(self.data[self.x])
        self.max_y = max(self.data[self.y])
        diff_x = self.max_x - self.min_x
        diff_y = self.max_y - self.min_y
        self.stdata = [((d[0] - self.min_x) / diff_x, (d[1] - self.min_y) / diff_y) for d in self._generator()]

    def _gradient_step(self):
        theta0_gradient = self._t0_gradient()
        theta1_gradient = self._t1_gradient()
        self.theta0 = self.theta0 - (self.LearningRate * theta0_gradient)
        self.theta1 = self.theta1 - (self.LearningRate * theta1_gradient)
   
    def rescale(self):
        diff_x = self.max_x - self.min_x
        diff_y = self.max_y - self.min_y
        self.theta1 = self.theta1 * diff_y / diff_x
        self.theta0 = self.theta0 * diff_y + self.min_y - self.theta1*self.min_x
        if self.v:
            print('[info] theta0 = {:.1f} and theta1 = {:.3f}'.format(self.theta0, self.theta1))
        if self.g:
            self._show_rslt()

    def train(self):
        rookie = True
        while rookie:
            err = self._fct_std_error()
            l = len(self.error)
            if l:
                if err == self.error[-1] or l == 10000:
                    rookie = False
            self.error.append(err)
            self._gradient_step()
        if self.v:
            if l != 10000:
                print('[info] \033[1;32mThe model just converged after', l, 'step.\033[0;m')
            else:
                print('[info] \033[1;31mThe model did not converged after', l, 'step.\033[0;m')
    
    def _show_rslt(self):
        X = self.data[self.x]
        Y = self.data[self.y]
        plt.subplot(211)
        plt.scatter(X, Y)
        plt.plot([self.min_x, self.max_x], [self.estimate(self.min_x), self.estimate(self.max_x)], linewidth=2, color='red')
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

    def _fct_error(self, t0, t1):
        return sum([(d[1] - (d[0] * t1 + t0)) ** 2 for d in self._generator()]) / self.m

    def _t0_gradient(self, t0 = None, t1 = None):
        if t0 == None or t1 == None:
            t0 = self.theta0
            t1 = self.theta1
        return sum([(d[0] * t1 + t0) - d[1] for d in self.stdata]) / self.m

    def _t1_gradient(self, t0 = None, t1 = None):
        if t0 == None or t1 == None:
            t0 = self.theta0
            t1 = self.theta1
        return sum([d[0] * ((d[0] * t1 + t0) - d[1]) for d in self.stdata]) / self.m

    def _show_plane(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(-20000, 50000, 2500)
        y = np.arange(-0.5, 0.5, 0.005)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self._fct_error(b,a) for b,a in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap = 'Reds', edgecolors='black')
        ax.set_xlabel('θ_0 (y-intercept)', fontweight='bold')
        ax.set_ylabel('θ_1 (slope)', fontweight='bold')
        ax.set_zticks([])
        ax.xaxis.labelpad=10
        ax.yaxis.labelpad=10
        plt.title('Sum of square Errors', fontweight='bold')
        fig.tight_layout()
        plt.show()

    def _fct_std_error(self, t0 = None, t1 = None):
        if t0 == None or t1 == None:
            t0 = self.theta0
            t1 = self.theta1
        return sum([(d[1] - (d[0] * t1 + t0)) ** 2 for d in self.stdata]) / self.m

    def estimate(self, mileage):
        return self.theta0 + self.theta1 * mileage

    def save_theta(self):
        print('''Enter the path to save thetas values (\033[1;37mReturn\033[0;m to use the standard path '/Users/raccoon/Documents/LinearRegression/LinearFunction/theta_values') :''')
        path = input().strip()
        if not path:
            path = '/Users/raccoon/Documents/LinearRegression/LinearFunction/theta_values'
        with open(path, 'w') as f:
            f.write(str(self.theta0) + ' ' + str(self.theta1) + '\n')

def main(argv):
    usage = '''usage: ./gradientdescent [-vg]
        -v: verbosity
        -g: graphical output'''
    v = False
    g = False
    if len(argv) > 2:
        print (usage)
        raise Exception()
    if len(argv) == 2:
        if argv[1] in ['-vg','-gv','-v', '-g']:
            v = True if 'v' in argv[1] else False
            g = True if 'g' in argv[1] else False
        else:
            print (usage)
            raise Exception()
    GD = GradientDescent(v, g)
    GD.get_data()
    GD.scaling()
    GD.train()
    GD.rescale()
    GD.save_theta()

if __name__ == "__main__":
    try:
	    main(sys.argv)
    except Exception as e:
        if str(e):
            print('Error : ' + str(e))
