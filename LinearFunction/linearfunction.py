#!/usr/bin/env python3

import sys
import argparse

class LinearFunction:
    def __init__(self, verbose = False):
        self.theta0 = 0
        self.theta1 = 0
        self.v = verbose
        self._PATH = '/Users/cpoulet/Documents/ft_LinearRegression/LinearFunction/theta_values'
        self._get_theta()

    def _get_theta(self):
        print('Enter the path to load the thetas values (\033[1;37mReturn\033[0;m to use the standard path ' + self._PATH + ') :')
        path = input().strip()
        if not path:
            path = self._PATH
        with open(path, 'r') as f:
            for l in f:
                self.theta0, self.theta1 = map(float, l.split())
                if self.v:
                    print('[info] theta0 = {:.2f} and theta1 = :{:.4f}'.format(self.theta0, self.theta1))

    def ask(self):
        print('Please, enter a mileage :')
        s = input()
        if s == 'q':
            sys.exit(0)
        try:
            self.mileage = float(s)
        except:
            print('Enter a valid float number or press \'q\' to exit the program.')
            self.ask()
        if self.mileage < 0:
            print('Enter a valid POSITIV float number or press \'q\' to exit the program.')
            self.ask()

    def answer(self, mileage = False):
        if mileage == False:
            mileage = self.mileage
        price = self.theta0 + self.theta1 * mileage
        if price >= 0:
            print('This car worth {:.1f} euros.'.format(price))
        else:
            print('This car worth nothing... ¯\_(ツ)_/¯')

def main():
    parser = argparse.ArgumentParser(description='Guess the price depending of the mileage using a linear function described by theta0 and theta1')
    parser.add_argument('-v', '--verbosity', action='store_true', help='verbosity expanded')
    args = parser.parse_args()
    LF = LinearFunction(args.verbosity)
    LF.ask()
    LF.answer()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if str(e):
            print('Error : ' + str(e))
