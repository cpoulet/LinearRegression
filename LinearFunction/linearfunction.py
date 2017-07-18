#!/usr/bin/env python3

import sys

class LinearFunction:
    def __init__(self, verbose = False):
        self.theta0 = 0
        self.theta1 = 0
        self.v = verbose
        self._get_theta()

    def _get_theta(self):
        print('''Enter the path to load the thetas values (\033[1;37mReturn\033[0;m to use the standard path '/Users/raccoon/Documents/LinearRegression/LinearFunction/theta_values') :''')
        path = input().strip()
        if not path:
            path = '/Users/raccoon/Documents/LinearRegression/LinearFunction/theta_values'
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

    def answer(self, mileage = False):
        if mileage == False:
            mileage = self.mileage
        price = self.theta0 + self.theta1 * mileage
        print('This car worth {:.1f} euros.'.format(price))

def main(argv):
    if len(argv) > 2:
        print('usage: ./linearfunction [-v]')
        raise Exception()
    LF = LinearFunction(True if '-v' in argv else False)
    LF.ask()
    LF.answer()

if __name__ == "__main__":
    try:
	    main(sys.argv)
    except Exception as e:
        if str(e):
            print('Error : ' + str(e))
