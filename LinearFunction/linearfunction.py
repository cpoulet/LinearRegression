#!/usr/bin/env python3

import sys

class LinearFunction:
    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.get_theta()

    def get_theta(self):
        with open('theta_values', 'r') as f:
            for l in f:
                theta0, theta1 = map(float, l.split())

    def ask(self):
        print('Please, enter a mileage :')
        self.mileage = float(input())

    def answer(self, mileage = False):
        if mileage == False:
            mileage = self.mileage
        price = self.theta0 + self.theta1 * mileage
        print('This car worth', price, 'euros.')

def main():
    LF = LinearFunction()
    LF.ask()
    LF.answer()

if __name__ == "__main__":
    try:
	    main()
    except Exception as e:
        print('Error : ' + str(e))
