import matplotlib.pyplot as plt
from sympy import *

def ExactSolution(test):

	if test == 0:

		f = symbols('f', cls=Function)
		x = symbols('x')
		eq = Eq(f(x).diff(x,2)+f(x), 0)
		eq_sol = dsolve(eq,f(x))

		fig = plt.figure()
