import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class NonLinearEquationSolver:
	
	def __init__(self, f, f_prime, eps: int = 1e-10, max_iterations: int = 1000):
		self.f = f
		self.f_prime = f_prime
		self.eps = eps
		self.max_iterations = max_iterations

	def bisection(self, a: float, b: float):

		assert f(a) * f(b) < 0, f'The interval [{a}, {b}] does not contain a root'

		x = np.mean([a, b])
		err = 1
		errors = [err]

		i = 0
		while (err > self.eps and i < self.max_iterations):
			if (f(a) * f(x) > 0):
				a = x
			else:
				b = x

			x_new = np.mean([a, b])

			err = abs(x - x_new)
			errors.append(err)

			x = x_new

			i += 1
		
		print(f"Iteration: {i}, f(x) = 0: {x}")

	def intersection(self):
		pass

	def newtown(self):
		pass

	
if __name__ == "__main__":
	x = sym.symbols('x')

	f_sym = x / 8 * (63*x**4 - 70*x**2 + 15) # Legendre Polynomial of Order 5
	f_prime_sym = sym.diff(f_sym, x)

	f = sym.lambdify(x, f_sym, 'numpy')
	f_prime = sym.lambdify(x, f_prime_sym, 'numpy')

	a = 0.75
	b = 1

	nles = NonLinearEquationSolver(f = f, f_prime = f_prime)

	nles.bisection(0.75, 1)