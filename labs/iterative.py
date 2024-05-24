import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

class IterativeSolver:

	def __init__(self, A, b, eps: int = 1e-10, max_iterations: int = 10000):
		self.A = A
		self.b = b
		self.eps = eps
		self.max_iterations = max_iterations

	def solve_with_gauss_seidel(self):
		N = len(self.A)
		x = np.zeros_like(self.b)
		x_old = np.zeros_like(self.b)

		tol = self.eps + 1

		iteration = 0

		while (tol > self.eps and iteration < self.max_iterations):
			iteration += 1

			for i in range(N):
				x[i] = (1 / A[i, i]) * (b[i] - np.dot(A[i, 0:i], x[0:i]) - np.dot(A[i, i + 1:N], x_old[i + 1:N]))

			res = b - np.dot(A, x)
			tol = np.linalg.norm(res, 2)

			x_old = x.copy()

		print(iteration, tol, x)

		return x
	
	def solve_with_gradient(self, P):
		N = len(self.A)
		x = np.zeros_like(self.b)

		tol = self.eps + 1
		
		iteration = 0

		r = self.b - np.dot(self.A, x)

		while (tol > self.eps and iteration < self.max_iterations):
			iteration += 1

			z = np.linalg.solve(P, r)
			alpha = np.dot(z.T, r) / np.dot(z.T, np.dot(self.A, z))
			x = x + alpha * z
			r = r - alpha * np.dot(self.A, z)
			tol = np.linalg.norm(r, 2)

		print(iteration, tol, x)

		return x

	def solve_with_conjugate_gradient(self, P):
		N = len(self.A)
		x = np.zeros_like(self.b)
		tol = self.eps + 1
		iteration = 0
		r = self.b - np.dot(self.A, x)
		p_old = np.zeros_like(self.b)
		rho_old = 1.
		while (tol > self.eps and iteration < self.max_iterations):
			iteration += 1
			z = np.linalg.solve(P, r)
			rho = np.dot(r, z)
			if (iteration > 1):
				beta = rho / rho_old
				p = z + beta * p_old
			else:
				p = z
			q = np.dot(self.A, p)
			alpha = rho / np.dot(p, q)
			x += p * alpha
			r -= q * alpha

			p_old = p
			rho_old = rho

			tol = np.linalg.norm(r, 2)

		print(iteration, tol)
		return x

	def solve_with_jacobi(self):
		N = len(self.A)
		x = np.zeros_like(self.b) # Unknown vector of zeros (initial guess)
		x_old = np.zeros_like(self.b) # Unkown vector of zeros (initial guess) old

		tol = self.eps + 1

		iteration = 0
		while (tol > self.eps and iteration < self.max_iterations):
			iteration += 1

			for i in range(N):
				x[i] = (1 / A[i, i]) * (b[i] - np.dot(A[i, 0:i], x_old[0:i]) - np.dot(A[i, i + 1:N], x_old[i + 1:N]))
			
			res = b - np.dot(A, x)
			tol = np.linalg.norm(res, 2)

			x_old = x.copy()

		print(iteration, tol, x)

		return x
	
if __name__ == "__main__":
	A = np.array([[5, -1, 2], [3, 8, -2], [1, 1, 4]])
	b = np.array([12, -25, 6])
	
	iterativeSolver = IterativeSolver(A, b)

	jacobi_solution = iterativeSolver.solve_with_jacobi()
	gauss_seidel_solution = iterativeSolver.solve_with_gauss_seidel()
	gradient_solution = iterativeSolver.solve_with_gradient(np.identity(len(A)))
	gradient_solution2 = iterativeSolver.solve_with_gradient(np.diag(np.diag(A)))
	gradient_solution3 = iterativeSolver.solve_with_gradient(A)
	gradient_solution = iterativeSolver.solve_with_conjugate_gradient(np.identity(len(A)))
	gradient_solution2 = iterativeSolver.solve_with_conjugate_gradient(np.diag(np.diag(A)))
	gradient_solution3 = iterativeSolver.solve_with_conjugate_gradient(A)