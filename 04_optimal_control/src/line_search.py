import scipy as sp
import autograd.numpy as np
from autograd import grad
from autograd.numpy import transpose as T
from autograd.numpy.linalg import norm as Norm
# https://github.com/gjkennedy/ae6310/blob/master/Line%20Search%20Algorithms.ipynb


def zoom(f, x, beta_l, beta_h, c1=10**-4, c2=0.9, verbose=False):
	# inner loop gradient function
	gf = grad(f)
	N,j = 10,0
	beta_star = np.inf
	delta = -gf(x)/Norm(gf(x))
	assert gf(x+beta*delta) < 0
	while j<N:
		assert beta_l < beta_h
		beta = 0.5 * (beta_l + beta_h)
		if verbose: print(f"\tzoom [{j}] beta[l]: {beta_l} beta_h:{beta_h} x: {x}")
		
		if f(x + beta*delta) > (f(x) + c1*beta*np.dot(T(gf(x)), delta))\
				or f(x + beta*delta) >= f(x + beta_l * delta):
			# Check for strong Wolfe condition with parameter c1 set to 10^-4
			if verbose: print(f"\tzoom [{j}] Sufficient decrease violation!")
			beta_h = beta # update high beta
		else:
			if abs(np.dot(T(gf(x_prev + beta*delta)),delta)) <= c2 * np.dot(T(gf(x_prev)),delta):
				# Check for curvature condition rule using c2 bet 0.1 and 0.9
				beta_star = beta # Update the upper bound
				if verbose: print(f"\tzoom [{j}] Wolfe condition satisfied.")
				break
			else:
				if verbose: print(f"\tzoom [{j}] Curvature condition violation!")
			if (beta_r - beta_l) * np.dot(T(gf(x_prev + beta_l*delta)),delta) >= 0.0:
				beta_h = beta_l 
		j += 1
	if verbose: print(f"\tzoom [{j}] beta*: {beta_star}")
	return beta_star

def line_search(f, x, beta_l=0.0, beta_h=1.0, c1=10**-4, c2=0.9, maxIter=20, epsilon=10**-6, verbose=False):
	"""
	Implementation of Strong Wolfe Conditions for gradient descent
	beta_l: beta where f(x+beta*delta) is at lowest
	beta_h: beta where f(x+beta*detlta) is at highest
	"""
	assert 0 < c1 < c2 < 1
	gf = grad(f)
	beta_prev = beta_l
	beta_max = beta_h
	N, j = maxIter, 0
	beta_star = np.inf
	delta = -gf(x)/Norm(gf(x))
	while j<N:
		beta = (beta_prev + beta_h)/2.
		# direction of strict descent criteria
		assert np.dot(T(gf(x+beta*delta)), delta) < 0
		if verbose: print(f"line_search [{j}] beta prev: {beta_prev} beta:{beta} x: {x} delta:{delta}")
		if f(x + beta*delta) > f(x) + c1 * beta* np.dot(T(gf(x)), delta)\
				or (j > 0 and f(x + beta*delta) >= f(x + beta_prev*delta)):
			# Check for Armijo-Goldstein inequality with parameter c1 set to 10^-4 for sufficient decrease
			if verbose: print(f"line_search sufficient decrease violation, run inner loop")
			beta_star = zoom(f, x, beta_prev, beta, c1, c2, verbose=verbose)
			break
		if abs(np.dot(T(gf(x + beta*delta)), delta) <= c2*np.dot(T(gf(x)), delta)):
			# Check for curvature condition rule using c2 bet 0.1 and 0.9
			if verbose: print(f"line_search Strong Wolfe condition satisfied")
			beta_star = beta
			break
		if np.dot(T(gf(x + beta_l*delta)), delta) >= 0:
			# f'(x_c;d) < 0 
			assert beta < beta_prev
			if verbose: print(f"line_search beta not optimized")
			beta_star = zoom(f, x, beta, beta_max, c1, c2, verbose=verbose)
			break
		j += 1
		beta_star = beta
		if abs(beta - beta_prev) < epsilon:
			break
		beta_prev = beta

	if beta_star >= beta_max:
		print(f"line_search [{j}] Failed!")

	assert beta_star < beta_max

	if verbose: print(f"line_search exit beta*:{beta_star}")

	return beta_star


def test():
	# Objective functions
	#f = lambda x: x[0]**2 + x[1]**2
	f = lambda x: x[0]**2 + x[0]*x[1] + x[1]**2

	# Initial input values
	x = np.array([-1., -1.], dtype=float)
	beta = line_search(f, x, beta_l=0, beta_h=1, verbose=True)

	# Setup the gradient function
	gf = grad(f)

	# Calculate the step direction
	delta = -gf(x)/Norm(gf(x))
	print(f"Custom beta:{beta:.6f} f(x):{f(x):.6f} > f(x+beta*delta): {f(x+beta*delta):.6f}")

	from scipy.optimize import line_search as line_search_
	beta_exp = line_search_(f, myfprime=gf, xk=x, pk=delta)[0]
	assert beta_exp is not None
	print(f"Scipy beta:{beta_exp:.6f} f(x): {f(x):.6f} > f(x+beta*delta): {f(x+beta_exp*delta):.6f}")

if __name__ == "__main__":
	test()
