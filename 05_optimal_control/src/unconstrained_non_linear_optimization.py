"""01_non_linear_optimization.ipynb"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import grad, jacobian
from scipy.optimize import line_search

def first_order_gradient_descent(f, x_a, tol_1, tol_2, tol_3, alpha_1, alpha_2, maxiter=50, verbose=False):
	"""
	First order gradient descent method
	:params:
	f: objective function
	x_a: ansatz
	tol_1: minimum distance between x_j and x_j+1
	tol_2: minimum norm value for the gradient
	alpha_1: Stopping parameter for Armijo condition rule
	alpha_2: Stopping parameter for curvature condition rule
	:returns:
	X_k
	"""
	k = 0
	Norm = np.linalg.norm
	g = grad(f)
	x_k = x_a
	x_next = np.inf
	X0 = [x_k[0]]
	X1 = [x_k[1]]
	while k < maxiter:
		D = g(x_k)
		# Search or descent direction
		p_k = -D/Norm(D)
		# Find alpha that satisfies strong Wolfe conditions.
		alpha = line_search(
			f=f, myfprime=g, xk=x_k, pk=p_k, c1=alpha_1, c2=alpha_2
		)[0]
		k += 1
		if alpha != None:
			if verbose:
				print(f"[{k}]\talpha={alpha:.6f}\tpk={p_k}\tx1:{x_k[0]:.6f}\tx2:{x_k[1]:.6f}")
			x_next = x_k + alpha*p_k
			# Check for stop conditions
			if Norm(x_next - x_k) < tol_1\
					and Norm(g(x_next)) < tol_2\
					or abs(f(x_next)-f(x_k)) < tol_3:
				x_k = x_next
				X0 += [x_k[0]]
				X1 += [x_k[1]]
				break
			else:
				x_k = x_next
				X0 += [x_k[0]]
				X1 += [x_k[1]]

	return x_k, f(x_k), list(zip(X0, X1))

def second_order_gradient_descent(
		f, x_a, eps_1=10**-5, eps_2=10**-5, eps_3=10**-5,
		alpha_1=10**-4, alpha_2=0.25, maxIter=50, verbose=False):
	"""
	Second order gradient descent using modified Newton's method.
	params:
		f: objective function
		x_a: initial guess
		eps_1: threshold step size between x_j and x_j+1
		eps_2: threshold gradient
		eps_3: distance between f(x) and f(x_next)
		alpha_1: parameter for Armijo condition rule
		alpha_2: parameter for curvature condition rule
	returns:
		x*: minimizer
		f(x*): minimum value
		{p_k} for k=1,2,...,n: path of gradient descent
	"""
	Norm = np.linalg.norm
	x_j = x_a
	x_0 = [x_j[0]]
	x_1 = [x_j[1]]
	gf = grad(f)
	Hf = jacobian(gf)
	x_star = [0,0]
	j = 0
	while j < maxIter:
		# Gradient
		g_j = gf(x_j)
		# Inverse of the Hessian [Hf]^-1(x_j)
		h_j = np.linalg.inv(Hf(x_j))
		# Direction of steepest descent
		delta = -h_j.dot(g_j)
		# Solve for the beta
		beta = sp.optimize.line_search(f=f, myfprime=gf, xk=x_j, pk=delta, c1=alpha_1, c2=alpha_2)[0]
		if verbose:
			print(f"[x_{j}]:{x_j[0]:.3f}, {x_j[1]:.3f} beta: {beta:.3f} delta: {delta}")
		j += 1
		if beta != None:
			x_next = x_j + beta*delta
			x_0 += [x_next[0]]
			x_1 += [x_next[1]]
			if Norm(x_next - x_j) < eps_1 and Norm(gf(x_next)) < eps_2 or abs(f(x_next)-f(x_j)) < eps_3:
				x_star = x_j
				break
			else:
				x_j = x_next
		else:
			print("Warning beta is None!")
	return x_star, f(x_star), list(zip(x_0, x_1))


def marquardt(
		f, x_a, eps_1=10**-5, eps_2=10**-5, eps_3=10**-5,
		alpha_1=10**-4, alpha_2=0.25, gamma=10**3, maxIter=50, verbose=False):
	""" 
	Marquardt method.
	params:
		f: objective function
		x_a: initial guess
		eps_1: threshold step size between x_j and x_j+1
		eps_2: threshold gradient
		eps_3: distance between f(x) and f(x_next)
		alpha_1: parameter for Armijo condition rule
		alpha_2: parameter for curvature condition rule
		gamma: large constant to maintain positive definite factor 
	returns:
		x*: minimizer
		f(x*): minimum value
		{p_k} for k=1,2,...,n: path of gradient descent
	"""
	Norm = np.linalg.norm
	x_j = x_a 
	x_0 = [x_j[0]]
	x_1 = [x_j[1]]
	gf = grad(f)
	Hf = jacobian(gf)
	x_star = [0,0]
	j = 0
	I = np.eye(len(x_j))
	while j < maxIter:
		# Gradient
		g_j = gf(x_j)
		
		# Hessian matrix with Marquadt step modification term
		h_tilde = Hf(x_j) + gamma*I
		
		# Inverse of the modified Hessian [Hf]^-1(x_j)
		h_inv_j = np.linalg.inv(h_tilde)
		
		# Direction of steepest descent
		delta = -h_inv_j.dot(g_j)
		
		# Solve for the beta
		beta = sp.optimize.line_search(f=f, myfprime=gf, xk=x_j, pk=delta, c1=alpha_1, c2=alpha_2)[0]
		
		if verbose:
			print(f"[x_{j}]:{x_j[0]:.3f}, {x_j[1]:.3f} beta: {beta:.3f} delta: {delta}")
		j += 1
		if beta != None:
			x_next = x_j + beta*delta
			x_0 += [x_next[0]]
			x_1 += [x_next[1]]
			if Norm(x_next - x_j) < eps_1 and Norm(gf(x_next)) < eps_2 or abs(f(x_next)-f(x_j)) < eps_3:
				x_star = x_j 
				break
			else:
				x_j = x_next
		else:
			print("Warning beta is None!")
	return x_star, f(x_star), list(zip(x_0, x_1))


def show_plot_3d(f, path, domain=[-6,6], samples=100):
	"""
	Shows graph of optimization space and path of gradient descent
	:params:
	f:target function
	path: x^n vector of the gradient descent
	domain: coverage of independent variables for plotting
	samples: number of levels in the contourl plane
	:returns:
	(x*, f(x*), path)
	"""
	x0_i = np.linspace(domain[0],domain[1], samples)
	x1_i = np.linspace(domain[0],domain[1], samples)
	x, y = np.meshgrid(x0_i, x1_i)
	z = f(([x,y]))
	fig = plt.figure(figsize=(6,5))
	ax = plt.axes(projection="3d")
	ax.contour3D(x, y, z, samples, cmap="binary")
	#ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
	ax.view_init(45,-35)
	# Gradient descent path
	plt.plot(path[:,0], path[:,1], "go-", ms=5.5)
	# Initial values
	plt.plot(path[0][0], path[0][1], c="b", marker="o",\
			 label="x0=[{x0:.1f},{x1:.1f}]".format(x0=path[0][0],x1=path[0][1]))
	plt.plot(path[-1][0], path[-1][1], c="r", marker="x",\
			 label="x*=[{x0:.1f},{x1:.1f}]".format(x0=path[-1][0],x1=path[-1][1]))
	plt.legend()
	plt.show()
	
def show_plot_contour(f, path, domain=[-6,6], samples=100):
	"""
	Show countour graph of f
	"""
	x1 = np.linspace(domain[0], domain[1], samples)
	x2 = np.linspace(domain[0], domain[1], samples)
	z = np.zeros(([len(x1), len(x2)]))
	for i in range(0, len(x1)):
		for j in range(0, len(x2)):
			z[j, i] = f([x1[i], x2[j]])

	contour=plt.contour(x1, x2, z, samples, cmap=plt.cm.gnuplot)
	plt.clabel(contour, inline=1, fontsize=10)
	plt.xlabel("$x_1$")
	plt.ylabel("$x_2$")
	#plt.plot_surface(x0, x1, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
	# Gradient descent path
	plt.plot(path[:,0], path[:,1], "go-", ms=5.5)
	# Initial values
	x_a = path[0]
	plt.plot(x_a[0], x_a[1], c="b", marker="o", label="x0=[{x0:.1f},{x1:.1f}]".format(x0=x_a[0],x1=x_a[1]))
	x_star = path[-1]
	plt.plot(x_star[0], x_star[1], c="r", marker="x", label="x*=[{x0:.1f},{x1:.1f}]".format(x0=x_star[0],x1=x_star[1]))
	plt.show()
