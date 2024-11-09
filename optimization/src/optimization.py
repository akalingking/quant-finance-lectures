import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from autograd import grad, jacobian, numpy as aunp


def show_plot(f, x_star, domain=[0,0], samples=100):
	"""Plots objective function, derivative, contour, and minimum"""
	fig,axs = plt.subplots(3,1,figsize=(6,5),sharex="col")

	# Objective functiot
	ax = axs[0]
	x_i = np.linspace(domain[0], domain[1], samples, dtype=float)
	ax.plot(x_i, f(x_i), label=r"$f(x)$")
	ax.set_ylabel(r"$f(x)$")
	ax.axvline(x=x_star, color='r', label=r'$x^*$', ls="--")
	ax.grid(":")
	ax.legend()

	# Gradient function
	gradient = grad(f)
	h=[gradient(x) for x in x_i]
	ax = axs[1]
	ax.plot(x_i, h, label=r"$\dfrac{df(x)}{dx}$")
	ax.set_ylabel(r"$f'(x)$")
	ax.grid(":")
	ax.legend()

	# Contour plane
	ax = axs[2]
	y_i = np.linspace(domain[0], domain[1], samples, dtype=float)
	x, y = np.meshgrid(x_i, y_i)
	z = f(x)
	contour = ax.contour(x, y, z, levels=30, cmap='coolwarm')
	ax.clabel(contour, inline=1, fontsize=8)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.axvline(x=x_star, color='r', label=r'$x^*$', ls="--")
	ax.legend()
	plt.tight_layout()
	plt.show()


def opt_newton(f, domain=[-1,1], epsilon=10**-7, maxIter=50, verbose=False):
	"""Newton Method"""
	limits = domain
	fp = grad(f)
	fpp = grad(fp)
	iter, maxiter = 0, maxIter
	x = domain[0]
	while iter <= maxiter:
		x_next = x - fp(x)/fpp(x)
		error = abs(x_next - x)
		x = x_next
		iter += 1
		if verbose:
			print(f"[{iter}] x: {x:.6f} error: {error:.6f}")
		if error <= epsilon:
			break
		if x > domain[1]:
			print(f"Warning max limit {x:.6f} > {domain[1]:.6f}!")
			break
	x_star = x
	f_xstar = f(x_star)
	print(f"x*:{x_star:.6f} f(x*): {f_xstar:.6f}")
	return x_star, f_xstar

def opt_secant(f, fp=None, domain=[], epsilone=10**-7, maxiter=50, verbose=False):
	if fp is None:
		fp = grad(f)
	a=domain[0]
	b=domain[1] 
	x = a
	j=0
	x_star = aunp.inf
	while True:
		if j > maxiter:
			break
		j += 1
		if verbose:
			print(f"[{j:.6f}] a:{a:.6f} b:{b:.6f} x:{x:.6f} f'(x):{fp(x):.6f} f'(a): {fp(a):.6f} f'(b): {fp(b)}:.6f")
		x = a - ((fp(a)*(b-a))/(fp(b)-fp(a)))
		# Exit conditions
		if fp(a)*fp(x) < 0:
			b = x
		elif fp(b)*fp(x) < 0:
			a = x
		elif f(x) == 0:
			print("Found exact solution!")
			break 
	x_star = x
	print(f"x*: {x_star:.6f} f(x*): {f(x_star):.6f}")
	return x_star, f(x_star)
