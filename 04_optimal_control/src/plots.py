import autograd.numpy as np
import matplotlib.pyplot as plt


def show_plot_3d(f, path=None, domain=[-6,6], samples=100):
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
	if path is not None:
		plt.plot(path[:,0], path[:,1], "go-", ms=5.5)
		# Initial values
		plt.plot(path[0][0], path[0][1], c="b", marker="o",\
				 label="x0=[{x0:.1f},{x1:.1f}]".format(x0=path[0][0],x1=path[0][1]))
		plt.plot(path[-1][0], path[-1][1], c="r", marker="x",\
				 label="x*=[{x0:.1f},{x1:.1f}]".format(x0=path[-1][0],x1=path[-1][1]))
		plt.legend(fontsize="small")
	plt.show()
	
def show_plot_contour(f, path=None, domain=[-6,6], samples=100):
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
	if path is not None:
		# Gradient descent path
		plt.plot(path[:,0], path[:,1], "go-", ms=5.5)
		# Initial values
		x_a = path[0]
		plt.plot(x_a[0], x_a[1], c="b", marker="o", label="x0=[{x0:.1f},{x1:.1f}]".format(x0=x_a[0],x1=x_a[1]))
		x_star = path[-1]
		plt.plot(x_star[0], x_star[1], c="r", marker="x", label="x*=[{x0:.1f},{x1:.1f}]".format(x0=x_star[0],x1=x_star[1]))
		plt.legend(fontsize="small")
	plt.show()
