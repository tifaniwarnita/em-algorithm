# modified from https://github.com/ajcr/em-explanation/blob/master/em-notebook-2.ipynb

from matplotlib.mlab import bivariate_normal
from scipy import stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

def visualize_contour(mean, cov,  alpha=1):
	delta = 1.0
	dots = 0.5
	x = np.arange(mean[0] - cov[0][0] - delta, mean[0] + cov[0][0] + delta, dots)
	y = np.arange(mean[1] - cov[1][1] - delta, mean[1] + cov[1][1] + delta, dots)
	X, Y = np.meshgrid(x, y)
	Z = bivariate_normal(X, Y, cov[0][0]/2, cov[1][1]/2, mean[0], mean[1])
	plt.contour(X,Y,Z, alpha=alpha)

def calculate_weight(likelihood, total_likelihood):
    return likelihood / total_likelihood

def estimate_mean(data, weight):
    mean_x = np.sum(data[:,0] * weight) / np.sum(weight)
    mean_y = np.sum(data[:,1] * weight) / np.sum(weight)
    return [mean_x, mean_y]

def estimate_cov(data, weight, mean):
    weighted_x = (data[:,0] - mean[0]) * weight
    weighted_y = (data[:,1] - mean[1]) * weight
    weighted = np.stack((weighted_x, weighted_y), axis=0)
    data_merge = np.stack((data[:,0], data[:,1]), axis=0)
    return np.cov(data_merge, aweights = weight)

def print_mean_cov(color, mean, cov):
	print("mean %s: %s" % (color, np.array(mean)))
	print("cov  %s: \n%s" % (color, cov))

def main_3_cluster():
	np.random.seed(123)
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

	red_mean = [5, 3]
	red_cov = [[5, 0], [0, 10]]  # diagonal covariance

	blue_mean = [18, 10]
	blue_cov = [[10, 0], [0, 8]]  # diagonal covariance

	red_data = np.random.multivariate_normal(red_mean, red_cov, 500)
	red_x, red_y = red_data.T
	blue_data = np.random.multivariate_normal(blue_mean, blue_cov, 500)
	blue_x, blue_y = blue_data.T
	both_x = np.concatenate([red_x, blue_x])
	both_y = np.concatenate([red_y, blue_y])
	both_data = np.concatenate([red_data, blue_data], axis=0)

	# estimates for the mean
	red_mean_guess = [1.0, 1.0]
	blue_mean_guess = [16.0, 9.0]
	green_mean_guess = [10.0, 2.2]

	# estimates for the standard deviation
	red_cov_guess = [[3, 0], [0, 1.0]]
	blue_cov_guess = [[3, 0], [0, 2]]
	green_cov_guess = [[6, 0], [0, 3]]

	# visualize first guess
	plt.plot(red_x, red_y, 'ro', color='r', ms=1)
	plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
	plt.axis('equal')
	plt.title(r'First Guess', fontsize=15);
	visualize_contour(red_mean_guess, red_cov_guess)
	visualize_contour(blue_mean_guess, blue_cov_guess)
	visualize_contour(green_mean_guess, green_cov_guess)
	plt.show()

	plt.clf()
	N_ITER = 10 # number of iterations of EM
	alphas = np.linspace(0.2, 1, N_ITER) # transparency of curves to plot for each iteration

	plt.plot(red_x, red_y, 'ro', color='r', ms=1)
	plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
	plt.axis('equal')
	visualize_contour(red_mean_guess, red_cov_guess, 0.1)
	visualize_contour(blue_mean_guess, blue_cov_guess, 0.1) 
	visualize_contour(blue_mean_guess, blue_cov_guess, 0.1) 

	for i in range(N_ITER):
		## Expectation step
		## ----------------

		likelihood_of_red = multivariate_normal.pdf(both_data, mean=red_mean_guess, cov=red_cov_guess)
		likelihood_of_blue = multivariate_normal.pdf(both_data, mean=blue_mean_guess, cov=blue_cov_guess)
		likelihood_of_green = multivariate_normal.pdf(both_data, mean=green_mean_guess, cov=green_cov_guess)

		red_weight = calculate_weight(likelihood_of_red, likelihood_of_red+likelihood_of_blue+likelihood_of_green)
		blue_weight = calculate_weight(likelihood_of_blue, likelihood_of_red+likelihood_of_blue+likelihood_of_green)	
		green_weight = calculate_weight(likelihood_of_green, likelihood_of_red+likelihood_of_blue+likelihood_of_green)
		
		# ## Maximisation step
		# ## -----------------

		red_cov_guess = estimate_cov(both_data, red_weight, red_mean_guess)
		blue_cov_guess = estimate_cov(both_data, blue_weight, blue_mean_guess)
		green_cov_guess = estimate_cov(both_data, green_weight, green_mean_guess)

		red_mean_guess = estimate_mean(both_data, red_weight)
		blue_mean_guess = estimate_mean(both_data, blue_weight)
		green_mean_guess = estimate_mean(both_data, green_weight)

		plt.plot(red_x, red_y, 'ro', color='r', ms=1)
		plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
		plt.axis('equal')
		visualize_contour(red_mean_guess, red_cov_guess, alphas[i])
		visualize_contour(blue_mean_guess, blue_cov_guess, alphas[i])
		visualize_contour(green_mean_guess, green_cov_guess, alphas[i])

		print("")
		print("Iteration %s" % str(i+1))
		print_mean_cov("red", red_mean_guess, red_cov_guess)
		print_mean_cov("blue", blue_mean_guess, blue_cov_guess)
		print_mean_cov("green", green_mean_guess, green_cov_guess)

	plt.title(r'Iteration ' + str(N_ITER), fontsize=15);
	plt.show()

def main():
	np.random.seed(123)
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

	red_mean = [5, 3]
	red_cov = [[5, 0], [0, 10]]  # diagonal covariance

	blue_mean = [18, 10]
	blue_cov = [[10, 0], [0, 8]]  # diagonal covariance

	red_data = np.random.multivariate_normal(red_mean, red_cov, 500)
	red_x, red_y = red_data.T
	blue_data = np.random.multivariate_normal(blue_mean, blue_cov, 500)
	blue_x, blue_y = blue_data.T
	both_x = np.concatenate([red_x, blue_x])
	both_y = np.concatenate([red_y, blue_y])
	both_data = np.concatenate([red_data, blue_data], axis=0)

	correct_red_mean = [np.mean(red_x), np.mean(red_y)]
	correct_red_cov = np.cov(np.stack((red_data[:,0], red_data[:,1]), axis=0))
	correct_blue_mean = [np.mean(blue_x), np.mean(blue_y)]
	correct_blue_cov = np.cov(np.stack((blue_data[:,0], blue_data[:,1]), axis=0))
	print_mean_cov("red", correct_red_mean, correct_red_cov)
	print_mean_cov("blue", correct_blue_mean, correct_blue_cov)

	# visualize known colors
	plt.plot(red_x, red_y, 'ro', color='r', ms=1)
	plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
	plt.axis('equal')
	plt.title(r'Distribution of Red and Blue Data (Known Colors)', fontsize=15);
	# plt.show()

	# visualize unknown colors
	plt.clf()
	plt.plot(both_x, both_y, 'ro', color='purple', ms=1)
	plt.axis('equal')
	plt.title(r'Distribution of Red and Blue Data (Unknown Colors)', fontsize=15);
	# plt.show()

	# visualize correct estimation
	plt.plot(red_x, red_y, 'ro', color='r', ms=1)
	plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
	plt.axis('equal')
	plt.title(r'Correct Estimation', fontsize=15);
	visualize_contour(correct_red_mean, correct_red_cov)
	visualize_contour(correct_blue_mean, correct_blue_cov)
	# plt.show()

	# estimates for the mean
	red_mean_guess = [1.0, 1.0]
	blue_mean_guess = [16.0, 9.0]

	# estimates for the standard deviation
	red_cov_guess = [[9, 0], [0, 2.0]]
	blue_cov_guess = [[4.3, 0], [0, 7]]

	# visualize first guess
	plt.plot(red_x, red_y, 'ro', color='r', ms=1)
	plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
	plt.axis('equal')
	plt.title(r'First Guess', fontsize=15);
	visualize_contour(red_mean_guess, red_cov_guess)
	visualize_contour(blue_mean_guess, blue_cov_guess)
	# plt.show()

	plt.clf()
	N_ITER = 10 # number of iterations of EM
	alphas = np.linspace(0.2, 1, N_ITER) # transparency of curves to plot for each iteration

	plt.plot(red_x, red_y, 'ro', color='r', ms=1)
	plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
	plt.axis('equal')
	visualize_contour(red_mean_guess, red_cov_guess, 0.1)
	visualize_contour(blue_mean_guess, blue_cov_guess, 0.1) 

	for i in range(N_ITER):
		## Expectation step
		## ----------------

		likelihood_of_red = multivariate_normal.pdf(both_data, mean=red_mean_guess, cov=red_cov_guess)
		likelihood_of_blue = multivariate_normal.pdf(both_data, mean=blue_mean_guess, cov=blue_cov_guess)

		red_weight = calculate_weight(likelihood_of_red, likelihood_of_red+likelihood_of_blue)
		blue_weight = calculate_weight(likelihood_of_blue, likelihood_of_red+likelihood_of_blue)
		
		# ## Maximisation step
		# ## -----------------

		red_cov_guess = estimate_cov(both_data, red_weight, red_mean_guess)
		blue_cov_guess = estimate_cov(both_data, blue_weight, blue_mean_guess)

		red_mean_guess = estimate_mean(both_data, red_weight)
		blue_mean_guess = estimate_mean(both_data, blue_weight)

		plt.plot(red_x, red_y, 'ro', color='r', ms=1)
		plt.plot(blue_x, blue_y, 'ro', color='b', ms=1)
		plt.axis('equal')
		visualize_contour(red_mean_guess, red_cov_guess, alphas[i])
		visualize_contour(blue_mean_guess, blue_cov_guess, alphas[i])

		print("")
		print("Iteration %s" % str(i+1))
		print_mean_cov("red", red_mean_guess, red_cov_guess)
		print_mean_cov("blue", blue_mean_guess, blue_cov_guess)

	plt.title(r'Iteration ' + str(N_ITER), fontsize=15);
	plt.show()

if __name__ == "__main__":
	main_3_cluster()