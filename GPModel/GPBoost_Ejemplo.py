import gpboost as gpb
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(1)
# Simulate Gaussian process: training and test data (the latter on a grid for visualization)
sigma2_1 = 0.35  # marginal variance of GP
rho = 0.1  # range parameter
sigma2 = 0.1  # error variance
n = 200  # number of training samples
nx = 50 # test data: number of grid points on each axis

# training locations (exclude upper right rectangle)
coords = np.column_stack((np.random.uniform(size=1)/2, np.random.uniform(size=1)/2))
while coords.shape[0] < n:
	coord_i = np.random.uniform(size=2)
	if not (coord_i[0] >= 0.6 and coord_i[1] >= 0.6):
		coords = np.vstack((coords,coord_i))

# test locations (rectangular grid)
s_1 = np.ones(nx * nx)
s_2 = np.ones(nx * nx)
for i in range(nx):
	for j in range(nx):
		s_1[j * nx + i] = (i + 1) / nx
		s_2[i * nx + j] = (i + 1) / nx
coords_test = np.column_stack((s_1, s_2))

n_all = nx**2 + n # total number of data points 
coords_all = np.vstack((coords_test,coords))
D = np.zeros((n_all, n_all))  # distance matrix
for i in range(0, n_all):
	for j in range(i + 1, n_all):
		D[i, j] = np.linalg.norm(coords_all[i, :] - coords_all[j, :])
		D[j, i] = D[i, j]
Sigma = sigma2_1 * np.exp(-D / rho) + np.diag(np.zeros(n_all) + 1e-10)
C = np.linalg.cholesky(Sigma)
b_all = C.dot(np.random.normal(size=n_all))
b_train = b_all[(nx*nx):n_all] # training data GP
# Mean function. Use two predictor variables of which only one has an effect for easy visualization
def f1d(x):
	return np.sin(3*np.pi*x) + (1 + 3 * np.maximum(np.zeros(len(x)),x-0.5)/(x-0.5)) - 3
X = np.random.rand(n, 2)
F_X_train = f1d(X[:, 0]) # mean
xi_train = np.sqrt(sigma2) * np.random.normal(size=n)  # simulate error term
y = F_X_train + b_train + xi_train  # observed data
# test data
x = np.linspace(0,1,nx**2)
x[x==0.5] = 0.5 + 1e-10
X_test = np.column_stack((x,np.zeros(nx**2)))
F_X_test = f1d(X_test[:, 0])
b_test = b_all[0:(nx**2)]
xi_test = np.sqrt(sigma2) * np.random.normal(size=(nx**2))
y_test = F_X_test + b_test + xi_test




gp_model = gpb.GPModel(gp_coords=coords, cov_function="exponential")

data_train = gpb.Dataset(X, y)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Training data')

params = { 'objective': 'regression_l2', 'learning_rate': 0.01,'max_depth': 3, 'min_data_in_leaf': 10, 'num_leaves': 2**10, 'verbose': 0 }

# Training

import time

t0 = time.time()
bst = gpb.train(params=params, train_set=data_train, gp_model=gp_model, num_boost_round=247)

# Make predictions: latent variables and response variable
pred = bst.predict(data=X_test, gp_coords_pred=coords_test, predict_var=True, pred_latent=True)
print(time.time() - t0)
# pred['fixed_effect']: predictions from the tree-ensemble.
# pred['random_effect_mean']: predicted means of the gp_model.
# pred['random_effect_cov']: predicted (co-)variances of the gp_model

pred_resp = bst.predict(data=X_test, gp_coords_pred=coords_test, predict_var=False, pred_latent=False)
y_pred = pred_resp['response_mean'] # predicted response mean # Calculate mean square error
MSE = np.mean((y_pred - y_test)**2)

plt.subplot(2, 2, 2)
plt.scatter(coords_test[:, 0], coords_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title('Predicted response')

plt.subplot(2, 2, 3)
plt.scatter(coords_test[:, 0], coords_test[:, 1], c=y_test, cmap='coolwarm')
plt.title('Real')

plt.show()

