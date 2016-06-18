import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Set seed for reproducible models
np.random.seed(414)

# Generate random dataset to play with
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

# Split into training and testing datasets
train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]
train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Create a simple linear fit regression
linear_fit = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Create a quadratic fit regression
quadratic_fit = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

# Compare parameters
linear_fit_params = linear_fit.params
quadratic_fit_params = quadratic_fit.params

# Compare residuals
linear_fit_residuals = linear_fit.df_resid
quadratic_fit_residuals = quadratic_fit.df_resid

