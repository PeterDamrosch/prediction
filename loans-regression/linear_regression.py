import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold

from modules.clean_loans import GetCleanLoanData

# Get dataframe with cleaned columns
columns = ['Debt.To.Income.Ratio', 'Monthly.Income', 'FICO.Range', 'Interest.Rate']
loans = GetCleanLoanData(columns).run()

# Specify dependent section and independent section of dataframe
X_original = loans[['Debt.To.Income.Ratio', 'Monthly.Income', 'FICO.Range']]
y_original = loans['Interest.Rate']

# Create an empty results dataframe for kfold regression
results_columns = ['Fold', 'Mean Squared Error', 'Mean Absolute Error', 'R2 Score']
kfold_results_df = pd.DataFrame(columns=results_columns)

# Get indices for 10 K-folds
kfold_total = KFold(len(X_original), n_folds=10, shuffle=True, random_state=4)

# Run regression for each fold and evaluate against test data
fold_count = 0
for train, test in kfold_total:
	# Set training and test datasets
	X_train, X_test = X_original.iloc[train], X_original.iloc[test]
	y_train, y_test = y_original.iloc[train], y_original.iloc[test]

	# Create and fit model
	fold_model = LinearRegression()
	fold_model.fit(X_train, y_train)

	# Evaluate model
	fold_predicted = fold_model.predict(X_test)
	fold_mse = mse(y_test, fold_predicted)
	fold_mae = mae(y_test, fold_predicted)
	fold_r2 = r2_score(y_test, fold_predicted)

	# Add results to results dataframe
	kfold_results_df.loc[fold_count] = [fold_count, fold_mse, fold_mae, fold_r2]

	# Increment the fold count and repeat
	fold_count += 1

# Do a linear regression without kfolds to compare as a baseline
regular_model = LinearRegression()
regular_model.fit(X_original, y_original)

# Evaluate regular linear regresison model
regular_model_predicted = regular_model.predict(X_original)
regular_mse = mse(y_original, regular_model_predicted)
regular_mae = mae(y_original, regular_model_predicted)
regular_r2 = r2_score(y_original, regular_model_predicted)

# Compare results from regular model to average of kfolds
kfolds_mse = kfold_results_df['Mean Squared Error'].mean()
kfolds_mae = kfold_results_df['Mean Absolute Error'].mean()
kfolds_r2 = kfold_results_df['R2 Score'].mean()

print("Regular MSE: {}, K-Folds: {}".format(regular_mse, kfolds_mse))
print("Regular MAE: {}, K-Folds: {}".format(regular_mae, kfolds_mae))
print("Regular R2: {}, K-Folds: {}".format(regular_r2, kfolds_r2))
