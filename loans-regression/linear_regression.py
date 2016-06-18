from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

from modules.clean_loans import GetCleanLoanData

# Get dataframe with cleaned columns
columns = ['Debt.To.Income.Ratio', 'Monthly.Income', 'FICO.Range', 'Interest.Rate']
loans = GetCleanLoanData(columns).run()

# Run a linear regression model
X = loans[['Debt.To.Income.Ratio', 'Monthly.Income', 'FICO.Range']]
y = loans['Interest.Rate']
model = LinearRegression()
model.fit(X, y)

# Evaluate model
predicted = model.predict(X)
print("Mean squared error: {}".format(mse(y, predicted)))
print("Mean absolute error: {}".format(mae(y, predicted)))
print("R2 score: {}".format(r2_score(y, predicted)))
