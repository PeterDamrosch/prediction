import pandas as pd

class CleanLendingClubData:

	""" Returns a dataframe with cleaned Lending Club data.

	Optionally, can specify which columns should be included in the 
	dataframe by supplying a list of column names. """

	LOANS_URL = "https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv"
	DEFAULT_COLUMNS = ['Amount.Requested', 'Amount.Funded.By.Investors',
       'Loan.Length', 'Loan.Purpose', 'Debt.To.Income.Ratio', 'State',
       'Home.Ownership', 'Monthly.Income', 'FICO.Range', 'Open.CREDIT.Lines',
       'Revolving.CREDIT.Balance', 'Inquiries.in.the.Last.6.Months',
       'Interest.Rate']

	def __init__(self, columns_specified=DEFAULT_COLUMNS):
		self.columns_specified = columns_specified

	# Helper functions for cleaning data
	def _clean_percentage(self, rate):
		only_numbers = rate.rstrip('%')
		float_from_percent = float(only_numbers) / 100
		return round(float_from_percent, 4)

	def _clean_length(self, length):
		length = length.rstrip(' months')
		return int(length)

	def _clean_and_split_fico(self, score):
		score_range = score.split('-')
		first_score = score_range[0]
		return int(first_score)

	def _make_clean_dataframe(self):
		# Get initial dataframe with specified columns
		loans_df = pd.read_csv(self.LOANS_URL, usecols=self.columns_specified)

		# Dictionary for matching a column to the function that cleans its data
		match_columns_to_clean_function = {
			'Interest.Rate': self._clean_percentage,
			'Loan.Length': self._clean_length,
			'Debt.To.Income.Ratio': self._clean_percentage,
			'FICO.Range': self._clean_and_split_fico
		}

		# Check for (and clean) columns in the cleaning dictionary
		for column in self.columns_specified:
			if column in match_columns_to_clean_function:
				clean_function = match_columns_to_clean_function[column]
				loans_df[column] = loans_df[column].map(clean_function)

		return loans_df

	def run(self):
		return self._make_clean_dataframe()
