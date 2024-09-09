# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib import pyplot
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# To see all columns at once
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

#####################################################################################################################
###################################### Load and Examine the Dataset #################################################
#####################################################################################################################

df = pd.read_csv('Advertising.csv')

'''
This data expresses sales according to the type of advertisement and the size of the cost. It  contains 
200 rows of 3 features [ TV , Radio , Newspaper] and target variable [Sales].
'''

# Examine the dataset
def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)
'''
All of the variables are float64-type except ID column but it is going to be dropped.
There is no missing data.
Shape of the dataset is (200, 5).
No unusual rows in first and last 5.
'''
print(df.columns)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head(5)

#####################################################################################################################

# Plot histograms for each feature
df.hist(bins=20, figsize=(15, 10))
plt.show()

# Newspaper has right-skewed distribution so apply log transformation.
df['Newspaper'] = np.log1p(df['Newspaper'])

#####################################################################################################################

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Correlation of all the variables with just the target
# Compute the correlation matrix
corr_matrix = df.corr()

# Extract correlations with the target variable 'Sales'
target_corr = corr_matrix['Sales'].sort_values(ascending=False)

# Display the correlations with the target variable
print(target_corr)

'''
Sales is correlated with outher variables in this order: TV(0.782) > Radio(0.576) > Newspaper(0.165)
'''

#####################################################################################################################

# No skewness observed for TV and Radio but check it to be safe.
# Step 1: Check the Distribution
# Plot histograms for TV and Radio
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['TV'], kde=True, color='blue')
plt.title('Distribution of TV')

plt.subplot(1, 2, 2)
sns.histplot(df['Radio'], kde=True, color='green')
plt.title('Distribution of Radio')

plt.show()

# Calculate skewness for TV and Radio
tv_skewness = skew(df['TV'])
radio_skewness = skew(df['Radio'])

print(f"Skewness of TV: {tv_skewness:.3f}")
print(f"Skewness of Radio: {radio_skewness:.3f}")

# Step 2: Assess the Relationship with the Target Variable (Sales)
# Scatter plots of TV and Radio against Sales
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df['TV'], y=df['Sales'], color='blue')
plt.title('TV vs Sales')

plt.subplot(1, 2, 2)
sns.scatterplot(x=df['Radio'], y=df['Sales'], color='green')
plt.title('Radio vs Sales')

plt.show()

# Check correlation between TV, Radio, and Sales
correlation_matrix = df[['TV', 'Radio', 'Sales']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

'''
It seems they are not skewed.
'''

#####################################################################################################################

# Distribution of the target variable (Sales)
sns.histplot(df['Sales'], bins=20, kde=True)
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.title('Distribution of Sales')
plt.show()

#####################################################################################################################
###################################### Outlier Handling #############################################################
#####################################################################################################################

# Box-and-whiskers plots of the variables to handle with outliers
for column in df.columns:
    plt.figure(figsize=(10, 5))
    df.boxplot(column=column, vert=False)
    plt.title(f'Box-and-Whisker Plot for {column}')
    plt.show()

'''
There are two observation being outliers in the newspaper column.
'''

#####################################################################################################################

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for the Newspaper variable to replace outliers with upper
#threshold.
Q1 = df['Newspaper'].quantile(0.25)
Q3 = df['Newspaper'].quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Determine the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find the outliers
outliers = df[(df['Newspaper'] < lower_bound) | (df['Newspaper'] > upper_bound)]

# Report the outliers
print(f"Q1 (25th percentile): {Q1}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR (Interquartile Range): {IQR}")
print(f"Lower bound for outliers: {lower_bound}")
print(f"Upper bound for outliers: {upper_bound}")
print(f"Number of outliers: {len(outliers)}")
print("Outliers:")
print(outliers[['Newspaper']])

# Replace outliers with the upper threshold
df['Newspaper'] = df['Newspaper'].apply(lambda x: upper_bound if x > upper_bound else x)

# Report the modified values
print(f"Upper bound for outliers: {upper_bound}")
print("Modified Newspaper values:")
print(df['Newspaper'])

#####################################################################################################################
###################################### Feature Engineering ##########################################################
#####################################################################################################################

df['TV_sqroot'] = df['TV']**1/2   # to make the model more tolerable to outliers
df['Radio'] = df['Radio']**2   # to make the model more tolerable to outliers

#####################################################################################################################
###################################### Training the Model ###########################################################
#####################################################################################################################

# Split the data into training and testing sets
X = df.drop('Sales', axis=1)  # Features
y = df['Sales']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#####################################################################################################################
###################################### Scaling ######################################################################
#####################################################################################################################

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#####################################################################################################################
###################################### Making Predictions ###########################################################
#####################################################################################################################

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = model.predict(X_test_scaled)

#####################################################################################################################
###################################### Model Evaluation #############################################################
#####################################################################################################################

# Calculate Mean Squared Error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")   # 3.13
print(f"R^2 Score: {r2:.2f}")   # 0.90

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# Residuals
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red', linewidth=2)
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted  Sales')
plt.show()

#####################################################################################################################

# Feature importance
# Get the feature names from the DataFrame
feature_names = X.columns

# Get the coefficients from the trained model
coefficients = model.coef_

# Create a DataFrame for feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort the features by the absolute value of coefficients in descending order
feature_importance = feature_importance.reindex(
    feature_importance.Coefficient.abs().sort_values(ascending=False).index
)

# Generate a color map
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))

# Plot the feature importances
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance from Linear Regression Model')

# Optional: Add value labels on the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}',
             va='center', ha='left', color='black')

plt.show()