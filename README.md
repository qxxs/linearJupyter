# Snake Count Linear Regression Analysis

This guide will help you perform linear regression analysis on the snake count data using Python and Jupyter Notebook.

## Prerequisites

First, make sure you have Python installed. Then install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Step-by-Step Instructions

1. **Create a new Jupyter Notebook**
   - Open your terminal
   - Navigate to your project directory
   - Run: `jupyter notebook`
   - Click "New" → "Python 3" to create a new notebook

2. **Copy and paste the following code into separate cells in your notebook:**

### Cell 1: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
```

### Cell 2: Load and View Data
```python
# Read the CSV file
df = pd.read_csv('snakes_count_100.csv')
df.head()
```

### Cell 3: Prepare Data and Create Model
```python
# Prepare the data
X = df['Game Number'].values.reshape(-1, 1)  # Independent variable
y = df['Game Length'].values  # Dependent variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f'R-squared score: {r2:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'Slope: {model.coef_[0]:.4f}')
print(f'Intercept: {model.intercept_:.4f}')
```

### Cell 4: Create Visualization
```python
# Create a scatter plot with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Game Number')
plt.ylabel('Game Length')
plt.title('Linear Regression: Game Length vs Game Number')
plt.legend()
plt.grid(True)
plt.show()
```

### Cell 5: Make Predictions
```python
# Make predictions for new game numbers
new_games = np.array([101, 102, 103]).reshape(-1, 1)
predictions = model.predict(new_games)

print("\nPredictions for future games:")
for game, pred in zip(new_games, predictions):
    print(f"Game {game[0]}: Predicted Length = {pred:.2f}")
```

## Understanding the Results

1. **R-squared score**: Shows how well the model fits the data (0 to 1, higher is better)
2. **Mean Squared Error**: Average squared difference between predictions and actual values
3. **Slope**: How much the game length changes for each unit increase in game number
4. **Intercept**: The predicted game length when game number is 0

## Running the Analysis

1. Make sure your `snakes_count_100.csv` file is in the same directory as your notebook
2. Run each cell in sequence (Shift + Enter)
3. The final cell will show predictions for games 101, 102, and 103

## Troubleshooting

If you encounter any errors:
- Make sure all required packages are installed
- Verify that the CSV file is in the correct location
- Check that the column names in your CSV match the code ('Game Number' and 'Game Length') 