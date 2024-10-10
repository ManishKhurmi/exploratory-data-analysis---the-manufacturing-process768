import numpy as np
import matplotlib.pyplot as plt

# Define the logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Define the first derivative of the logistic function
def logistic_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

# Define the second derivative of the logistic function
def logistic_second_derivative(x):
    return -np.exp(-x) * (1 - np.exp(-x)) / (1 + np.exp(-x))**3

# Generate x values
x_values = np.linspace(-6, 6, 400)

# Compute y values for logistic function, first and second derivatives
y_logistic = logistic(x_values)
y_derivative = logistic_derivative(x_values)
y_second_derivative = logistic_second_derivative(x_values)

# Plot all three curves on the same plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_logistic, label='Logistic Function', color='blue', linewidth=2)
plt.plot(x_values, y_derivative, label='First Derivative', color='green', linestyle='--', linewidth=2)
plt.plot(x_values, y_second_derivative, label='Second Derivative', color='red', linestyle=':', linewidth=2)

# Add labels and legend
plt.title('Logistic Function, First and Second Derivatives')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')

# Display grid and plot
plt.grid(True)
plt.tight_layout()
plt.show()
