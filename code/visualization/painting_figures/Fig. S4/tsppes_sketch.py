# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:32:05 2025

@author: Realunknown
"""

import numpy as np
import matplotlib.pyplot as plt


#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

x = np.linspace(0, 5, 50)
y_true = x**2
noise = 0.3 * x * np.random.randn(50)
y_noisy = y_true + noise
y = y_noisy


unrelated_data_x = [0.5]*10
unrelated_data_y = [15]*10
unrelated_data_x = [ele + 0.2 * np.random.randn() for ele in unrelated_data_x] 
unrelated_data_y = [ele + 2 * np.random.randn() for ele in unrelated_data_y] 

x_new = np.append(x, unrelated_data_x)
y_new = np.append(y, unrelated_data_y)


# Use numpy's polyfit for polynomial fitting, e.g., fitting a quadratic polynomial
coefficients = np.polyfit(x_new, y_new, 2)  # 2 means a quadratic polynomial
 
# Create a polynomial function from the fitted coefficients
p = np.poly1d(coefficients)
 
# Print the polynomial
print("Fitted polynomial:", p)
 
# Use matplotlib to plot the original data and the fitted curve
xp = np.linspace(0, 5, 100)
plt.figure(figsize = (6,6), dpi = 600)
plt.scatter(x_new[:50], y_new[:50], c = "#8CBF87" )  # original data points
plt.scatter(x_new[50:], y_new[50:], c = "#8D2F25")  # original data points
if p.coeffs[1] > 0 and p.coeffs[2] > 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² + {}x + {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
elif p.coeffs[1] < 0 and p.coeffs[2] > 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² - {}x + {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
elif p.coeffs[1] > 0 and p.coeffs[2] < 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² + {}x - {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
elif p.coeffs[1] < 0 and p.coeffs[2] < 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² - {}x - {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
plt.legend()
plt.savefig("./Arial/urelated_fitting.png",dpi =600)
plt.show()

# Use numpy's polyfit for polynomial fitting, e.g., fitting a quadratic polynomial
coefficients = np.polyfit(x, y, 2)  # 2 means a quadratic polynomial
 
# Create a polynomial function from the fitted coefficients
p = np.poly1d(coefficients)
 
# Print the polynomial
print("Fitted polynomial:", p)
 
# Use matplotlib to plot the original data and the fitted curve
xp = np.linspace(0, 5, 100)
plt.figure(figsize = (6,6), dpi = 600)
plt.scatter(x[:50], y[:50], c = "#8CBF87" )  # original data points
plt.scatter(x[50:], y[50:], c = "#8D2F25")  # original data points
if p.coeffs[1] > 0 and p.coeffs[2] > 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² + {}x + {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
elif p.coeffs[1] < 0 and p.coeffs[2] > 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² - {}x + {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
elif p.coeffs[1] > 0 and p.coeffs[2] < 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² + {}x - {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
elif p.coeffs[1] < 0 and p.coeffs[2] < 0:
    plt.plot(xp, p(xp), color='red', ls='--', label = "y = {}x² - {}x - {}".format(str(abs(p.coeffs[0]))[:5],str(abs(p.coeffs[1]))[:5],str(abs(p.coeffs[2]))[:5]))
plt.legend()
plt.savefig("./Arial/perfect_fitting.png",dpi =600)
plt.show()



