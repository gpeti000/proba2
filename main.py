import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from _04_synthetic_data_polinomial import generate_synthetic data
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def fit_linear_regression(x,y):
    lr = LinearRegression()
    lr.fit(x.reshape(-1,1),y)
    return lr

def fit_polynomial_regression(x,y,degree):
    polynomial_regression = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_regression.fit(x.reshape(-1,1),y)
    return polynomial_regression

def generate_synthetic_data(x, coefficients, seed=42, noise_std = 1):
    np.random.seed(seed)
    y = np.polyval(coefficients[::-1],x) + np.random.normal(0, noise_std, len(x))
    return x,y


def visualize_data_and_fit(x,y, models,degrees):
    plt.scatter(x,y)
    plt.xlabel("Feature (x)")
    plt.ylabel("Target (y)")
    plt.title("Synhetic Data with Polynomial Relationship and Noise")
    x_pred = np.linspace(min(x), max(x), len(x)).reshape(-1,1)
    colors = ['red', 'blue', 'green']
    for model, degree, color in zip(models,degrees,colors):
        y_pred = model.predict(x_pred)
        plt.plot(x_pred, y_pred, color = color, label = degree)
    #plt.plot(x_pred,y_pred, color = 'red', label = 'Linear Regression Fit')
    plt.legend()
    plt.show()
    
def main():
    coefficients = [1, 0.02, -0.002, 0.014]
    x_values = np.linspace(-10, 10, 100)
    x,y = generate_synthetic_data(x_values, coefficients)
    degrees = [1,3,10]
    models = [fit_polynomial_regression(x,y,degree) for degree in degrees]
    #lr = fit_linear_regression(x,y)
    visualize_data_and_fit(x,y, models, degrees)

if __name__ == '__main__':
    main()