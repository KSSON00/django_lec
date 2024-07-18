import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data[['compass_x', 'compass_y', 'compass_z']].to_numpy()

def ellipsoid_residuals(params, data):
    xc, yc, zc, a, b, c, alpha, beta, gamma = params
    R_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    R_beta = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    R_gamma = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    rotation_matrix = R_alpha @ R_beta @ R_gamma
    transformed_data = (data - [xc, yc, zc]) @ rotation_matrix.T
    residuals = (transformed_data[:, 0] / a) ** 2 + (transformed_data[:, 1] / b) ** 2 + (
            transformed_data[:, 2] / c) ** 2 - 1
    return residuals

def fit_ellipsoid(data):
    x0 = np.array([np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2]),
                   np.std(data[:, 0]), np.std(data[:, 1]), np.std(data[:, 2]),
                   0, 0, 0])
    result = least_squares(ellipsoid_residuals, x0, args=(data,))
    return result.x

def calibrate_magnetometer(data):
    params = fit_ellipsoid(data)
    xc, yc, zc, a, b, c, alpha, beta, gamma = params
    center = np.array([xc, yc, zc])
    radii = np.array([a, b, c])
    R_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    R_beta = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    R_gamma = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    rotation_matrix = R_alpha @ R_beta @ R_gamma
    transform_matrix = rotation_matrix @ np.diag(1 / radii)
    calibrated_data = (data - center) @ transform_matrix.T

    print("Center:", center)
    print("Radii:", radii)
    print("Rotation Matrix:\n", rotation_matrix)
    print("Transformation Matrix:\n", transform_matrix)

    return calibrated_data

def plot_data(raw_data, calibrated_data=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2], color='r', label='Raw Data')
    if calibrated_data is not None:
        ax.scatter(calibrated_data[:, 0], calibrated_data[:, 1], calibrated_data[:, 2], color='b', label='Calibrated Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def main():
    file_path = "C:/Users/admin/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.12/raw8.csv"
    calibrated_file_path = "C:/Users/admin/AppData/Local/Programs/Python/Python312/새 폴더/calibrated_data.txt"

    compass_raw = read_data(file_path)
    plot_data(compass_raw)

    compass_cal = calibrate_magnetometer(compass_raw)
    plot_data(compass_raw, compass_cal)

    true_field = np.array([np.mean(compass_cal[:, 0]), np.mean(compass_cal[:, 1]), np.mean(compass_cal[:, 2])])
    rmse = np.sqrt(np.mean((compass_cal - true_field) ** 2, axis=0))
    average_rmse = np.mean(rmse)

    print(f"RMSE per axis: {rmse}")
    print(f"Average RMSE: {average_rmse}")

    np.savetxt(calibrated_file_path, compass_cal, delimiter=',')
    print(f"Calibrated data saved to {calibrated_file_path}")

if __name__ == "__main__":
    main()
