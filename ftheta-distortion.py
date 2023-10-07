import numpy as np
import cv2


def get_undistorted_position_of_ray(x, y, cx, cy, inv_polynomials, fx, fy):
    """This Function Undistorts the distorted ray

    Args:
        x (numpy array): x cordinates of the distorted points
        y (numpy array): y coordinates of the distorted points
        cx (integer): the x value of principal point the image
        cy (integet): the y value of principal of the image
        inv_polynomials (numpy array): These are the j1, j2, j3, j4 in ftheta paper
        fx (int): fx in camera matrix
        fy (int): fy in camera matrix

    Returns:
         np arrays: updated x is the undistorted x values of x values provided
         np arrays: updated y is the undistorted y values of y values provided
    """
    px = x - cx
    py = y - cy
    pd = np.sqrt(px ** 2 + py ** 2)
    length_pd = len(pd)

    theta_values = np.zeros(length_pd)
    for place, coeff in enumerate(inv_polynomials):
        value = (pd ** (place + 1)) * coeff
        theta_values = np.add(theta_values, value)

    sin_theta_values = np.sin(theta_values)
    updated_x = np.multiply(px, sin_theta_values) / pd
    updated_y = np.multiply(py, sin_theta_values) / pd
    z_value = np.cos(theta_values)

    updated_x = np.rint(updated_x * fx / z_value + cx).astype(int)
    updated_y = np.rint(updated_y * fy / z_value + cy).astype(int)


    return updated_x, updated_y


def get_undistorted_image_ftheta_distortion(image, camera_matrix, inv_polynomials, calculate_forward_polynomial=True):
    """This functions gives you undistorted image for  ftheta distortion model

    Args:
        image (cv2 image): Image you want to undistort
        camera_matrix (numpy array): camera matrix of the camera
        inv_polynomials (numpy array): These are the j1, j2, j3, j4 in ftheta paper

    Returns:
        cv2 image: undistorted image
    """

    rows, columns, _ = image.shape
    fx, fy = camera_matrix[0][0], camera_matrix[1][1]
    cx, cy = camera_matrix[0][2], camera_matrix[1][2],

    undistorted_image = np.full(
        (int(rows), int(columns), 3), 0, dtype=np.uint8)
    row_indices, column_indices = np.meshgrid(
        np.arange(rows), np.arange(columns))
    y, x = row_indices.flatten(), column_indices.flatten()

    updated_x, updated_y = get_undistorted_position_of_ray(
        x, y,
        cx, cy,
        inv_polynomials,  fx, fy
    )

    # Keeping only the points which can fit in undistorted_image
    median_x = np.median(updated_x)
    median_y = np.median(updated_y)

    x_lower_bound = int(median_x-(columns/2))
    x_upper_bound = int(median_x+(columns/2))
    y_lower_bound = int(median_y-(rows/2))
    y_upper_bound = int(median_y+(rows/2))

    x_range_mask = (updated_x >= x_lower_bound) & (updated_x < x_upper_bound)
    y_range_mask = (updated_y >= y_lower_bound) & (updated_y < y_upper_bound)
    updated_x = np.where(x_range_mask, updated_x, x_upper_bound-1)
    updated_y = np.where(y_range_mask, updated_y, y_upper_bound-1)

    undistorted_image[updated_y, updated_x] = image[y, x]

    save_path = "/image_name"
    save_path_undistorted = save_path+"_undistorted"
    save_path = save_path+".jpg"
    save_path_undistorted = save_path_undistorted+".jpg"
    cv2.imwrite(save_path, image)
    cv2.imwrite(save_path_undistorted, undistorted_image)

    return undistorted_image