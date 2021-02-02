"""
Auto Cutter for deep-apple-learning 

First pre-processing step: 
1. Cut images out of grid
2. Brightness and Contrast correction, if imagery is RGB
3. Edge detection
4. Crop each slice to produce a single apple as result

To do:
- bug fix 
- decrease images when cutting out of grid 
- resize all images for model to same dimension
- distinguish between rgb and infrared images when exporting (folder structure)
"""

import numpy as np
from pathlib import Path
import cv2
import math


home = str(Path.home())
filepath = str(Path(__file__).parent)
filename = "grid8"
input_image_type = ".png"
output_image_type = ".jpg"
image_style = "rgb"  # or infrared
input_image = filename + input_image_type
output_image = filename + output_image_type

input_path = filepath + "/input/"
if image_style == "rgb":
    output_path = filepath + "/output/rgb/"
elif image_style == "infrared":
    output_path = filepath + "/output/infrared/"

images_horizontally = 9
images_vertically = 4


def equalize_image1(img_in):
    # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype("uint8")

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype("uint8")
    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype("uint8")
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    # print(equ)

    cv2.imwrite(filepath + filename + "_equ1.jpg", equ)
    return img_out


def equalize_image2(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(filepath + filename + "_equ2.jpg", img_output)


def crop_image(image):

    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ROI = image[y : y + h, x : x + w]
        break

    # Get a square image
    ROI_w = w - x
    ROI_h = h - y

    if ROI_w < ROI_h:
        # Increase width
        ROI_delta = ROI_h - ROI_w
        ROI_delta_half = math.ceil(ROI_delta / 2)

        # Resize Image
        ROI_square = image[
            y : y + h,
            x + ROI_delta_half : x + w + ROI_delta_half,
        ]

    elif ROI_w > ROI_h:
        # Increase Height
        ROI_delta = ROI_w - ROI_h
        ROI_delta_half = math.ceil(ROI_delta / 2)
        # Resize Image
        ROI_square = image[
            y - ROI_delta_half : y + h + ROI_delta_half,
            x : x + w,
        ]

    elif ROI_w == ROI_h:
        ROI_square = image[
            y : y + h,
            x : x + w,
        ]

    return ROI_square


def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def cut_grid(input_image, images_x, images_y):
    # Get image characteristics
    grid_height, grid_width, channels = input_image.shape
    print(grid_width, grid_height)

    images_width = grid_width / images_x
    images_height = grid_height / images_y
    print(images_width, images_height)

    x_start = 0
    x_end = images_width
    y_start = 0
    y_end = images_height
    counter = 0

    images_total = np.zeros(images_x * images_y, dtype=object)

    for i in range(images_x):
        x_start = int(i * images_width)
        x_end = int((i + 1) * images_width)

        for j in range(images_y):
            y_start = int(j * images_height)
            y_end = int((j + 1) * images_height)

            images_total[counter] = input_image[y_start:y_end, x_start:x_end]
            counter += 1

    return images_total


def main():

    # Open image
    image = cv2.imread(input_path + input_image)

    images = cut_grid(image, images_horizontally, images_vertically)

    for i in range(len(images)):
        image = images[i]

        # Apply pre-processing on rgb images for more contrast & to make brighter
        if image_style == "rgb":
            image = apply_brightness_contrast(image, 70, 50)

        # Crop
        result = crop_image(image)

        # Save image
        cv2.imwrite(output_path + "crop_" + str(i + 1) + ".jpg", result)
        print("Image " + str(i) + " done")


# --- MAIN ---
if __name__ == "__main__":
    # execute only if run as a script
    main()