import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def contrastLimit(img):
    # Converting image to LAB Color so CLAHE can be applied to the luminance channel
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)
    # Apply histogram equalization to the L channel
    equ = cv2.equalizeHist(l)
    # Combine the Hist. equalized L-channel back with A and B channels
    updated_lab_img = cv2.merge((equ, a, b))
    # Convert LAB image back to color (RGB)
    hist_eq_img = cv2.cvtColor(updated_lab_img, cv2.COLOR_LAB2BGR)
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)
    # Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img, a, b))
    # Convert LAB image back to color (RGB)
    CLAHE_IMG = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    return CLAHE_IMG


def readImage(file):
    return cv2.imread(file, 1)


def canny(gray_img):
    return cv2.Canny(gray_img, 200, 200)


def removeSmallComponents(image, threshold):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)
    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def resizeImg(img):
    return cv2.resize(img, (480, 480))


def preprocess_img(img):
    image = contrastLimit(img)
    image = canny(image)
    return image


def findContour(image):
    # find contours in the thresholded image
    return cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def findCenterObject(contours, image):
    cnts = imutils.grab_contours(contours)
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


#         # draw the contour and center of the shape on the image
#         cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#         cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
#         cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         # show the image
#         cv2.imshow("Image", image)


def main():
    original_img = readImage('image9.jpg')
    resized_img = resizeImg(original_img)
    preprocessing_img = preprocess_img(resized_img)
    filter_small_comp = removeSmallComponents(preprocessing_img, 150)
    contour_img = findContour(filter_small_comp)
    findCenterObject(contour_img, resized_img)
    #     cv2.drawContours(resized_img, contour_img, -1, (0, 255, 0), 2)
    #     cv2.imshow('Contours Image', resized_img)

    #     cv2.imshow('Original Image', original_img)
    #     cv2.imshow('Resized Image', resized_img)
    #     cv2.imshow('Preprocessing Image', preprocessing_img)
    #     cv2.imshow('Filter Out Small Component On Image', filter_small_comp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()