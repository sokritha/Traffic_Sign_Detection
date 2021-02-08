import cv2
from skimage import io, morphology
from matplotlib import pyplot as plt


# Reading image
img = cv2.imread("image1.jpg", 1)

original_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Converting image to LAB Color so CLAHE can be applied to the luminance channel
lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#Splitting the LAB image to L, A and B channels, respectively
l, a, b = cv2.split(lab_img)

###########Histogram Equlization#############
#Apply histogram equalization to the L channel
equ = cv2.equalizeHist(l)

#Combine the Hist. equalized L-channel back with A and B channels
updated_lab_img1 = cv2.merge((equ,a,b))

#Convert LAB image back to color (RGB)
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

###########CLAHE#########################
#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
gray = cv2.cvtColor(CLAHE_img, cv2.COLOR_BGR2GRAY)


# Laplacian Original Image
ori_laplacian_img = cv2.Laplacian(original_grey, cv2.CV_64F)
# cv2.imshow("Edge Laplacian Original image", ori_laplacian_img)

# Laplacian CLAHE Image
clahe_laplacian_img = cv2.Laplacian(gray, cv2.CV_64F)
# cv2.imshow("Edge Laplacian CLAHE image", clahe_laplacian_img)

#Canny Original Image
ori_edges = cv2.Canny(original_grey, 100, 200)
# cv2.imshow("Edge Canny Original image", ori_edges)

#Canny CLAHE
clahe_edges = cv2.Canny(gray, 200, 200)
cv2.imshow('Edge Canny CLAHE image', clahe_edges)


# cv2.imshow("Original image", img)
# cv2.imshow("Equalized image", hist_eq_img)
cv2.imshow('Gray original Image', original_grey)
# cv2.imshow('Gray CLAHE Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()