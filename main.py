import os
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

#Deleting main_img.jpg file before making new
if os.path.isfile('main_img.jpg') == True:
    os.remove('main_img.jpg')

#Break before capture picture
time.sleep(5)
vid = cv2.VideoCapture(0)


#Starting Camera
while(True):
    ret, frame = vid.read()
    cv2.imshow('main_img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('main_img.jpg', frame)
        break




main_img = cv2.imread("main_img.jpg")
gray_main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)

#importing photos
img_list = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
ssim_scores = []
for img_path in img_list:
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the images to the same size as the main image
    size = (500, 500)
    gray_main_img_resized = cv2.resize(gray_main_img, size)
    gray_img_resized = cv2.resize(gray_img, size)

    ssim_score = ssim(gray_main_img_resized, gray_img_resized)
    ssim_scores.append((ssim_score, img_path))

ssim_scores.sort(reverse=True)

#Show most similar picture with SSIM score
most_similar_img_path = ssim_scores[0][1]
most_similar_ssim_score = ssim_scores[0][0]
print(f"Most similar photo is {most_similar_img_path} with SSIM score {most_similar_ssim_score}.")

#Show image
most_similar_img = cv2.imread(most_similar_img_path)
cv2.imshow("Main Image", main_img)
cv2.imshow("Most Similar Image", most_similar_img)
cv2.waitKey(0)


vid.release()

# Close all windows
cv2.destroyAllWindows()
