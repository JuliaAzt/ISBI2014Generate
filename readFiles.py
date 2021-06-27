import scipy.io
import cv2
import os


def saveImage(mat, dir, column, extension, gt):
    i= 0
    if not os.path.exists(dir): os.makedirs(dir)
    for image in mat[column]:
        if(gt): cv2.imwrite(dir+str(i)+extension, image[0]* 255)
        else: cv2.imwrite(dir+str(i)+extension, image[0])
        i+=1


mat = scipy.io.loadmat('isbi_test90.mat')
saveImage(mat, "./test/", "ISBI_Test90",".png", False)
mat = scipy.io.loadmat('isbi_test90_GT.mat')
saveImage(mat, "./test/","test_Nuclei",".jpg", True)

mat = scipy.io.loadmat('isbi_train.mat')
saveImage(mat, "./train/", "ISBI_Train",".png", False)
mat = scipy.io.loadmat('isbi_train_GT.mat')
saveImage(mat, "./train/","train_Nuclei",".jpg", True)