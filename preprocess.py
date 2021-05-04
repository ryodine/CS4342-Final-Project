import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("train.csv")
"""
def testim(hsv):
    def colorband(r, g, b, bw=20):
        color = np.uint8([[[b,g,r]]])
        hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)[0][0]
        hvs_color_lower = list(hsv_color)
        hvs_color_upper = list(hsv_color)
        hvs_color_lower[0] = hvs_color_lower[0] - bw 
        hvs_color_lower[1] = 60
        hvs_color_lower[2] = 60
        hvs_color_upper[0] = hvs_color_upper[0] + bw
        return np.array(hvs_color_lower), np.array(hvs_color_upper)

    l, u = colorband(0,255,0)
    print(l, u)
    mask = cv2.inRange(hsv, l, u)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    res1 = cv2.bitwise_and(img,img, mask= mask)

    plt.imshow(img[...,::-1],cmap = 'gray')
    plt.title('Orig'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.imshow(res1[...,::-1],cmap = 'gray')
    plt.title('Grn'), plt.xticks([]), plt.yticks([])
    plt.show()

    l, u = colorband(255,255,0)
    mask = cv2.inRange(hsv, l, u)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    res2 = cv2.bitwise_and(img,img, mask= mask)

    plt.imshow(res2[...,::-1],cmap = 'gray')
    plt.title('Ylw'), plt.xticks([]), plt.yticks([])
    plt.show()

    res3 = cv2.bitwise_or(res1,res2)

    plt.imshow(res3[...,::-1],cmap = 'gray')
    plt.title('either'), plt.xticks([]), plt.yticks([])
    plt.show()

    res4 = cv2.bitwise_and(res1,res2)

    plt.imshow(res4[...,::-1],cmap = 'gray')
    plt.title('both'), plt.xticks([]), plt.yticks([])
    plt.show()

    res5 = cv2.subtract(res1,res2)

    plt.imshow(res5[...,::-1],cmap = 'gray')
    plt.title('grn-yellow'), plt.xticks([]), plt.yticks([])
    plt.show()

for x in range(10):
    print(df.label[x])
    img = cv2.imread("train_images/" + df.image_id[x])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    testim(hsv)
"""

"""
img = cv2.imread("train_images/6103.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([30,60,130])
upper_blue = np.array([166,255,255])

#Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
#res = cv2.bitwise_and(img,img, mask= mask)
res = img
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,255,255)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,101,2)
ret, thresh2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img, mask= thresh)

hist = cv2.calcHist( [hsv], [0, 1], None, [30, 30], [0, 259, 0, 256] )

plt.imshow(res[...,::-1],cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(thresh,cmap = 'gray')
plt.title('Thresh Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(thresh2,cmap = 'gray')
plt.title('Thresh2 Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(hist)
plt.show()

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,750,650)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img[...,::-1]),plt.colorbar(),plt.show()
plt.show()
"""

DO_N_IMG = 1000

imgs = df.sample(n=2)
print(len(df))
preproc = np.zeros((len(df), 30, 30))
labs = np.zeros((len(df),))

"""

for i, im in df.iterrows():

    img = cv2.imread("train_images/" + im.image_id)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # lower_blue = np.array([30,50,50])
    # upper_blue = np.array([166,255,255])

    # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(img,img, mask= mask)

    hist = cv2.calcHist([hsv], [0, 1], None, [30, 30], [0, 259, 0, 256])
    print(i, im.label)

    # plt.imshow(hist)
    # plt.show()

    preproc[i] = hist
    labs[i] = im.label

    if i > DO_N_IMG:
        break

"""

"""
for i, im in imgs.iterrows():
    img = cv2.imread("train_images/" + im.image_id)

    #cv2.imshow('im',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    plt.imshow(img[...,::-1])
    plt.title('Image ' + im.image_id), plt.xticks([]), plt.yticks([])
    plt.show()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([30,50,50])
    upper_blue = np.array([166,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    plt.imshow(res[...,::-1],cmap = 'gray')
    plt.title('Image' + str(im.label)), plt.xticks([]), plt.yticks([])
    plt.show()

    hist = cv2.calcHist( [hsv], [0, 1], None, [20, 20], [30, 170, 20, 256] )
    plt.imshow(hist,interpolation = 'nearest')
    plt.show()

    hist = cv2.calcHist( [res], [0, 1], None, [20, 20], [30, 170, 20, 256] )
    plt.imshow(hist,interpolation = 'nearest')
    plt.show()
"""

for i, im in df.iterrows():

    img = cv2.imread("train_images/" + im.image_id)
    diff = np.zeros((len(img), len(img[0]), 3))

    axisInfo = [(1, len(img[0]) - 1, -1), (1, 0, 1), (0, len(img) - 1, -1), (0, 0, 1)]
    for j in range(3):
        oneColor = np.array(img[:, :, j]).astype(int)
        for axisVal, rowCol, leftRight in axisInfo:
            shift = np.roll(oneColor, leftRight, axis=axisVal)
            shift = np.abs(oneColor - shift)
            if axisVal == 1:
                shift[:, rowCol] = 0
            else:
                shift[rowCol] = 0
            diff[:, :, j] = diff[:, :, j] + shift

    if i == 0:
        plt.imshow(img)
        plt.show()
        plt.imshow(diff)
        plt.show()

    hist = cv2.calcHist([shift], [0, 1], None, [30, 30], [0, 256, 0, 256])
    print(i, im.label)

    if i == 3:
        plt.imshow(hist)
        plt.show()

with open('data.npy', 'wb') as f:
    np.save(f, preproc)

with open('labels.npy', 'wb') as f:
    np.save(f, labs)
