#####################################################################
#KÖNYVTÁRAK MEGHÍVÁSA
#####################################################################

import cv2
import numpy as np
#import matplotlib.pyplot as plt

#####################################################################
#KÉPEK BEOLVASÁSA
#####################################################################

# a vizsgált kép
img1= cv2.imread('forintermek.png') 

# a keresett érme a képen
img2 = cv2.imread('letoltes.png')
img4 = cv2.imread('ketszazas2.png')

#####################################################################
#MINTAKERESÉS
#####################################################################

# SHIFT detektor
orb = cv2.ORB_create()

# jellemzőpontok megkeresése és leírók a SHIFT-tel
keypoints1, descriptors1 = orb.detectAndCompute(img1,None)
keypoints2, descriptors2 = orb.detectAndCompute(img2,None)

# egyszerű Bruce-Force algoritmus megadása
bruce_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# leírok összekötése
matches = bruce_force.match(descriptors1,descriptors2)

# távolság alapján kiválogatni az összekötött leírókat
matches = sorted(matches, key = lambda x:x.distance)

# az első tíz találat kirajzolása
img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:10], None , flags=6)

#találatok ábrázolása
cv2.imshow('Jellemzopontok',img3)
cv2.waitKey(0)

#####################################################################
#ÉRMÉK ÖSSZESZÁMOLÁSA
#####################################################################

#kép méretezés
im_m= img1[0:600, 0:800] 

# Otsu's thresholding after Gaussian filtering 
blur = cv2.GaussianBlur(im_m,(25,25),0)
(thresh, im) = cv2.threshold(blur, 55, 130, cv2.THRESH_BINARY_INV )

# erózió és dilatáció 
K = np.array([[0,1,0],[1,1,1],[0,1,0]])
Id_erozio=cv2.erode(im,K.astype(np.uint8),iterations=1)#erózió
Id_dilatacioId_erozio=cv2.erode(Id_erozio,K.astype(np.uint8),iterations=1)

# foltdetektáló paramétereinek megadása
params = cv2.SimpleBlobDetector_Params()

#paraméterek megadása
params.minThreshold = 10;
params.maxThreshold = 225;

params.filterByArea = True
params.minArea = 1000
params.maxArea = 150000

params.filterByCircularity = False
params.minCircularity = 0.5

params.filterByConvexity = False
params.minConvexity = 0.2

params.filterByInertia = False
params.minInertiaRatio = 0.01

#foltdetektálás a megadott paraméterekkel az opencv verziója alapján
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)

#jellemzőpontok megadása
keypoints = detector.detect(Id_dilatacioId_erozio)

#a jellemzőpontok alapján a foltok bejelölése a vizsgált képen
im_key = cv2.drawKeypoints(Id_dilatacioId_erozio, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# a végeredmény megjelenítése
cv2.imshow("Penzermek", im_key)
cv2.waitKey(0)
cv2.destroyAllWindows()

# jellemző pontok alapján a foltok összeszámolása
s = len(keypoints)
print('Pénzérmék száma: ', s)