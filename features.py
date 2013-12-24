import numpy as np
from random import random
import cv2

SIFT = cv2.SIFT(nfeatures=20)
BRISK = cv2.BRISK()

# class WindowTypeFeatureFinder():
#     def __init__(self):
#         self.descriptors = []

#     def descriptor_at(self, img, loc):


#     def descriptors_from(self, img):
#         for loc in self.locations_in(img):
#             yield self.descriptor_at(img, loc)

#     def add_image(self, img):
#         descriptors = self.descriptors_from(img)
#         self.descriptors.extend(descriptors)


PATCH_RADIUS = 5

def random_patches(img, count, radius=5):
    for i in xrange(count):
        patch = random_patch(img, radius=radius)
        yield np.reshape(patch, (2*radius)**2)

def random_patch(img, radius=5):
    rx = random()
    ry = random()
    rows, cols = img.shape
    cx = int(rx * (cols - 2 * radius - 2)) + radius + 1
    cy = int(ry * (rows - 2 * radius - 2)) + radius + 1
    return img[cy-radius:cy+radius, cx-radius:cx+radius]

def as_img(data):
    img = np.zeros(data.shape, np.uint8)
    img[:] = 5 * (data + 15) # make all data positive then scale...
    return img

def from_specgram(Pxx):
    Pxx = np.log(Pxx)
    print "Stats:", np.max(Pxx), np.min(Pxx), np.mean(Pxx), np.median(Pxx), np.std(Pxx)

    img = as_img(Pxx)

    print SIFT.detect(img)

    return [random(), random()]
