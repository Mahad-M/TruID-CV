# from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 256
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # skip frame if match quality is bad
    if matches[4].distance >= 22:
        return None, None

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("tmp/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    vid = cv2.VideoCapture("resources/card_videos/card_mahad.mp4")
    for i in range(34):
        ret, imReference = vid.read()
    vid = cv2.VideoCapture("resources/card_videos/card_mahad.mp4")
    while 1:
        ret, im = vid.read()
        imReg, h = alignImages(im, imReference)
        if imReg is not None:
            cv2.imshow('', cv2.resize(imReg, fx=0.25, fy=0.25, dsize=None))
        else:
            cv2.imshow('', np.zeros(cv2.resize(imReference, fx=0.25, fy=0.25, dsize=None).shape))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
