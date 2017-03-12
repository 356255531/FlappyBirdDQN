from scipy.misc import imresize
import numpy as np
import pdb
import cv2


def image_preprocess(frame_set):
    """
        Cut the images and perform downsampling
    """
    frame_set = frame_set[0:399, :, :]
    # pdb.set_trace()
    for x in xrange(4):
        frame_set[:, :, x] = cv2.threshold(frame_set[:, :, x], 1, 255, cv2.THRESH_BINARY)
    # pdb.set_trace()`
    frame_set = imresize(frame_set, [80, 80, 4])
    return frame_set


def main():
    from scipy.misc import imread
    screenshot = imread("screenshot.png")
    screenshot = image_preprocess(screenshot)

    import matplotlib.pyplot as plt
    plt.imshow(screenshot)
    plt.show()


if __name__ == '__main__':
    main()
