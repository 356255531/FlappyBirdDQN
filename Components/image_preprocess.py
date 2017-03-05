from scipy.misc import imresize
import numpy as np


def image_preprocess(frame_set):
    """
        Cut the images and perform downsampling
    """
    frame_set = frame_set[0:400, :, :]
    return imresize(frame_set, [80, 80, 4])


def main():
    from scipy.misc import imread
    screenshot = imread("screenshot.png")
    screenshot = image_preprocess(screenshot)

    import matplotlib.pyplot as plt
    plt.imshow(screenshot)
    plt.show()


if __name__ == '__main__':
    main()
