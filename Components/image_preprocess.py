from scipy.misc import imresize
import numpy as np
import pdb


def image_preprocess(frame_set):
    """
        Cut the images and perform downsampling
    """
    frame_set = frame_set[0:399, :, :]
    idx_1 = frame_set > 0
    idx_2 = frame_set <= 0
    frame_set[idx_1] = 255
    frame_set[idx_2] = 0

    new_frame_set = imresize(frame_set, [80, 80, 4])

    return new_frame_set


def main():
    from scipy.misc import imread
    screenshot = imread("screenshot.png")
    screenshot = image_preprocess(screenshot)

    import matplotlib.pyplot as plt
    plt.imshow(screenshot)
    plt.show()


if __name__ == '__main__':
    main()
