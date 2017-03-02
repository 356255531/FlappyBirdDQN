from scipy.misc import imresize
import numpy as np


def image_preprocess(frame_set):
    """
        Cut the images and perform downsampling
    """
    frame_set = frame_set[:, 0:400, :]
    frame_set_size = frame_set.shape
    new_frame_set = []
    for x in xrange(frame_set_size[0]):
        frame = imresize(
            frame_set[x, :, :], [80, 80]
        )
        new_frame_set.append(frame)
    return np.array(new_frame_set)


def main():
    from scipy.misc import imread
    screenshot = imread("screenshot.png")
    screenshot = image_preprocess(screenshot)

    import matplotlib.pyplot as plt
    plt.imshow(screenshot)
    plt.show()


if __name__ == '__main__':
    main()
