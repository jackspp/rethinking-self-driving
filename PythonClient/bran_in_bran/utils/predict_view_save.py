import matplotlib.pyplot as plt
import numpy as np
#from visualize_seg import labels_to_cityscapes_palette
from IPython.core.debugger import Tracer

def predict_view_save(target, predicts, step):
    """
    Visualize predicted results and targets

    # Parameters:
       - targets: [depth_imgs, rgb_imgs, seg_imgs]
       - predicts: [depth_imgs, rgb_imgs, seg_imgs]

    """

    row = len(predicts) - 1
    columns = 2

    # target imgs
    # t --> target, d --> depth etc.
    t_rgb_img = np.squeeze(target)

    # predict imgs
    p_d_img = np.squeeze(predicts[0])
    p_rgb_img = np.squeeze(predicts[1])
    p_rgb_img[p_rgb_img<0] = 0
    p_rgb_img[p_rgb_img>1] = 0

    p_seg_img = np.squeeze(np.argmax(predicts[2], -1))
    p_seg_img = labels_to_cityscapes_palette(p_seg_img)[:,:,::-1] / 255.0

    # Show

    fig = plt.figure(figsize=(15,10))



    p_ax = fig.add_subplot(row, columns, 2)
    plt.imshow(p_d_img,)
    p_ax.set_title("Perceptron results")

    t_ax = fig.add_subplot(row, columns, 3)
    plt.imshow(t_rgb_img)
    t_ax.set_title("input RGB")

    fig.add_subplot(row, columns, 4)
    plt.imshow(p_rgb_img)

    fig.add_subplot(row, columns, 6)
    plt.imshow(p_seg_img)

    plt.savefig("./predict_test/%s.png"%str(step))
    plt.close()




