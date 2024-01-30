
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from scipy.ndimage import zoom

def show_loader(loader):
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show
    show_tensor(images)


def show_tensor(tensor):
    def imshow(img):
        # unnormalize
        # img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    imshow(torchvision.utils.make_grid(tensor))


def show_heatmap(img, heatmap, title=''):
    # import matplotlib
    # matplotlib.use('TkAgg')

    scale = img.shape[0] / heatmap.shape[0]
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(zoom(heatmap, zoom=(scale, scale)), cmap='jet', alpha=0.35)
    ax.axis('off')
    ax.set_title(title)
    fig.set_facecolor((0.2,0.2,0.2))
    # plt.get_current_fig_manager().window.state('zoomed')
    # fig.set_size_inches(10, 10)
    # plt.tight_layout(pad=1)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()


def show_image(img, title=''):
    import matplotlib
    matplotlib.use('TkAgg')

    if img.ndim == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    if img.ndim == 2: 
        ax.imshow(img, cmap='gray')
    else: 
        ax.imshow(img)

    ax.axis('off')
    ax.set_title(title)
    fig.set_facecolor((0.2,0.2,0.2))  
    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout(pad=1)
    plt.show(block=True)


def show_image2(img1, img2, title1='show_image1', title2='show_image2'):
    import matplotlib
    matplotlib.use('TkAgg')
    
    if img1.ndim > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if img2.ndim > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2)

    if img1.ndim == 2: ax[0].imshow(img1, cmap='gray')
    else: ax[0].imshow(img1)

    if img2.ndim == 2: ax[1].imshow(img2, cmap='gray') 
    else: ax[1].imshow(img2)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title(title1)
    ax[1].set_title(title2)

    # fig.set_facecolor((0.2,0.2,0.2))  
    plt.get_current_fig_manager().window.state('zoomed')
    plt.tight_layout(pad=1)
    plt.show(block=True)


def conv_visualization(img, conv, pred, pred_w, pred_b):
    idx = 0
    conv = conv.cpu().detach()[idx]
    pred = pred.cpu().detach()[idx]
    pred_w = pred_w.cpu().detach()
    pred_b = pred_b.cpu().detach()
    img = img[idx]

    target = np.argmax(pred).squeeze()
    w, b = pred_w, pred_b
    weights = w[target, :].numpy()

    c = conv.squeeze().numpy().reshape(-1, 13*13)
    heatmap = weights @ c
    heatmap = heatmap.reshape(13, 13)

    show_heatmap(img, heatmap, 'show heatmap')
    print('')