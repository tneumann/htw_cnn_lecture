import numpy as np
import torch
import matplotlib.pyplot as plt


def show_image_grid(X, y=None, y_true=None, title=None, nrow=6, ncol=4, sz_factor=2):
    max_num = nrow*ncol
    X = X[:max_num]
    if len(X) < max_num:
        ncol = len(X) // nrow + 1
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if X.dim() != 4:
        X = X[:, None]

    plt.figure(title, figsize=(sz_factor*nrow, sz_factor*ncol + (0 if y is None else 1.5)))
    if title:
        plt.title(title)
        
    y = [None] * len(X) if y is None else y
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    y_true = [None] * len(X) if y_true is None else y_true
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
        
    for i, (Xi, yi, yi_true) in enumerate(zip(X, y, y_true)):
        plt.subplot(ncol, nrow, i+1)
        img = Xi.numpy().transpose((1, 2, 0))
        if img.shape[2] == 1:
            img = img[..., 0]
        plt.imshow(img)

        title = ""
        color = 'black'
        if yi is not None:
            title += "pr: %d" %yi
        
        if yi_true is not None:
            title += " is: %d" % yi_true
            color = 'green' if (yi == yi_true) else 'red'

        if len(title) > 0:
            plt.title(title, color=color)

        plt.axis('off')
    
    plt.axis('off')
