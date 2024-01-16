import pickle
import numpy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

with open("eval.pkl", 'rb') as _file:
    _dict = pickle.load(_file)


NC = list(numpy.linspace(10,150, 10))
NS = list(numpy.linspace(1, 2.5, 10))

observed_sparsity = _dict['observed_sparsity']
rsq_zipf = _dict['rsq_zipf']
observed_s = _dict['observed_s']
rsq_joint_zipf = _dict['rsq_joint_zipf']

fig, axs = plt.subplots(1,4)
axs[0].imshow(observed_sparsity)
axs[1].imshow(rsq_zipf)
axs[2].imshow(observed_s)
axs[3].imshow(rsq_joint_zipf)
plt.show()

#####===========
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=[f"{x:.2f}" for x in col_labels])
    ax.set_yticks(np.arange(data.shape[0]), labels=[int(x) for x in row_labels])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


from  matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 

from  matplotlib.colors import LinearSegmentedColormap
cmap2=LinearSegmentedColormap.from_list('gr',[ "g" ,"w", 'r'], N=256) 

fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Replicate the above example with a different font size and colormap.

im, _ = heatmap(observed_sparsity, NC, NS, ax=ax,
                cmap=cmap, cbarlabel="sparsity")
annotate_heatmap(im, valfmt="{x:.2f}", size=7)

# Create some new data, give further arguments to imshow (vmin),
# use an integer format on the annotations and provide some colors.

im, _ = heatmap(observed_s, NC, NS, ax=ax2, 
                cmap="Wistia", cbarlabel="Zipf s")
annotate_heatmap(im, valfmt="{x:.2f}", size=7)

# Sometimes even the data itself is categorical. Here we use a
# `matplotlib.colors.BoundaryNorm` to get the data into classes
# and use this to colorize the plot, but also to obtain the class
# labels from an array of classes.


im, _ = heatmap(rsq_zipf, NC, NS, ax=ax3,
                cmap=cmap, cbarlabel="R² Zipf")
annotate_heatmap(im, valfmt="{x:.2f}", size=7)

im, _ = heatmap(rsq_joint_zipf, NC, NS, ax=ax4,
                cmap=cmap2, cbarlabel="R² bigramme Zipf")
annotate_heatmap(im, valfmt="{x:.2f}", size=7)


# def func(x, pos):
#     return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

# annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)


plt.tight_layout()
plt.show()
