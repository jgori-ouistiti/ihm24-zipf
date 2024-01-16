import numpy
import matplotlib.pyplot as plt

plt.style.use(style="fivethirtyeight")



def plot_transition_matrix(matrix, alpha_array = None, ax = None, imshow_kwargs = None):
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, ncols =1)
    
    if imshow_kwargs is None:
        imshow_kwargs = {}

    if alpha_array is None:
        alpha_array = numpy.ones(matrix.shape)

    default_imshow_kwargs = dict(cmap = 'cool')
    default_imshow_kwargs.update(imshow_kwargs)
    
    ax.imshow(matrix, alpha = alpha_array, **default_imshow_kwargs)
    ax.set_xlabel("To command #")
    ax.xaxis.tick_top()
    ax.set_ylabel("From command #")    
    ax.xaxis.set_label_position('top')


    return ax

def permut(transition_matrix, alpha_array):
        p = numpy.random.permutation(transition_matrix.shape[0])
        return transition_matrix[p], alpha_array[p]

def tmat_shuffle(transition_matrix, alpha_array):
    t, a = permut(transition_matrix, alpha_array)
    t,a = permut(t.T, a.T)
    return t.T, a.T

if __name__ == '__main__':


    mat = numpy.zeros((150,150))
    mat[:,3] = 1
    mat[:,45] = 1

    counts = numpy.tile(numpy.array(list(range(1,151))).reshape(-1,1), [1,150])
    counts = counts/numpy.max(counts)


    plot_transition_matrix(mat, counts)
    plt.show()
