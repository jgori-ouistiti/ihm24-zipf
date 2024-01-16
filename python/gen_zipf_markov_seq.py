import numpy
from dataGen2_dali import *
import matplotlib.pyplot as plt
import scipy.optimize as opti
import seaborn
from heatmap_transition_matrix import plot_transition_matrix, tmat_shuffle

plt.style.use(style="fivethirtyeight")


num_commands = 20
sparsity = .4
zipf_param = 1.75

def zipf_plot(frequencies, ax = None, regplot_kwargs = None):
    import statsmodels.api as sm
    if ax is None:
        fig,ax = plt.subplots(1,1)

    default_regplot_kwargs = dict(label = 'Observed Frequencies', line_kws=dict(color="r",  lw = '2', alpha = .8))

    if regplot_kwargs is None:
        regplot_kwargs =  {}


    default_regplot_kwargs.update(regplot_kwargs)

    ranks = range(1, len(frequencies)+1)
    x = numpy.log(ranks)
    x_fit = sm.add_constant(numpy.log(ranks))
    fit_results = sm.OLS(numpy.log(frequencies), x_fit).fit()

    default_regplot_kwargs['line_kws'].update(dict(label=f'Zipf fit -- s= {-fit_results.params[1]:.2f}'))

    
    seaborn.regplot(y=numpy.log(frequencies), x=numpy.log(ranks), fit_reg = True, **default_regplot_kwargs, ax = ax)

    ax.set_xlabel(r'$\log k$')
    ax.set_ylabel(r'$\log f(s,k)$')

    return ax

def global_matrix_search(num_commands=40, zipf_param = 1.75, sparsity = .5, max_iter = None,inits = 100, verbose = True, error_stop = 0.01):

    Ps = []
    error = numpy.inf
    for ni, i in enumerate(range(inits)):
        init_m = initialize_matrix(num_commands,sparsity)
        P, trial, new_error = get_matrix_bis(num_commands, zipf_param = zipf_param, init_matrix = init_m, verbose = verbose, max_iter = 10, error_stop = error_stop)
        Ps.append(P)

        if abs(new_error) < abs(error):
            out = ni

    P, trial , error = get_matrix_bis(num_commands, zipf_param = zipf_param, init_matrix = Ps[out], verbose = verbose, max_iter = max_iter, error_stop=error_stop)

    return P, trial , error

    
if __name__ == '__main__':
    fig, axs = plt.subplots(2,2, figsize = (10,10))
    num_commands = 20
    sparsity = .3
    P, trial , error = global_matrix_search(num_commands=num_commands, zipf_param = zipf_param, sparsity = sparsity, max_iter = 10000,inits = 20, verbose = True, error_stop = 0.0001)
    # goal = get_goal_measure(num_commands,zipf_param)
    zipf_plot(trial, ax = axs[0,0])
    plot_transition_matrix(tmat_shuffle(P, numpy.ones(P.shape))[0], alpha_array = None, ax = axs[1,0], imshow_kwargs = dict(cmap = 'YlOrRd'))
    axs[0,0].legend()
    # seaborn.regplot(y=numpy.log(trial), x=numpy.log([i for i in range(1, num_commands+1)]), fit_reg = True,  ax = axs[0,0])
    # seaborn.regplot(y=numpy.log(goal), x=numpy.log([i for i in range(1, num_commands+1)]), fit_reg = True,  ax = axs[0,1])

    num_commands = 150
    sparsity = .3
    P, trial , error = global_matrix_search(num_commands=num_commands, zipf_param = zipf_param, sparsity = sparsity, max_iter = 1000,inits = 20, verbose = True, error_stop = 0.0001)
    import pickle
    with open('Pmatrix.pkl', 'wb') as _file:
        pickle.dump(P, _file)
    goal = get_goal_measure(num_commands,zipf_param)
    zipf_plot(trial, ax = axs[0,1])
    axs[0,1].legend()
    plot_transition_matrix(tmat_shuffle(P, numpy.ones(P.shape))[0], alpha_array = None, ax = axs[1,1], imshow_kwargs = dict(cmap = 'YlOrRd'))

    fig.subplots_adjust(wspace = .1, hspace = .1)
    fig.tight_layout()
    
    
    

    
    plt.show()


