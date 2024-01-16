from gen_zipf_markov_seq import global_matrix_search
from bigram import play_transition_matrix, zipf_plot, map_to_ints, joint_probs, cp_to_joint_ranks, plot_joint_independent_zipf
from heatmap_transition_matrix import plot_transition_matrix, tmat_shuffle

import numpy
import matplotlib.pyplot as plt

zipf_param = 1.88
NC = 239
sparsity = 0.5





if __name__ == '__main__':


    

    P, trial , error = global_matrix_search(num_commands=NC, zipf_param = zipf_param, sparsity = sparsity, max_iter = 10000, inits = 20, verbose = True, error_stop = 0.0001)

    fig, axs = plt.subplots(nrows = 1, ncols = 3, width_ratios=[1, 1, 1.5], figsize = (15,6))
    observed_sparsity = numpy.sum(P < 0.01)/NC**2
    seq = play_transition_matrix(P, 10808)
    ax, fit_results_zip_freq = zipf_plot(seq, mode = 'sequence', ax = axs[0], fit = True, ax_kws = {'label': f'Synthetic Sequence'})
    axs[0].set_title(f"Indep. Zipf, s={-fit_results_zip_freq.params[1]:.2f}, R²={fit_results_zip_freq.rsquared:.2f}")
    rsq_zipf = fit_results_zip_freq.rsquared
    observed_s = -fit_results_zip_freq.params[1]
    int_seq, mapping = map_to_ints(seq, sort = True)
    cp = joint_probs(int_seq, distinct = False)
    obj = cp_to_joint_ranks(cp)
    plot_transition_matrix(tmat_shuffle(P, numpy.ones(P.shape))[0], alpha_array = None,  imshow_kwargs = dict(cmap = 'YlOrRd'), ax = axs[2])
    axs[2].set_title(f'Sparsity={observed_sparsity:.2f}')
    ax, bigrams = zipf_plot(obj,  mode = 'joint', fit = False, bigrams_out = True, ax = axs[1])
    ax, rsq = plot_joint_independent_zipf(fit_results_zip_freq, *bigrams, ax = axs[1])
    axs[1].set_title(f"Joint Zipf, R²={rsq:.2f}")
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel('log rang joint')
    axs[1].set_ylabel('log prob. bigramme')
    axs[2].set_xlabel('À la commande #')
    axs[2].set_ylabel('De la commande #')
    plt.tight_layout(w_pad = -1)
    rsq_joint_zipf = rsq
    plt.show()


                



    



    