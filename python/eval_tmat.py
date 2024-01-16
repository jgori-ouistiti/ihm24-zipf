from gen_zipf_markov_seq import global_matrix_search
from bigram import play_transition_matrix, zipf_plot, map_to_ints, joint_probs, cp_to_joint_ranks, plot_joint_independent_zipf
import numpy
from tqdm import tqdm

NC = numpy.linspace(10,150, 10)
NS = numpy.linspace(1, 2.5, 10)

observed_sparsity = numpy.zeros((len(NC), len(NS)))
observed_s = numpy.zeros((len(NC), len(NS)))
rsq_zipf = numpy.zeros((len(NC), len(NS)))
rsq_joint_zipf = numpy.zeros((len(NC), len(NS)))
zipf_param = 1.75




if __name__ == '__main__':

    for nc, num_commands in tqdm(enumerate(NC)):
        for ns, zipf_param in tqdm(enumerate(NS)):
            # try:
            num_commands = int(num_commands)
            P, trial , error = global_matrix_search(num_commands=num_commands, zipf_param = zipf_param, sparsity = 0.5, max_iter = 10000, inits = 20, verbose = False, error_stop = 0.0001)

            observed_sparsity[nc,ns] = numpy.sum(P < 0.01)/num_commands**2
            seq = play_transition_matrix(P, int(1e5))
            ax, fit_results_zip_freq = zipf_plot(seq, mode = 'sequence')
            rsq_zipf[nc,ns] = fit_results_zip_freq.rsquared
            observed_s[nc,ns] = -fit_results_zip_freq.params[1]

            int_seq, mapping = map_to_ints(seq, sort = True)
            cp = joint_probs(int_seq, distinct = False)
            obj = cp_to_joint_ranks(cp)
            ax,  bigrams = zipf_plot(obj,  mode = 'joint', fit = False, bigrams_out = True)
            ax, rsq = plot_joint_independent_zipf(fit_results_zip_freq, *bigrams, ax = None)
            rsq_joint_zipf[nc,ns] = rsq

                
            # except:
            #     observed_sparsity[nc,ns] = numpy.nan
            #     rsq_zipf[nc,ns]= numpy.nan
            #     observed_s[nc,ns]= numpy.nan
            #     rsq_joint_zipf[nc,ns]= numpy.nan


    _dict = {"observed_sparsity": observed_sparsity,
            "rsq_zipf": rsq_zipf,
            "observed_s": observed_s,
                "rsq_joint_zipf": rsq_joint_zipf}
    import pickle
    with open("eval.pkl", 'wb') as _file:
        pickle.dump(_dict, _file)



    



    