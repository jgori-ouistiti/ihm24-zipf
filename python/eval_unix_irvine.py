import os 
import pandas as pd
import pandas
import numpy as np
import seaborn
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter
import statsmodels.api as sm
from heatmap_transition_matrix import plot_transition_matrix, tmat_shuffle
from bigram import zipf_plot, plot_joint_independent_zipf
from tqdm import tqdm
import re





def path(user) : 
    return "data/unix_uc_irvine/"+user

def remove_items(test_list, item = "**SOF**"):
 
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
    return res


def remove_options(test_list):
    res = [i for i in test_list if '-' not in i]
    return res

def remove_token(test_list):
    res = [i for i in test_list if not re.match(r"<[0-9]+>", i)]
    return res

def read_file(file_path) :
    file = open(file_path, 'r')
    Lines = file.readlines()
    command_sequence = []
    for line in Lines :
        command_sequence.append(line[:-1])
    
    return command_sequence



def map_to_ints(seq, sort = False) : 
    if not sort: 
        all_commands = list(set(seq))
        dictionary = {}
        rev_dictionary = {}
        for index,command in enumerate(all_commands) :
            dictionary[command] = index
            rev_dictionary[index] = command

        mapped = pd.Series(seq).map(dictionary)
        return list(mapped),rev_dictionary
    else:
        counter = Counter(seq)
        counts = counter.most_common(None)
        out = {}
        for indx, (cmd, cts) in enumerate(counts):
            out[cmd] = indx + 1
        out_seq = []
        for s in seq:
            out_seq.append(out[s])
        return out_seq, out


def mix_key(command1, command2):
    return str(command1) + '->' + str(command2)

def joint_probs(command_sequence, distinct = False):
    int_seq, mapping = map_to_ints(command_sequence)
    cp = {}
    for e,actual in enumerate(int_seq[1:]) :
        prev = int_seq[e]

        if distinct:
            if cp.get(mix_key(prev, actual), None) is None:
                cp[mix_key(prev, actual)] = 0
        
            cp[mix_key(prev, actual)]+= 1
       
        else:

            key = cp.get(mix_key(prev, actual), None)
            if key is None:
                key = cp.get(mix_key(actual, prev), None)
            if key is None:
                cp[mix_key(prev, actual)] = 0
            try:
                cp[mix_key(prev, actual)]+= 1
            except KeyError:
                cp[mix_key(actual, prev)]+= 1

    return cp

def transition_matrix(command_sequence, sort_by_most_frequent = False) :
    int_seq, mapping = map_to_ints(command_sequence)
    dim = len(set(int_seq))
    Matrix = np.zeros((dim,dim))
    for e,actual in enumerate(int_seq[1:]):
        if actual == "**SOF**" or e == "**SOF**":
            continue
        prev = int_seq[e]
        actual = int_seq[e+1]
        Matrix[prev,actual] += 1


    TMatrix = Matrix/numpy.tile(numpy.sum(Matrix, axis = 1).reshape(1,-1), [dim,1]).T

    if not sort_by_most_frequent:
        return TMatrix, Matrix, mapping

    raise NotImplementedError('must be updated')
    df = pandas.DataFrame(Matrix)
    s = df.sum()
    idx =s.sort_values(ascending=False).index
    df = df[idx]
    _map = {k:mapping[i] for k, i in enumerate(idx)}
    return numpy.asarray(df), counts, _map






def cp_to_joint_ranks(cp):
    joint_rank = []
    counts = []
    for key, value in cp.items():
        k, k_prime = key.split('->')
        k = int(float(k) +1)
        k_prime = int(float(k_prime)) +1
        joint_rank.append(k*k_prime)
        counts.append(value)
    return counts, joint_rank


def gen_zipf_seq(n_elements,size, s = 1.5) :

    freqs = numpy.array([i**(-s) for i in range(1, n_elements+1) ])
    probs = freqs/numpy.sum(freqs)
    print(probs)
    return numpy.random.choice(n_elements, size = size, replace = True, p = probs)

def play_transition_matrix(P, size):
    dim = P.shape[0]
    seq = [int(numpy.random.random()*dim)]
    for n in range(size):
        seq.append(numpy.random.choice(dim, p = P[seq[n],:]))
    return seq[1:]

if __name__ == '__main__':
    plt.style.use(style="fivethirtyeight")


    users = [u for u in os.listdir("data/unix_uc_irvine") if 'USER' in u]
    observed_sparsity = numpy.zeros((len(users),))
    observed_s = numpy.zeros((len(users),))
    rsq_zipf = numpy.zeros((len(users),))
    rsq_joint_zipf = numpy.zeros((len(users),))


    for n_user, user in tqdm(enumerate(users)): 
        cs= read_file(path(user))
        cs = remove_options(cs)
        cs = remove_token(cs)
        cs = remove_items(cs, item = '**EOF**' )

        
        fig, axs = plt.subplots(nrows = 1, ncols = 3, width_ratios=[1, 1, 1.5], figsize = (15,6))
        
        trans_matrix, count_matrix, mapping  = transition_matrix(cs, sort_by_most_frequent=False)         


        plot_transition_matrix(tmat_shuffle(trans_matrix, numpy.ones(trans_matrix.shape))[0], alpha_array = None,  imshow_kwargs = dict(cmap = 'YlOrRd'), ax = axs[2])


        observed_sparsity[n_user] = numpy.sum(trans_matrix < 0.01)/trans_matrix.shape[0]**2

        axs[2].set_title(f'Sparsity={observed_sparsity[n_user]:.2f}')
        

        cs = remove_items(cs, item = '**SOF**' )
        ax, fit_results_zip_freq = zipf_plot(cs, mode = "sequence", ax = axs[0], fit = True, ax_kws = {'label': f'Sequence {n_user}'})
        axs[0].set_title(f"Indep. Zipf, s={-fit_results_zip_freq.params[1]:.2f}, R²={fit_results_zip_freq.rsquared:.2f}")
        rsq_zipf[n_user] = fit_results_zip_freq.rsquared
        observed_s[n_user] = -fit_results_zip_freq.params[1]

        int_seq, mapping = map_to_ints(cs, sort = True)
        cp = joint_probs(int_seq, distinct = False)
        obj = cp_to_joint_ranks(cp)
        ax, bigrams = zipf_plot(obj,  mode = 'joint', fit = False, bigrams_out = True, ax = axs[1])
        ax, rsq = plot_joint_independent_zipf(fit_results_zip_freq, *bigrams, ax = axs[1])
        axs[1].set_title(f"Joint Zipf, R²={rsq:.2f}")
        rsq_joint_zipf[n_user] = rsq

        axs[0].legend()
        axs[1].legend()

        plt.tight_layout(w_pad = -1)
        # plt.savefig(f'img/unix_irvine_{n_user}.pdf')
        plt.close()


import pandas
df = pandas.DataFrame({"Sparsity": observed_sparsity, "R² Zipf": rsq_zipf, "R² Joint Zipf": rsq_joint_zipf, "Observed s": observed_s})

fig, ax = plt.subplots(1,1)
seaborn.barplot(df, ax = ax)
ax.set_ylabel('Observed value')
ax.bar_label(ax.containers[0], fontsize=14, fmt = "%.2f", padding = 10)
plt.tight_layout()
plt.show()
            
