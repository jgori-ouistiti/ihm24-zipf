import os 
import pandas as pd
import pandas
import numpy as np
import seaborn
import numpy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter

from heatmap_transition_matrix import plot_transition_matrix


plt.style.use(style="fivethirtyeight")

users = os.listdir("data/unix_uc_irvine")
users = [u for u in users if "USER0" in u]


ZIPF = False
TRANSITION_MATRIX = True

def path(user) : 
    return "data/unix_uc_irvine/"+user

def remove_items(test_list, item = "**SOF**"):
 
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
    return res



def read_file(file_path) :
    file = open(file_path, 'r')
    Lines = file.readlines()
    command_sequence = []
    for line in Lines :
        command_sequence.append(line[:-1])
    
    return command_sequence

COMMAND_SEQUENCES = []
for user in users : 
    COMMAND_SEQUENCES.append(read_file(path(user)))

for i in range(len(COMMAND_SEQUENCES)) :
    COMMAND_SEQUENCES[i] = remove_items(remove_items(COMMAND_SEQUENCES[i]), item = '**EOF**' )
    
def compute_rsq_bigram(fit_results_zipf_frequency, bigrams, ax = None):
    x,y = bigrams # should be log values
    a, b = 2*fit_results_zipf_frequency.params[0], fit_results_zipf_frequency.params[1]
    actual = y
    predict = [a + _x*b for _x in x]
    corr_matrix = numpy.corrcoef(actual, predict)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq

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
    for e,actual in enumerate(int_seq[1:]) :
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

matrices = [transition_matrix(cs, sort_by_most_frequent=False) for cs in COMMAND_SEQUENCES]

trans_matrix, count_matrix, mapping = matrices[0]


  
def zipf_plot(_object, mode = "sequence", ax = None, fit = True, ax_kws = None, bigrams_out = False):
    if mode == "sequence":
        i = pd.Series(_object)
        temp = i.value_counts().to_dict()
        count_dict = {}
        for rank,i in enumerate(temp.keys()):
            count_dict[rank + 1] = temp[i]
        ranks = np.array(list(count_dict.keys()))
        occurances = np.array(list(count_dict.values()))

    elif mode == 'joint':
        occurances, ranks = _object

    else:
        sorted_cmds = sorted(_object.items(), key=lambda x:x[1], reverse = True)

        ranks = list(range(1, len(sorted_cmds)+1))
        occurances = numpy.array([s[1] for s in sorted_cmds])

    frequencies = occurances/numpy.sum(occurances)
    

    ax_kws_default = {'label' : 'observed data'}
    if ax_kws is None:
        ax_kws = {}

    ax_kws_default.update(ax_kws)

    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    x=np.log(ranks)
    x_fit = sm.add_constant(np.log(ranks))
    fit_results = sm.OLS(np.log(frequencies), x_fit).fit()
    seaborn.regplot(y=np.log(frequencies), x=np.log(ranks), fit_reg = fit, **ax_kws_default, line_kws=dict(color="r", label=f'Zipf fit -- s= {-fit_results.params[1]:.2f}', lw = '2', alpha = .8), ax = ax)
    ax.set_xlabel(r'$\log k$')
    ax.set_ylabel(r'$\log f(s,k)$')
    if fit:
        ret =  ax, fit_results
    else:
        ret = (ax,)

    if bigrams_out:
        return *ret, (x, np.log(frequencies))

    return ret

    


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

def plot_joint_independent_zipf(fit_results_frequency_zipf, x, y, ax = None):
    a, b = 2*fit_results_frequency_zipf.params[0], fit_results_frequency_zipf.params[1]
    _abs = [numpy.min(x), numpy.max(x)]
    if ax is not None:
        ax.plot([x for x in _abs], [a +x*b for x in _abs], '-', label = "joint-zipf")
        ax.set_xlabel('Log joint rank')
        ax.set_ylabel('Log Bigram prob.')
    rsq = compute_rsq_bigram(fit_results_frequency_zipf, (x,y))
    return ax, rsq

if __name__ == '__main__':

    if ZIPF:

        import statsmodels.api as sm

        for user,i in enumerate(COMMAND_SEQUENCES) :
            zipf_plot(i)
            # from sklearn.linear_model import LinearRegression
            # plt.xlabel("log occurance")
            # plt.ylabel("log rank")
            # model = LinearRegression().fit(np.log(occurances).reshape(-1,1),np.log(ranks))
            # plt.plot(np.log(occurances),model.predict(np.log(occurances).reshape(-1,1)))
            # plt.savefig("LogLog plot for the Linux Data.jpg")
            # plt.show()
            if user == 0 :
                break


    if TRANSITION_MATRIX:


                


        fig, axs = plt.subplots(nrows = 3, ncols = 2)
        # id zipf
        zipf_seq = gen_zipf_seq(150, int(1e5), s = 1.75)
        ax, fit_results = zipf_plot(zipf_seq, mode = 'sequence', ax  = axs[0,0], ax_kws = {'label': 'Independent Zipf'})
        int_seq, mapping = map_to_ints(zipf_seq, sort = True)
        cp = joint_probs(int_seq, distinct = False)
        obj = cp_to_joint_ranks(cp)
        zipf_plot(obj,  mode = 'joint', ax = axs[0,1], fit = False, ax_kws = {'label': 'Independent Zipf Bigram'})
        a, b = 2*fit_results.params[0], fit_results.params[1]
        _abs = [1, 150*150]
        axs[0,1].plot([numpy.log(x) for x in _abs], [a +numpy.log(x)*b for x in _abs], '-', label = "joint-zipf")
        axs[0,1].set_xlabel('Log joint rank')
        axs[0,1].set_ylabel('Log Bigram prob.')

        # indep zipf
        import pickle
        with open("Pmatrix.pkl", 'rb') as _file:
            P = pickle.load(_file)
        indep_seq = play_transition_matrix(P, int(1e5))
        ax, fit_results = zipf_plot(indep_seq, mode = 'sequence', ax  = axs[1,0], ax_kws = {'label': 'Dependent Zipf'})
        int_seq, mapping = map_to_ints(indep_seq, sort = True)
        cp = joint_probs(int_seq, distinct = False)
        obj = cp_to_joint_ranks(cp)
        zipf_plot(obj,  mode = 'joint', ax = axs[1,1], fit = False, ax_kws = {'label': 'Dependent Zipf Bigram'})
        a, b = 2*fit_results.params[0], fit_results.params[1]
        _abs = [1, 150*150]
        axs[1,1].plot([numpy.log(x) for x in _abs], [a +numpy.log(x)*b for x in _abs], '-', label = "joint-zipf")
        axs[1,1].set_xlabel('Log joint rank')
        axs[1,1].set_ylabel('Log Bigram prob.')






        # real seq
        real_zipf_seq = COMMAND_SEQUENCES[0]

        
        ax, fit_results = zipf_plot(real_zipf_seq, mode = 'sequence', ax  = axs[2,0], ax_kws = {'label': 'Unix Irvine'})
        int_seq, mapping = map_to_ints(real_zipf_seq, sort = True)
        cp = joint_probs(int_seq, distinct = False)
        obj = cp_to_joint_ranks(cp)
        zipf_plot(obj,  mode = 'joint', ax = axs[2,1], fit = False, ax_kws = {'label': 'Unix Irvine Bigram'})
        ax, rsq = plot_joint_independent_zipf(fit_results_frequency_zipf, x, ax = axs[2,1])


        axs[0,0].legend()
        axs[1,0].legend()
        axs[2,0].legend()

        axs[0,1].legend()
        axs[1,1].legend()
        axs[2,1].legend()
        

        axs[0,1].set_xlim([-0.2,7])
        axs[1,1].set_xlim([-0.2,7])
        axs[2,1].set_xlim([-0.2,7])
        plt.tight_layout(h_pad = -.5, w_pad = -.5)
        plt.show()
        exit()
    
