import os 
import pandas as pd
import pandas
import numpy as np
import seaborn
import numpy
import matplotlib.pyplot as plt

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
    

def map_to_ints(seq) : 

    all_commands = list(set(seq))
    dictionary = {}
    rev_dictionary = {}
    for index,command in enumerate(all_commands) :
        dictionary[command] = index
        rev_dictionary[index] = command

    mapped = pd.Series(seq).map(dictionary)
    return list(mapped),rev_dictionary

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



# print commands with high transition

# condition = numpy.logical_and(transition_matrix > 0.2, transition_matrix < 0.99)
# rindx, cindx = numpy.where(condition)   

# for r, c, in zip(rindx, cindx):
#     print(f"{mapping[r]} --> { mapping[c]}")
#     print(transition_matrix[r,c])




    


if ZIPF:

    import statsmodels.api as sm

    for user,i in enumerate(COMMAND_SEQUENCES) :
        i = pd.Series(i)
        temp = i.value_counts().to_dict()
        count_dict = {}
        for rank,i in enumerate(temp.keys()):
            count_dict[rank + 1] = temp[i]
        ranks = np.array(list(count_dict.keys()))
        occurances = np.array(list(count_dict.values()))
        frequencies = occurances/numpy.sum(occurances)
        fig, ax = plt.subplots(1,1)
        
        x=np.log(ranks)
        x_fit = sm.add_constant(np.log(ranks))
        fit_results = sm.OLS(np.log(frequencies), x_fit).fit()
        seaborn.regplot(y=np.log(frequencies), x=np.log(ranks), fit_reg = True, label = 'Unix-Irvine', line_kws=dict(color="r", label=f'Zipf fit -- s= {-fit_results.params[1]:.2f}', lw = '2', alpha = .8), ax = ax)
        ax.set_xlabel(r'$\log k$')
        ax.set_ylabel(r'$\log f(s,k)$')
        ax.legend()
        plt.tight_layout()
        plt.show()
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

    def gen_zipf_seq(n_elements,size, s = 1) :

        freqs = numpy.array([i**(-s) for i in range(1, n_elements+1) ])
        probs = freqs/numpy.sum(freqs)
        return numpy.random.choice(n_elements, size = size, replace = True, p = probs)

    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}

    fig1, (ax1,ax2) = plt.subplots(nrows=1, ncols = 2, figsize = (10,5))

    # seaborn.heatmap(matrices[0], ax=ax, square=False, vmin=0, vmax=1, cbar=True,

    #             cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',

    #             cbar_kws={"orientation": "horizontal", "fraction": 0.1,

    #                       "label": "Transition probability"})

    count_matrix[count_matrix > 4] = 4
    count_matrix = count_matrix/5 + 0.2
    plot_transition_matrix(trans_matrix,  count_matrix, ax = ax1, imshow_kwargs = dict(cmap = 'YlOrRd'))
    counts_array = numpy.ones(count_matrix.shape)
    plot_transition_matrix(trans_matrix, counts_array, ax = ax2, imshow_kwargs = dict(cmap = 'YlOrRd'))



    zipf_seq = gen_zipf_seq(trans_matrix.shape[0], len(COMMAND_SEQUENCES[0]), s = 1.75)
    zipf_matrix, count_matrix, mapping = transition_matrix(zipf_seq)

    s = 1.75
    for e in range(zipf_matrix.shape[0]):
        _sum = 0
        for ni,i in enumerate(zipf_matrix[:,e]):
            if ni == 0:
                _sum += i
            else:
                _sum += i*(ni+1)**(-s)
        print(_sum)
    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}

    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize = (10,5))
 
    # shuffle matrix randomly

    def permut(transition_matrix, alpha_array):
        p = numpy.random.permutation(transition_matrix.shape[0])
        return transition_matrix[p], alpha_array[p]

    def tmat_shuffle(transition_matrix, alpha_array):
        t, a = permut(transition_matrix, alpha_array)
        t,a = permut(t.T, a.T)
        return t.T, a.T

    # numpy.random.shuffle(zipf_matrix.T)
    # numpy.random.shuffle(zipf_matrix)
    # seaborn.heatmap(zm, ax=ax, square=True, vmin=0, vmax=1, cbar=True,
    #             cbar_ax=cbar_ax, cmap='binary',  cbar_kws={"orientation": "horizontal", "fraction": 0.1,

    #                     "label": "Transition probability"})

    zipf_matrix, count_matrix = tmat_shuffle(zipf_matrix, count_matrix)



    count_matrix[count_matrix > 4] = 4
    count_matrix = count_matrix/5 + 0.2
    plot_transition_matrix(zipf_matrix,  count_matrix, ax = ax1, imshow_kwargs = dict(cmap = 'YlOrRd'))
    counts_array = numpy.ones(count_matrix.shape)
    plot_transition_matrix(zipf_matrix, counts_array, ax = ax2, imshow_kwargs = dict(cmap = 'YlOrRd'))
    

    plt.tight_layout()

    fig1.savefig('transition_matrix_unix_irvine.pdf')
    fig2.savefig('transition_matrix_pure_zipf.pdf')

    plt.show()

