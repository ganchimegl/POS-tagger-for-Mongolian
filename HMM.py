
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time
import xml.etree.ElementTree as ET
from IPython.display import display
sentences = []

tree = ET.parse('xml/MONC004PRESSTEXTS.xml')
root = tree.getroot()
for sen in root[1][:5000]:
    sentence = []
    for word in sen:
        t = (word.text, word.get('pos'))
        sentence.append(t)
    sentences.append(sentence)

# reading the Treebank tagged sentences
nltk_data = sentences
'''
# download the treebank corpus from nltk
nltk.download('treebank')

# download the universal tagset from nltk
nltk.download('universal_tagset')

# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
'''
# print the first two sentences along with tags
print(nltk_data[:2])

train_set,test_set =train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state = 101)

print(test_set)

train_tagged_words = [ tup for sent in train_set for tup in sent ]
test_tagged_words = [ tup for sent in test_set for tup in sent ]

print(test_tagged_words)
print(len(train_tagged_words))
print(len(test_tagged_words))

tags = {tag for word, tag in train_tagged_words}
print(len(tags))
print(tags)

# check total words in vocabulary
vocab = {word for word, tag in train_tagged_words}
print(vocab)

def word_given_tag(word, tag, train_bag=train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)  # total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    # now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)

# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0] / t2_given_t1(t2, t1)[1]

print(tags_matrix)

tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
display(tags_df)


def Viterbi(words, train_bag=train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['PUN', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0] / word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))


# Let's test our Viterbi algorithm on a few sample sentences of test dataset
random.seed(1234)  # define a random seed to get same sentences when run multiple times

# choose random 10 numbers
rndom = [random.randint(1, len(test_set)) for x in range(100)]
print(rndom, len(test_set))
# list of 10 sents on which we test the model
test_run = [test_set[i] for i in rndom]
#test_run = [test_set[:10]]
# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]
print("test", test_tagged_words)
# Here We will only test 10 sentences to check the accuracy
# as testing the whole training set takes huge amount of time
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end - start

print("Time taken in seconds: ", difference)

# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]

accuracy = len(check) / len(tagged_seq)
print('Viterbi Algorithm Accuracy: ', accuracy * 100)