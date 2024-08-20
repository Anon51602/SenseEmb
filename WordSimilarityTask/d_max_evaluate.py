#!/usr/bin/env python3
#
# Copyright (c) 2017-present, All rights reserved.
# Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
#
# This file is part of Dict2vec.
#
# Dict2vec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Dict2vec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at the root of this repository for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with Dict2vec.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import argparse
import numpy as np
import scipy.stats as st

FILE_DIR = "./eval/"
results      = dict()
missed_pairs = dict()
missed_words = dict()


def embDot(word_emb, w1, w2):
    """Return the Tanimoto similarity between v1 and v2 (numpy arrays)"""
    v1 = word_emb[w1]
    v2 = word_emb[w2]
    similarity = np.dot(v1, v2.T)  # / (np.linalg.norm(color1_centers) * np.linalg.norm(color2_centers))
    return np.max(similarity)


def embCos(word_emb, w1, w2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    v1 = word_emb[w1]
    v2 = word_emb[w2]
    similarity = np.dot(v1, v2.T)   / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.max(similarity)
def init_results():
    """Read the filename for each file in the evaluation directory"""
    for filename in os.listdir(FILE_DIR):
        if not filename in results:
            results[filename] = []


def evaluate(filename,method="dot"):
    """Compute Spearman rank coefficient for each evaluation file"""
    # Emb file
    # step 0 : read the first line to get the number of words and the dimension
    import pickle
    # with open('./all_hidden_mc30.pkl', 'rb') as handle:
    with open(filename, 'rb') as handle:
        word_emb = pickle.load(handle)

    word_emb = {key.replace(" ", "_") if " " in key else key: value for key, value in word_emb.items()}
   
    # step 1 : iterate over each evaluation data file and compute spearman
    for filename in results:
        pairs_not_found, total_pairs = 0, 0
        words_not_found, total_words = 0, 0
        with open(os.path.join(FILE_DIR, filename)) as f:
            file_similarity = []
            embedding_similarity = []
            for line in f:
                w1, w2, val = line.split()
                w1, w2, val = w1.lower(), w2.lower(), float(val)
                total_words += 2
                total_pairs += 1
                if not w1 in word_emb:
                    words_not_found += 1
                if not w2 in word_emb:
                    words_not_found += 1

                if not w1 in word_emb or not w2 in word_emb:
                    pairs_not_found += 1
                else:
                    if method == "dot":

                        cosine = embDot(word_emb, w1, w2)
                    elif method == "cos":
                        cosine = embCos(word_emb, w1, w2)
                    file_similarity.append(val)
                    embedding_similarity.append(cosine)


            rho, p_val = st.spearmanr(file_similarity, embedding_similarity)
            results[filename].append(rho)
            missed_pairs[filename] = (pairs_not_found, total_pairs)
            missed_words[filename] = (words_not_found, total_words)


def stats():
    """Compute statistics on results"""
    title = "{}| {}| {}| {}| {}| {} ".format("Filename".ljust(16),
                              "AVG".ljust(5), "MIN".ljust(5), "MAX".ljust(5),
                              "STD".ljust(5), "Missed words/pairs")
    print(title)
    print("="*len(title))

    weighted_avg = 0
    total_found  = 0

    for filename in sorted(results.keys()):
        average = sum(results[filename]) / float(len(results[filename]))
        minimum = min(results[filename])
        maximum = max(results[filename])
        std = sum([(results[filename][i] - average)**2 for i in
                   range(len(results[filename]))])
        std /= float(len(results[filename]))
        std = math.sqrt(std)

        # For the weighted average, each file has a weight proportional to the
        # number of pairs on which it has been evaluated.
        # pairs evaluated = pairs_found = total_pairs - number of missed pairs
        pairs_found = missed_pairs[filename][1] - missed_pairs[filename][0]
        weighted_avg += pairs_found * average
        total_found  += pairs_found

        # ratio = number of missed / total
        ratio_words = missed_words[filename][0] / missed_words[filename][1]
        ratio_pairs = missed_pairs[filename][0] / missed_pairs[filename][1]
        missed_infos = "{:.0f}% / {:.0f}%".format(
                round(ratio_words*100), round(ratio_pairs*100))

        print("{}| {:.3f}| {:.3f}| {:.3f}| {:.3f}| {} ".format(
              filename.ljust(16),
              average, minimum, maximum, std, missed_infos.center(20)))

    print("-"*len(title))
    print("{0}| {1:.3f}".format("W.Average".ljust(16),
                                weighted_avg / total_found))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
             description="Evaluate semantic similarities of word embeddings.",
             )

    #parser.add_argument('filenames', metavar='FILE', nargs='+',
    #                    help='Filename of word embedding to evaluate.')

    args = parser.parse_args()
    filenames =  ["./wiki.pkl"]

    for  f in filenames:
        print(f)

        results      = dict()
        missed_pairs = dict()
        missed_words = dict()
        init_results()
        
        evaluate(f,"dot")
        stats()

