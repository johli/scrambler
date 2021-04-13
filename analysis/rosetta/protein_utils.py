import string
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter

def one_hot_encode_msa(msa, ns=21) :
	
	one_hot = np.zeros((msa.shape[0], msa.shape[1], ns))
	for i in range(msa.shape[0]) :
		for j in range(msa.shape[1]) :
			one_hot[i, j, int(msa[i, j])] = 1.
	
	return one_hot

def parse_a3m(filename):
	seqs = []
	table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

	# read file line by line
	for line in open(filename,"r"):
		# skip labels
		if line[0] != '>':
			# remove lowercase letters and right whitespaces
			seqs.append(line.rstrip().translate(table))

	# convert letters into numbers
	alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
	msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
	for i in range(alphabet.shape[0]):
		msa[msa == alphabet[i]] = i

	# treat all unknown characters as gaps
	msa[msa > 20] = 20

	return msa
