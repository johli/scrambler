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

def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):

	fp = FontProperties(family="DejaVu Sans", weight="bold")
	globscale = 1.35
	LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
				"G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
				"A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
				"C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
				"UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
				"DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
				"(" : TextPath((-0.25, 0), "(", size=1, prop=fp),
				"." : TextPath((-0.125, 0), "-", size=1, prop=fp),
				")" : TextPath((-0.1, 0), ")", size=1, prop=fp)}
	COLOR_SCHEME = {'G': 'orange', 
					'A': 'red', 
					'C': 'blue', 
					'T': 'darkgreen',
					'UP': 'green', 
					'DN': 'red',
					'(': 'black',
					'.': 'black', 
					')': 'black'}


	text = LETTERS[letter]

	chosen_color = COLOR_SCHEME[letter]
	if color is not None :
		chosen_color = color

	t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
		mpl.transforms.Affine2D().translate(x,y) + ax.transData
	p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
	if ax != None:
		ax.add_artist(p)
	return p

def plot_dna_logo(pwm, sequence_template=None, figsize=(12, 3), logo_height=1.0, plot_start=0, plot_end=164, plot_sequence_template=False, save_figs=False, fig_name=None) :

	#Slice according to seq trim index
	pwm = np.copy(pwm[plot_start: plot_end, :])
	sequence_template = sequence_template[plot_start: plot_end]

	pwm += 0.0001
	for j in range(0, pwm.shape[0]) :
		pwm[j, :] /= np.sum(pwm[j, :])

	entropy = np.zeros(pwm.shape)
	entropy[pwm > 0] = pwm[pwm > 0] * -np.log2(pwm[pwm > 0])
	entropy = np.sum(entropy, axis=1)
	conservation = 2 - entropy

	fig = plt.figure(figsize=figsize)

	ax = plt.gca()

	height_base = (1.0 - logo_height) / 2.

	for j in range(0, pwm.shape[0]) :
		sort_index = np.argsort(pwm[j, :])

		for ii in range(0, 4) :
			i = sort_index[ii]

			nt_prob = pwm[j, i] * conservation[j]

			nt = ''
			if i == 0 :
				nt = 'A'
			elif i == 1 :
				nt = 'C'
			elif i == 2 :
				nt = 'G'
			elif i == 3 :
				nt = 'T'

			color = None
			if sequence_template[j] != 'N' :
				color = 'black'
				if plot_sequence_template and nt == sequence_template[j] :
					nt_prob = 2.0
				else :
					nt_prob = 0.0

			if ii == 0 :
				dna_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax, color=color)
			else :
				prev_prob = np.sum(pwm[j, sort_index[:ii]] * conservation[j]) * logo_height
				dna_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax, color=color)

	plt.xlim((0, plot_end - plot_start))
	plt.ylim((0, 2))
	plt.xticks([], [])
	plt.yticks([], [])
	plt.axis('off')
	plt.axhline(y=0.01 + height_base, color='black', linestyle='-', linewidth=2)

	for axis in fig.axes :
		axis.get_xaxis().set_visible(False)
		axis.get_yaxis().set_visible(False)

	plt.tight_layout()
	
	if save_figs :
		plt.savefig(fig_name + ".png", transparent=True, dpi=300)
		plt.savefig(fig_name + ".eps")

	plt.show()
