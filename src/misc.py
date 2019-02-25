from scipy.sparse import csr_matrix, vstack as sparse_vstack 
import os, subprocess, math, itertools, matplotlib, glob
import numpy as np
from collections import deque
from itertools import islice

import matplotlib.pyplot as plt
import pylab, threading
try:
	plt.switch_backend('Qt5Agg')
except:
	plt.switch_backend('agg')

class CustomDictionary:
	
	def __init__(self):
		self.dict = {}

	def initialize(self, key):
		if key not in self.dict:
			self.dict[key] = []

	def add_to_dict(self, key, data):
		if key not in self.dict:
			self.dict[key] = [data]
		else:
			self.dict[key].append(data)
# ---------------------------------------------------------------------------------
def save_as_movie(img_prefix, save_filename, verbose = False):
	total_pngs = len(glob.glob('{}_*.png'.format(img_prefix)))
	frames_per_second = int(total_pngs / 60.) + 1

	if verbose:
		print 'Total configs:', total_pngs, '| Frames per second:', frames_per_second
		print 'Total length of movie:', total_pngs * frames_per_second

	command = "ffmpeg -r {} -i {}_*.png -vcodec mpeg4 -pix_fmt bgra -y {}.mp4".format(frames_per_second,
			  img_prefix, save_filename)

	x = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	x.communicate()
# ---------------------------------------------------------------------------------
def unit_vector(vector):
	return vector / np.linalg.norm(vector)

def angle(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.math.atan2(np.linalg.det([v1_u,v2_u]),np.dot(v1_u,v2_u))

def rotate(theta,v):
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c,-s), (s, c)))
	return np.dot(R,v)
# ---------------------------------------------------------------------------------
def anneal_epsilon(epsilon, epoch, N_episodes):
	return epsilon * (1.0 - (epoch+1) / float(N_episodes))
	#return epsilon * (1.0 / (1 + np.sqrt(epoch)))
# ---------------------------------------------------------------------------------
def chunks_generator(iterable, size=10000):
	iterator = iter(iterable)
	for first in iterator:
		yield list(itertools.chain([first], itertools.islice(iterator, size - 1)))

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()
# ---------------------------------------------------------------------------------
def fig_window(_id = 0, scale_x = 1, scale_y = 1):
	plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'],
	                  'monospace':['Computer Modern Typewriter']})
	fig_width_pt = 2*252  # Get this from LaTeX using \showthe\columnwidth
	inches_per_pt = 1.0/72.27               # Convert pt to inch
	golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
	fig_width = fig_width_pt*inches_per_pt  # width in inches
	fig_height = fig_width*golden_mean      # height in inches
	#fig_size =  [fig_width,fig_height]
	params = {'backend': 'ps',
	          'axes.labelsize': 12,
	          'font.size': 12,
	          'legend.fontsize': 12,
	          'legend.handlelength': 1,
	          'legend.columnspacing': 1,
	          'xtick.labelsize': 12,
	          'ytick.labelsize': 12,
	          #'axes.linewidth': 0.1,
	          'text.usetex': True,
	          'text.latex.preamble': [r"\usepackage{amstext}", r"\usepackage{mathpazo}"],
	          #'xtick.major.pad': 10,
	          #'ytick.major.pad': 10
	    }
	fig = plt.figure(_id, figsize = (scale_x * fig_width, scale_y * fig_height), facecolor='w', edgecolor='k')
	pylab.rcParams.update(params)
	return fig
# ---------------------------------------------------------------------------------     
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
# ---------------------------------------------------------------------------------
def load_sparse_csr(filename):
	loader = np.load(filename)
	return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def save_sparse_tree(array,fid):   
	array = csr_matrix(array) 
	matrix_parameters = [array.data, array.indices, array.indptr, array.shape]
	pickle.dump(matrix_parameters,fid,pickle.HIGHEST_PROTOCOL)
