import numpy as np, os, psutil, time, traceback
import random, copy, sys, math, argparse   
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import hashlib, cPickle as pickle 

from RLFold.src.trajectory import Trajectory
from RLFold.src.actions import Actions, available_moves
from RLFold.src.chain import Chain
from RLFold.src.moves import Moves 
from RLFold.src.misc import save_as_movie, chunks_generator, chunks

from multiprocessing import Pool
from functools import partial

import matplotlib
matplotlib.pyplot.switch_backend('agg')

def yield_conf(configurations):
	counter = 0 
	for state in configurations:
		yield counter, configurations[state]
		counter += 1 

def worker(config_details, Lx = 20, Ly = 20, verbose = False, draw = False, move_details = False):	
	try:
		p, config = config_details

		T = Trajectory()
		A = Actions(Lx = Lx, Ly = Lx, use_neural_model = False, available = move_details)
		C = Chain(seq, Lx = Lx, Ly = Lx)

		C.load_configuration_from_coors(config)
		initial_chain, initial_grid = C.chain, C.grid
		initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds = C.compute_energy()
		particle_directions = C.particle_directions()
		key = "".join(str(x) for x in particle_directions)

		if draw:
			file_str = 'L_{}_{}'.format(len(seq),key)
			T.draw(initial_chain, initial_grid, initial_bonds_dict, key, file_str, Lx, Ly)

		moves = Moves(C)
		moves_dict = moves.available()
		legal_moves = {particle: {'moves': [], 'weights': [], 'this_state': [], 'next_state': []} for particle in moves_dict}

		for particle in moves_dict:
			particle_moves = moves_dict[particle]['moves']
			
			for move in particle_moves:
				# If move is the not-move, save and continue
				if (move[0] == move[1]) and move not in legal_moves[particle]['moves']:
					continue

					legal_moves[particle]['moves'].append(move)
					legal_moves[particle]['this_state'].append(key)
					legal_moves[particle]['next_state'].append(key)
					if verbose:
						print key, p, particle, move, expected_reward, 0.1, key

				# Perform move and check the reward for legality.
				else:
					_moves_dict_ = {particle: {'moves': [move], 'weights': []}}
					state = A.initialize_state(initial_grid, initial_bonds_grid, None)
					action_idx, particle, action, expected_reward = A.select_move(state, 
																				  _moves_dict_, 
																				  epsilon = 0.0)
					try:
						reward, next_chain, next_grid = moves.perform_one_move(particle, action)
						particle_directions = C.particle_directions()
						next_state = "".join(str(x) for x in particle_directions)
						if reward >= 0.0:
							if move not in legal_moves[particle]['moves']:
								legal_moves[particle]['moves'].append(move)
								legal_moves[particle]['this_state'].append(key)
								legal_moves[particle]['next_state'].append(next_state)
						
								if verbose:
									print key, p, particle, action, expected_reward, reward, next_state
					
					except Exception as E:
						if E.args[0] == 'You missed a direction':
							#print E, p, particle, action
							pass		
						else:
							print "Failing ... ", E	

				'''
				Re-initialize chain object
				'''
				T = Trajectory()
				A = Actions(Lx = Lx, Ly = Lx, use_neural_model = False, available = move_details)
				C = Chain(seq, Lx = Lx, Ly = Lx)
				
				C.load_configuration_from_coors(config)
				initial_chain, initial_grid = C.chain, C.grid
				initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds = C.compute_energy()
				moves = Moves(C)

				particle_directions = C.particle_directions()
				key = "".join(str(x) for x in particle_directions)

		return initial_chain, legal_moves

	except:
		print traceback.format_exc()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--grid_size', type = int, default = 50,
						help = 'Length of the grid, default is 1 + length')
	parser.add_argument('--chain_length', type = int, default = 5,
						help = 'Length of the chain. Default is 5')
	parser.add_argument('--cores', type = int, default = psutil.cpu_count(),
						help = 'Number of cores. Default is {}'.format(psutil.cpu_count()))
	parser.add_argument('--draw', type = bool, default = False,
						help = 'Write conf to png. Default is false.')
	parser.add_argument('--verbose', type = bool, default = False,
						help = 'Print progress to stdout.')
	parser.add_argument('--conf_directory', type = str, default = 'examples',
						help = 'Load configurations from here. Default = ./examples')
	parser.add_argument('--save_directory', type = str, default = 'images',
						help = 'Where to save images. Default = ./images')
	args = parser.parse_args()

	chain_length	= int(args.chain_length)
	grid_size		= int(args.grid_size)
	cores			= int(args.cores)
	draw			= bool(args.draw)
	verbose 		= bool(args.verbose)
	conf_dir 		= str(args.conf_directory)
	save_dir 		= str(args.save_directory)

	'''
	There needs to be a directory already containing a .trj file. 
	'''
	if not os.path.isdir(conf_dir):
		print "No directory called {}. Exiting.".format(conf_dir)


	Lx = Ly = grid_size
	seq = ''.join(['P' for x in range(chain_length)])
	
	'''
	Load the text files containing the HP-lattice generated configurations
	'''	
	traj_file = "{}/0.trj".format(chain_length)
	traj_location = os.path.join(conf_dir, traj_file)
	
	if not os.path.isfile(traj_location):
		print "You need to run make_configs.py first. Exiting."
		sys.exit(1)
	
	with open(traj_location, "r") as fid:
		configurations = {}
		for k,line in enumerate(fid.readlines()):
			line = line.strip('\n').strip('[').strip(']').split("), ")
			line = [l.strip('(').strip(')').split(', ') for l in line]
			line = [[int(x[0]),int(x[1])] for x in line]
			configurations[k] = line 
	N_states = len(configurations)

	'''
		Delete any pre-existing trajectory file
	'''
	formatter = '{}/states_trajectory_{}.pkl'.format(chain_length, grid_size)
	traj_file = os.path.join(conf_dir, formatter)
	if os.path.isfile(traj_file):
		os.remove(traj_file)

	######################################################################
	'''
		Set up multiprocessing
	'''
	if cores > 24: cores = 24
	_p = Pool(processes=cores)
	print "Using {} cpus".format(cores)

	available_move_details = available_moves(Lx,Ly)
	f = partial(worker, Lx = Lx, Ly = Ly, verbose = verbose, 
				draw = draw, move_details = available_move_details)

	t0 = time.time()

	'''
		Find all possible "legal" configurations
	'''
	total = 0
	total_configs = len(configurations.keys())
	size = 100 * cores * min(100, total_configs)
	for chunk in chunks_generator(yield_conf(configurations), size = size):
		for result in _p.imap(f, chunk, chunksize = size):
			
			initial_chain,legal_moves = result 
			
			with open(traj_file, "a+") as fid:
				pickle.dump([Lx,Ly,initial_chain,legal_moves],
							fid,pickle.HIGHEST_PROTOCOL
							) 

			if (total + 1) % 100 == 0: 
				print "... working on configuration {} / {}".format(total+1,total_configs)

			total += 1

	'''
	Save configurations to png, stitch together to make a movie.
	'''
	if draw:
		name = 'L_{}'.format(len(seq))
		movie_path = os.path.join(save_dir, str(chain_length))
		traj_id = os.path.join(movie_path, name)

		for _dir in [save_dir, movie_path]:
			if not os.path.isdir(_dir):
				os.makedirs(_dir)

		save_as_movie(traj_id, traj_id, verbose = verbose)

	tf = float("%0.3f" % (time.time() - t0))
	print "Finished sequence {} in {} s, total unique states {}".format(seq,tf,total)
