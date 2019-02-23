import numpy as np, os, psutil, time, traceback
import random, copy, sys, math   
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

def yield_conf(configurations):
	counter = 0 
	for state in configurations:
		yield counter, configurations[state]
		counter += 1 

def worker(config_details, Lx = 20, Ly = 20, verbose = False, draw = False, move_details = False):
	# Will need to at least import Actions here, so keras junk doesn't get confused.	
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

	length = int(sys.argv[1])
	Lx = Ly = 50
	seq = ''.join(['P' for x in range(length)])
	
	if length < 12:
		draw = False
		verbose = False
	else:
		draw = False
		verbose = False

	'''
	Load the text files containing the HP-lattice generated configurations
	'''	
	with open("examples/{}/0.trj".format(length), "r") as fid:
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
	if os.path.isfile('examples/{}/states_trajectory_{}.pkl'.format(length,Lx)):
		os.remove('examples/{}/states_trajectory_{}.pkl'.format(length,Lx))

	######################################################################
	'''
		Set up multiprocessing
	'''

	NCPUS = psutil.cpu_count()
	if NCPUS > 24: NCPUS = 24
	_p = Pool(processes=NCPUS)
	print "Using {} cpus".format(NCPUS)

	available_move_details = available_moves(Lx,Ly)
	f = partial(worker, Lx = Lx, Ly = Ly, verbose = verbose, draw = draw, move_details = available_move_details)

	t0 = time.time()
	
	total = 0
	total_configs = len(configurations.keys())

	size = min(100, total_configs)
	for chunk in chunks_generator(yield_conf(configurations), size = 100 * NCPUS * size):

		for result in _p.imap(f, chunk, chunksize = size):
			
			initial_chain,legal_moves = result 
			
			with open('examples/{}/states_trajectory_{}.pkl'.format(length,Lx), "a+") as fid:
				pickle.dump([Lx,Ly,initial_chain,legal_moves],fid,pickle.HIGHEST_PROTOCOL) 

			if (total + 1) % 100 == 0: 
				print "... working on configuration {} / {}".format(total+1,total_configs)

			total += 1

	if draw:
		name = 'L_{}'.format(len(seq))
		traj_id = '/tmp/{}'.format(name)
		save_as_movie(traj_id, traj_id)
		os.system('mv {}*.png images'.format(traj_id))
		#os.system('rm {}*.png'.format(traj_id))
		os.system('mv {}.mp4 images/{}.mp4'.format(traj_id,name))

	tf = float("%0.3f" % (time.time() - t0))
	print "Finished sequence {} in {} s, total unique states {}".format(seq,tf,total)
