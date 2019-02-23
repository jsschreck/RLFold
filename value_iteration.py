import numpy as np, os, psutil, copy, random, matplotlib, time  
import random, copy, sys, math, itertools  
from scipy.sparse import csr_matrix
import hashlib, cPickle as pickle 

from RLFold.src.trajectory import Trajectory
from RLFold.src.actions import Actions, available_moves
from RLFold.src.episode import Episode
from RLFold.src.chain import Chain
from RLFold.src.moves import Moves 
from RLFold.src.misc import save_as_movie

from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import pylab, traceback
plt.switch_backend('agg')

def yield_from_trajectory(fileName, max_states = 1e10):
	with open(fileName, 'rb') as fid:
		counter = 0 
		while True:
			try:
				yield pickle.load(fid)
				counter += 1 
			except:
				break 
			if counter >= max_states:
				break 

def yield_starting_conf(unique_states, N_total = 1, max_bonds = 1e10):
	total = 0 
	_items_ = [(x,y) for (x,y) in unique_states.items() if y['bonds'] < max_bonds] #unique_states.items()
	while total < N_total:
		random.shuffle(_items_)
		for k, (key,conf_details) in enumerate(_items_):
			Lx = conf_details['Lx']
			Ly = conf_details['Ly']
			initial_chain = conf_details['chain']
			legal_moves = conf_details['moves']
			state = conf_details['state'] 
			yield total, key, Lx, Ly, initial_chain, legal_moves
			total += 1

			if total >= N_total: 
				break 

class Iteration:
	def __init__(self, states, state_action_values, gamma = 1.0, max_bonds = 10, verbose = False):
		self.states = states 
		self.P = state_action_values 
		self.gamma = gamma 
		self.max_bonds = max_bonds
		self.verbose = verbose

	def value_iteration(self, eps = 1e-10, max_iterations = 100000):
		v = {key: 1000.0 for key in self.states.keys() if self.states[key]['bonds'] != self.max_bonds}  # initialize value-function
		q = {}
		for i in range(max_iterations):
			prev_v = copy.deepcopy(v)
			for state_key in sorted(v.keys()):
				actions = np.where(np.isfinite(self.states[state_key]['state'].rewards))[0]
				q[state_key] = {'actions': [], 'rewards': []}
				q_sa = []
				for a in actions:
					reward = self.P[state_key,a]['reward'] 
					next_state = self.P[state_key,a]['next_state']
					probability = self.P[state_key,a]['prob']
					if next_state not in v: # next_state is the winning state. 
						score = reward
					else:
						score = reward + self.gamma * prev_v[next_state]
					q[state_key]['actions'].append(a)
					q[state_key]['rewards'].append(score)
					self.states[state_key]['state'].rewards[a] = score * probability
					q_sa.append(score * probability)
				q_sa = np.array(q_sa)
				selected_action = random.choice(np.where(np.array(q_sa)==max(q_sa))[0])
				v[state_key] = q_sa[selected_action]
			diff = np.array([prev_v[key] - v[key] for key in v])
			if (np.sum(np.fabs(diff)) <= eps):
				if self.verbose:
					print 'Value-iteration converged at iteration %d' %(i+1)
				with open(logger_fileName, "a+") as fid:
					fid.write('Value-iteration converged at iteration %d\n' % (i+1))
				break
		return v, q

	def extract_policy(self, v):
		policy_f = {key: [] for key in v}
		for state_key in sorted(v.keys()):
			actions = np.where(np.isfinite(self.states[state_key]['state'].rewards))[0]
			q_sa = []
			for a in actions:
				reward = self.P[state_key,a]['reward']
				next_state = self.P[state_key,a]['next_state']
				probability = self.P[state_key,a]['prob']
				if next_state not in v: # next_state is the winning state. 
					score = reward
				else:
					score = reward + self.gamma * v[next_state] 
				q_sa.append(score * probability)
			selected_actions = np.where(np.array(q_sa)==max(q_sa))[0]
			for _a in selected_actions:
				policy_f[state_key].append(actions[_a])
		return policy_f

	def compute_policy_v(self, policy, eps = 1e-10):
		v = {key: 0.0 for key in self.states.keys() if self.states[key]['bonds'] != self.max_bonds}
		while True:
			prev_v = copy.deepcopy(v)
			for state_key in sorted(v.keys()):
				action = policy[state_key]
				reward = self.P[state_key,action]['reward'] 
				next_state = self.P[state_key,action]['next_state']
				probability = self.P[state_key,action]['prob']
				if next_state not in v: # next_state is the winning state. 
					score = reward 
				else:
					score = reward + self.gamma * prev_v[next_state] 
				v[state_key] = score * probability
			diff = np.array([prev_v[key] - v[key] for key in v])
			if (np.sum(np.fabs(diff)) <= eps):
				break
		return v

	def policy_iteration(self):
		_policy = {}
		for key in self.states:
			if states[key]['bonds'] == self.max_bonds:
				continue
			actions = np.where(np.isfinite(self.states[key]['state'].rewards))[0]
			selection = random.choice(actions)
			_policy[key] = selection 

		max_iterations = 200000
		for i in range(max_iterations):
			old_policy_v = self.compute_policy_v(_policy)
			new_policy = self.extract_policy(old_policy_v)
			if (np.all(_policy == new_policy)):
				if self.verbose:
					print 'Policy-Iteration converged at step %d' % (i+1)
				break
			_policy = new_policy
		return _policy

	def episode(self, policy, draw = False, total_runs = 1, top = 1, max_episode_length = 1000): 
		total_state_rewards = {}
		trajectories = {}
		N = {key: [0,0] for key in self.states}
		for iteration in range(total_runs):
			for total, key in enumerate(policy.keys()):
				if self.states[key]['bonds'] == self.max_bonds:
					continue
				step = 0
				trajectory = []
				total_reward = 0.0
				starting_state = key
				finished = False
				#print '-------'
				while not finished and step < max_episode_length:
					action = random.choice(policy[key])
					reward = self.P[key,action]['reward']
					next_state = self.P[key,action]['next_state']
					next_bonds = self.states[next_state]['bonds']
					total_reward += self.gamma**step * reward

					#print starting_state, key, action, next_state, reward, total_reward 
					
					N[key][0] += 1
					N[key][1] = self.states[key]['bonds']
					step += 1
					if next_bonds == self.max_bonds:
						N[next_state][0] += 1
						N[next_state][1] = next_bonds
						finished = True
					if draw:
						trajectory.append(key)
					key = next_state
					if finished:
						trajectory.append(key)
				total_state_rewards[starting_state] = [total_reward, step]
				#print "State: {}, N: {}, t*: {}".format(key, total, len(trajectory))
				if total < top:
					trajectories[starting_state] = trajectory

		J = [x[1] for x in total_state_rewards.values()]
		print "Average trajectory length:", np.mean(J)
		print "Shortest trajectory length:", min(J)
		print "Longest trajectory length:", max(J)

		if draw:
			t0 = time.time()
			if self.verbose:
				print "Making movies for the top-{} longest trajectories ... ".format(top)
			T = Trajectory()
			for total,(starting_state,trajectory) in enumerate(sorted(trajectories.items(), key = lambda x: len(x[1]), reverse = True)[:top]):
				N = len(trajectory)
				name = '{}_{}_{}_{}'.format(seq,total,N-1,starting_state)
				for step,key in enumerate(trajectory):
					Lx = self.states[key]['Lx']
					Ly = self.states[key]['Ly']
					initial_chain = self.states[key]['chain']
					C = Chain(seq, Lx = Lx, Ly = Lx)
					C.load_configuration(initial_chain)
					initial_chain = C.chain 
					initial_grid = C.grid
					initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds_f = C.compute_energy()
					string_id = '{}_{}'.format(name,step)
					T.draw(initial_chain, initial_grid, initial_bonds_dict, key, string_id, Lx, Ly)
				traj_id = '/tmp/{}'.format(name)
				save_as_movie(traj_id, traj_id)
				os.system('rm {}*.png'.format(traj_id))
				os.system('mv {}.mp4 images/{}/{}.mp4'.format(traj_id,length,name))
			if self.verbose:
				print "... finished in {} s".format(time.time()-t0)
			
		ave_value = np.mean([x[0] for x in total_state_rewards.values()])
		traj_length = np.mean([x[1] for x in total_state_rewards.values()])
		return total_state_rewards, ave_value, traj_length, N 
	 
if __name__ == "__main__":

	seq = str(sys.argv[1])
	length = len(seq)

	#N = length + 2      # grid size
	N = 50
	topK = 25
	gamma = 0.99
	max_episode_length = 1000

	draw = False
	verbose = True
	use_neural_model = False

	'''
		Set up filenames
	'''
	load_states_fileName  = 'examples/{}/data/states_trajectory_{}.pkl'.format(length,N)
	save_states_fileName  = 'examples/{}/states/states_{}.pkl'.format(length,N)
	save_optimal_fileName = 'examples/{}/states/{}.pkl'.format(length,seq)
	save_episode_fileName = 'examples/{}/episodes/{}.pkl'.format(length,seq)
	logger_fileName 	  = 'logs/logger_{}.txt'.format(seq)
	save_training_states  = "fps/all_states_{}.pkl".format(length)
	save_testing_states   = "fps/test_{}.pkl".format(length)
	
	'''
		Delete pre-existing files
	'''
	for _fn in [logger_fileName, save_optimal_fileName, save_episode_fileName]:
		if os.path.isfile(_fn):
			os.system('rm {}'.format(_fn))

	with open(logger_fileName, "a+") as fid:
		fid.write('Working on sequence {}\n'.format(seq))
	
	'''
		Load from trajectory file containing unique states for length N
	'''
	t_start = time.time()
	if not os.path.isfile(load_states_fileName):
		print "You must load a trajectory file: state number, configuration, allowed moves"
		sys.exit(1)
	else:
		max_bonds = 0.0 
		max_states = 1e10

		# Policy pi(state, action) returns (reward, next_state)
		P = {}
		unique_states = {}
		states_generator = yield_from_trajectory(load_states_fileName)
		for q, conf_details in enumerate(states_generator):
			Lx, Ly, saved_chain, legal_moves = conf_details

			Lx = N 
			Ly = N 
			# Get move details for this lattice
			if not q:
				available_move_details = available_moves(Lx,Ly)
				print "Possible moves:", available_move_details[0]

			T = Trajectory()
			A = Actions(Lx = Lx, Ly = Ly, available = available_move_details, use_neural_model = use_neural_model)
			C = Chain(seq, Lx = Lx, Ly = Ly)
			
			C.load_configuration(saved_chain)
			initial_chain = C.chain
			initial_grid = C.grid
			particle_directions = C.particle_directions()
			key = "".join(str(x) for x in particle_directions)
			
			initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds = C.compute_energy()
			state = A.initialize_state(initial_grid, initial_bonds_grid, initial_bonds_dict)

			# Throw out not-moves.
			_legal_moves = {}
			for p,moves in legal_moves.items():
				for move,this_state,next_state in zip(moves['moves'],moves['this_state'],moves['next_state']):
					r0, r1 = move
					if r0 != r1:
						if p not in _legal_moves:
							_legal_moves[p] = {'moves': [], 'weights': [], 'this_state': [], 'next_state': []}
						_legal_moves[p]['moves'].append(move)
						_legal_moves[p]['this_state'].append(this_state)
						_legal_moves[p]['next_state'].append(next_state)
						action_vector_index = A.decoder[r0,r1]
						# Assign random values for legal state-actions
						state.rewards[action_vector_index] = 0.0
						P[(this_state,action_vector_index)] = {'reward': 0.0, 'next_state': next_state, 'prob': 1.0}

			legal_moves = _legal_moves
			
			unique_states[key] = {'Lx': Lx, 'Ly': Ly, 
								  'chain': saved_chain, 
								  'moves': legal_moves, 
								  'state': state,
								  'bonds': float(N_bonds),
								  'visit_count': 0,
								  'current_value': 0,
								  'converged': False}
			max_bonds = max(max_bonds, float(N_bonds))
	
	'''
		Exit if there are no states with H-bonds.
	'''
	if not max_bonds:
		print "{} cannot form any bonds. Exiting.".format(seq)
		sys.exit()

	'''
		Add to pi(s|a) (reward, next_state) details 
	'''
	for key in unique_states:
		rewards = unique_states[key]['state'].rewards
		actions = [_x for _x in P if _x[0] == key]
		if unique_states[key]['bonds'] == max_bonds:
			rewards[np.where(np.isfinite(rewards))] = 0.0
			for state_action in actions: 
				# Set max_bonded states to have no next_state.
				P[state_action]['reward'] = max_bonds
				P[state_action]['next_state'] = None
				P[state_action]['prob'] = 1.0
		else:
			this_bonds = unique_states[key]['bonds']
			z_sa = []
			for state_action in actions:
				next_state = P[state_action]['next_state']
				next_bonds = unique_states[next_state]['bonds']
				
				if next_bonds == max_bonds:
					P[state_action]['reward'] = max_episode_length * max_bonds
				else:
					P[state_action]['reward'] = next_bonds
					#if next_bonds > this_bonds:
					#	P[state_action]['reward'] = next_bonds
					#else:
					#	P[state_action]['reward'] = next_bonds
				P[state_action]['prob'] = np.exp(next_bonds-this_bonds)
				z_sa.append(P[state_action]['prob'])
				
	'''
		Find the optimal value network and the optimal policy
	'''
	env = Iteration(unique_states, P, gamma = gamma, max_bonds = max_bonds, verbose = verbose)
	t0 = time.time()
	v_star, q_star = env.value_iteration()
	optimal_policy = env.extract_policy(v_star)
	tf = time.time()

	'''
		Save training states
	'''
	train_total = 0 
	test_total  = 0 
	with open(save_training_states, "a+") as fid:
		with open(save_testing_states, "a+") as gid:
			for state in unique_states:
				State = unique_states[state]
				if State['bonds'] == max_bonds: continue

				if random.random() >= 0.0:
					State['state'].save_sparse(fid)
					train_total += 1 
				else:
					State['state'].save_sparse(gid)
					test_total += 1
	print "Saved {} train states to {}".format(train_total, save_training_states)
	print "Saved {} test states to {}".format(test_total, save_testing_states)
	sys.exit()

	#for key in unique_states:
	#	try:
	#		_r = unique_states[key]['state'].rewards
	#		print key, v_star[key], np.where(np.isfinite(_r)), _r[np.where(np.isfinite(_r))]
	#	except Exception as E:
	#		pass
	'''
	A = Actions(Lx = N, Ly = N, available = available_move_details, use_neural_model = True)
	A.states = unique_states
	A.train(0, 16, 100, split = 0.2, max_bonds = max_bonds)
	sys.exit()
	'''

	if verbose:
		for key,value in sorted(v_star.items(), key = lambda x: x[1], reverse = True):
			print key, value
		print "Average state value:", np.mean(v_star.values())
		#for key,value in sorted(q_star.items(), key = lambda x: x[1], reverse = True):
		#	print key, value['rewards']
			#print "Average state-action value:", np.mean(q_star['rewards'].values())
	
	'''
		Write stats to logger
	'''
	with open(logger_fileName, "a+") as fid:
		if verbose:
			_t_ = float("%0.3f" % (tf - t0))
			print '... which took {} s'.format(_t_)
		fid.write('... which took {} s\n'.format(_t_))

	'''
		Run 1 episode for all of the unique states using the optimal policy
	'''
	results = env.episode(optimal_policy, draw = draw, total_runs = 1000, top = topK, max_episode_length = max_episode_length)
	total_state_rewards, ave_value, traj_length, N = results
	results = sorted(total_state_rewards.items(), key = lambda x: (x[1][0],x[1][1]), reverse = True)

	'''
		Write episode results to logger
	'''
	with open(logger_fileName, "a+") as fid:
		for key, result in results:
			#if verbose:
			#	print key, result[0], result[1]
			fid.write('{} {} {}\n'.format(key, result[0], result[1]))
		#if verbose:
			#print "Average state value:", ave_value
			#print "Average trajectory length:", traj_length 
		fid.write("Average state value: {}\n".format(ave_value))
		fid.write("Average trajectory length: {}\n".format(traj_length))

	'''
		Dump unique_states to pickle. 
	'''
	with open(save_optimal_fileName, "wb") as gid:
		with open(save_states_fileName, "wb") as fid:
			for state in unique_states:
				if state in v_star:
					pickle.dump([v_star[state],q_star[state],optimal_policy[state]],gid,pickle.HIGHEST_PROTOCOL)
				#if 'P' not in seq: 
					# Only need to save this once, so only save when HHHH ... HH
				pickle.dump([state,unique_states[state]],fid,pickle.HIGHEST_PROTOCOL)
	
	with open(logger_fileName, "a+") as fid:
		fid.write("Saved state-action dictionary to {}\n".format(save_states_fileName))
		fid.write("Saved optimal results to {}\n".format(save_optimal_fileName))
		