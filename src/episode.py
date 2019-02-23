from RLFold.src.trajectory import Trajectory
from RLFold.src.actions import Actions, available_moves
from RLFold.src.chain import Chain
from RLFold.src.moves import Moves 

import numpy as np, os
import cPickle as pickle 

class Episode:

	def __init__(self, seq, unique_states, available_move_details = None, epsilon = 0.0, neural = False, verbose = True, draw = False):

		self.seq = seq
		self.epsilon = epsilon
		self.verbose = verbose
		self.neural  = neural 
		self.draw    = draw

		self.unique_states = unique_states
		self.episode = {}
		self.t = 0 

		if not available_move_details:
			one_key = unique_states.keys()[0]
			self.available = available_moves(unique_states[one_key]['Lx'],unique_states[one_key]['Ly'])
		else:
			self.available = available_move_details

	def Step(self, result, max_bonds, is_final_action = False):
		episode_no, key, Lx, Ly, initial_chain, legal_moves = result
		state = self.unique_states[key]['state'] 
		
		T = Trajectory()
		A = Actions(Lx = Lx, Ly = Lx, available = self.available, use_neural_model = self.neural)
		C = Chain(self.seq)
		moves = Moves(C)

		C.load_configuration(initial_chain)
		initial_chain = C.chain
		initial_grid = C.grid 
		initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds_0 = C.compute_energy()

		value = np.sum(state.rewards[state.rewards>-1])
		string_id = 'C_{}_{}'.format(episode_no,self.t)
		
		if self.draw:
			T.draw(initial_chain, initial_grid, initial_bonds_dict, value, string_id)

		'''
		Select and carry out move
		'''
		action_idx, particle, action, expected_reward = A.select_move(state, 
																	  legal_moves, 
																	  epsilon = self.epsilon)
		config_reward, next_chain, next_grid = moves.perform_one_move(particle, action)
		
		'''
		Re-initialize chain object with next config.
		'''
		T = Trajectory()
		A = Actions(Lx = Lx, Ly = Lx, available = self.available, use_neural_model = self.neural)
		C = Chain(self.seq)
		moves = Moves(C)
		
		C.load_configuration(next_chain)
		initial_chain = C.chain 
		initial_grid = C.grid
		initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds_f = C.compute_energy()

		'''
		Update state rewards
		'''

		#excess_bonds = N_bonds_f - N_bonds_0
		#if excess_bonds > 0:
		#	step_reward = 0
		#	if N_bonds_f == max_bonds:
		#		step_reward = max_bonds
		#elif excess_bonds == 0:
		#	step_reward = -0.1
		#else:
		#	step_reward = -1 

		if (N_bonds_f == max_bonds):
			step_reward = 1.0
		#elif is_final_action:
		#	step_reward = -1.0
		else:
			step_reward = -0.01

		#step_reward = N_bonds_f - N_bonds_0
		self.episode[self.t] = {'key': key,
					  			'state': state, 
					  			'action': action_idx, 
					  			'reward': step_reward}

		if self.verbose:
			print episode_no, self.t+1, particle, action, float("%0.3f"%step_reward), N_bonds_f
		 
		#try:
		#	'''
		#		Get details about the next state from the parent dictionary
		#	'''
		#	particle_directions = C.particle_directions()
		#	key = "".join(str(x) for x in particle_directions)
		#	#key = np.array_str(np.hstack([initial_grid,initial_bonds_grid]))
		#	legal_moves = self.unique_states[key]['moves']
		#	#state = self.unique_states[key]['state']
		
		#except Exception as E:
		#	print "Exception", E 
		#	print particle, action
		#	print key
		#	print self.unique_states.keys()
		#	return False

		particle_directions = C.particle_directions()
		key = "".join(str(x) for x in particle_directions)

		if N_bonds_f == max_bonds:
			state = self.unique_states[key]['state'] 
			self.episode[self.t+1] = {'key': key,
					  				  'state': state, 
					  				  'action': None, 
					  				  'reward': None}
		"""
			Stop the code if we reach a state not in the dict, prob bug in chain.canonicalize()
			Note that this won't work once we simulate large chains, for which we do not know 
			all of the possible states. 
		"""
		assert (key in self.unique_states),"Key not in states dictionary"

		legal_moves = self.unique_states[key]['moves']

		# Stop the code if we reach target state.
		assert (N_bonds_f != max_bonds),"Reached end of game"

		self.t += 1 
		return episode_no, key, Lx, Ly, initial_chain, legal_moves

	def Run(self, result, N_steps, max_bonds):

		if not result:
			return self.episode

		try:
			is_final_action = False
			for k in range(N_steps):
				if k == (N_steps-1): 
					is_final_action = True
				result = self.Step(result, max_bonds, is_final_action = is_final_action)
				
		except AssertionError as A:
			if A.args == 'Reached end of game':
				pass 
			if A.args == 'Key not in states dictionary':
				return False
		
		return self.episode

	def Save(self, fileName):
		with open(fileName, "a+") as fid:
			pickle.dump(self.episode,fid,pickle.HIGHEST_PROTOCOL)

