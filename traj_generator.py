import numpy as np, os, psutil, copy, random, matplotlib, time  
import random, copy, sys, math, itertools  
from scipy.sparse import csr_matrix
import hashlib, cPickle as pickle 

from RLFold.src.trajectory import Trajectory
from RLFold.src.actions import Actions, available_moves
from RLFold.src.episode import Episode
from RLFold.src.chain import Chain
from RLFold.src.moves import Moves 
from RLFold.src.misc import save_as_movie, anneal_epsilon, fig_window, threadsafe_iter, chunks_generator, running_mean

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
	
def anneal_epsilon(epsilon, epoch, N_epochs = 10):
	N_at_zero = 1 #np.floor(self.N_epochs)
	increment = min(1, epoch / float(N_epochs-N_at_zero))
	current_epsilon = epsilon * (1.0 - increment)
	return current_epsilon

def work(data, env = None):
	try:
		return env.worker(data)
	except:
		print traceback.format_exc()

class Simulate:
	def __init__(self, seq, states, available_moves = None, epsilon = 0.0, max_bonds = 1, N_time_steps = 10, N_epochs = 10, verbose = False, draw = False, save_episodes = False):
		self.seq = seq 
		self.states = states 
		self.epsilon = epsilon
		self.epsilon_decay = []

		self.max_bonds = max_bonds
		self.N_time_steps = N_time_steps 
		self.N_epochs = N_epochs
		self.episodes_per_epoch = len([x for (x,y) in states.items() if y['bonds'] < max_bonds])

		self.draw = draw 
		self.verbose = verbose
		self.save_episodes = save_episodes

		# Monitor t -> t+1 values, stop when converged.
		self.current_values = {key: {'value': [], 'visits': [], 'action_values': [], 'episode_lengths': [], 'last_episode': None} for key in self.states}  
		 
		# Get moves and encoder / decoder so we do not keep calling everytme Action is called.
		if not available_moves:
			one_key = states.keys()[0]
			self.available_move_details = available_moves(states[one_key]['Lx'],states[one_key]['Ly'])
			self.all_possible_moves, self.encoder, self.decoder = self.available_move_details
		else:
			self.all_possible_moves, self.encoder, self.decoder = available_moves

	def anneal_epsilon(self, epsilon, epoch, N_epochs = 100):
		# N_at_zero specifies how many sims to run at epsilon = 0.0
		N_at_zero = 1 #np.floor(self.N_epochs)
		increment = min(1, epoch / float(self.N_epochs-N_at_zero))
		self.current_epsilon = epsilon * (1.0 - increment)
		return self.current_epsilon

	def worker(self, results):
		episode_no = results[0]
		epoch = math.floor(episode_no / self.episodes_per_epoch)
		_epsilon = self.epsilon #self.anneal_epsilon(self.epsilon, epoch, self.N_epochs)
		E = Episode(self.seq, self.states, epsilon = _epsilon, verbose = self.verbose, draw = self.draw)
		episode = E.Run(results, self.N_time_steps, self.max_bonds)
		
		# Save episode to buffer - must restrict size, thing grows fast.
		if self.save_episodes:
			E.Save(save_episode_fileName)

		return episode_no, _epsilon, episode
 
if __name__ == "__main__":

	seq = str(sys.argv[1])
	length = len(seq)

	t_start = time.time()

	print "Train Q(s,a) for sequence {}".format(seq)

	#all_combos = map(list, itertools.product(['H', 'P'], repeat=length))

	use_neural_model = False
	load_states_fileName  = 'examples/{}/data/states_trajectory.pkl'.format(length)
	save_states_fileName  = 'examples/{}/states/{}.pkl'.format(length,seq)
	save_episode_fileName = 'examples/{}/episodes/{}.pkl'.format(length,seq)
	logger_fileName 	  = 'logs/logger_{}.txt'.format(seq)
	
	for _fn in [logger_fileName, save_episode_fileName]:
		if os.path.isfile(_fn):
			os.system('rm {}'.format(_fn))

	with open(logger_fileName, "a+") as fid:
		fid.write('sequence {}\n'.format(seq))

	'''
	Load from trajectory file containing unique states for length N
	'''
	if not os.path.isfile(load_states_fileName):
		print "You must load a trajectory file: state number, configuration, allowed moves"
	else:
		max_bonds = 0.0 
		max_states = 1e10

		# Policy takes (state, action) as input, returns (reward, next_state)
		policy = {}

		unique_states = {}
		states_generator = yield_from_trajectory(load_states_fileName)
		for q, conf_details in enumerate(states_generator):
			Lx, Ly, saved_chain, legal_moves = conf_details

			# Get move details for this lattice
			if not q:
				available_move_details = available_moves(Lx,Ly)

			T = Trajectory()
			A = Actions(Lx = Lx, Ly = Lx, available = available_move_details, use_neural_model = use_neural_model)
			C = Chain(seq, Lx = Lx, Ly = Lx)
			
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
						policy[(this_state,action_vector_index)] = {'reward': 0.0, 'next_state': next_state}

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
	
	if not max_bonds:
		print "{} cannot form any bonds. Exiting.".format(seq)
		sys.exit()

	for key in unique_states:
		_rewards = unique_states[key]['state'].rewards
		_actions = [_x for _x in policy if _x[0] == key]
		if unique_states[key]['bonds'] == max_bonds:
			_rewards[np.where(np.isfinite(_rewards))] = 0.0
			for state_action in _actions: # Set max_bonded states to have no next_state.
				policy[state_action]['reward'] = 1.0
				policy[state_action]['next_state'] = None
				P[state_action]['prob'] = 1.0
		else:
			for state_action in _actions:
				next_state = policy[state_action]['next_state']
				next_bonds = unique_states[next_state]['bonds']
				if next_bonds == max_bonds:
					policy[state_action]['reward'] = 1.0
					policy[state_action]['next_state'] = None
				else:
					policy[state_action]['reward'] = -1.0
				P[state_action]['prob'] = np.exp(next_bonds-this_bonds)

	# Make a deep copy to use for the last simulation to get epsilon = 0.0 results.
	initial_unique_states = copy.deepcopy(unique_states)

	# Set up mp.Pool
	NCPUS = psutil.cpu_count()
	if NCPUS > 24: NCPUS = 24
	p = Pool(processes=NCPUS)
	print "Using {} CPUs".format(NCPUS)

	########################################

	###############
	#
	# A. Decaying greedy-epsilon descent to optimal policy
	#
	###############

	draw = False
	verbose = True

	print_every = 100

	N_states = len([x for (x,y) in unique_states.items() if y['bonds'] < max_bonds])
	N_episodes = 1000 * N_states
	N_visit_min = 100
	N_time_steps = 1000
	strp = 'unique_states {}\nN_episodes {}\nN_steps {}\n'.format(N_states,N_episodes,N_time_steps)

	# This controls the size of the output printed to logger file.
	write_logger = True
	if N_states > 1000:
		write_logger = False

	# Anneal noise
	starting_epsilon = 1.0
	N_epsilons = 4
	epsilons = [anneal_epsilon(starting_epsilon, _iter, N_epsilons) for _iter in range(N_epsilons)]

	# Q-table hyper-parameters
	alpha_0 = 1.0
	alpha_f = 1.0
	alphas  = np.linspace(alpha_0, alpha_f, len(epsilons))

	gamma = 0.8

	convergence_cutoff = 1e-3
	
	epoch = 0

	episode_sizes = []

	# Set up class for use in simplifying mp.
	env = Simulate(seq, 
				   unique_states, 
				   available_moves = available_move_details,
				   epsilon = epsilons[0], 
				   max_bonds = max_bonds, 
				   N_time_steps = N_time_steps,
				   N_epochs = N_episodes,
				   verbose = False, 
				   draw = draw)

	t0 = time.time()

	for ep_iteration, epsilon in enumerate(epsilons):

		epoch_times = []

		env.epsilon = epsilon

		# Q-table update information 
		strp = 'alpha_0 {}\nalpha_f {}\ngamma {}\nepsilon {}\n'.format(alpha_0,alpha_f,gamma,epsilon)

		# Print hyper-parameters to file
		with open(logger_fileName, "a+") as fid:
			fid.write("{}".format(strp))

		# Encourage faster convergence after initial exploration round. 
		if ep_iteration: 
			N_visit_min = 10

		# Reset / initialize stats metrics
		for key in unique_states:
			this_state = env.current_values[key]
			this_state['visits'] = 0
			#this_state['action_values'] = []
			#this_state['value'] = []

		# Initialize starting conf generator: Watch out --> f(dictionary)
		yield_conf = yield_starting_conf(unique_states, N_total = N_episodes, max_bonds = max_bonds)

		# Set up multiple workers 
		p = Pool(processes=NCPUS)

		# Initialize worker function
		f = partial(work, env = env)

		is_converged = False
		while not is_converged:

			for chunk in chunks_generator(yield_conf, size = 1000):
				
				for (episode_no, _epsilon, episode) in p.imap(f, chunk, chunksize = 100):
					
					# Epoch updated if a pass over unique states just completed.
					print_verbose = False
					if episode_no % N_states == 0 and episode_no > 1:
						epoch += 1
						epoch_times.append(time.time()-t0)
						t0 = time.time()

						if epoch % print_every == 0 and write_logger:
							print_verbose = True 

					alpha = alphas[ep_iteration]
					env.epsilon_decay.append(epsilon)

					# Update rewards. 
					N = len(episode.keys())
					episode['episode_reward'] = 0 
					for t in range(N):
						
						key = episode[t]['key']

						if t > 0:
							policy[last_key,action]['reward'] = reward
							if policy[last_key,action]['next_state']:
								policy[last_key,action]['next_state'] = key
						if t == N-1:
							break

						state = episode[t]['state']
						action = episode[t]['action']
						reward = episode[t]['reward']
						old_value = state.rewards[action]
						next_max = max(episode[t+1]['state'].rewards)
						new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
						
						#print t, old_value, reward, next_max, new_value
						episode['episode_reward'] += reward

						# Add rewards to the self dict
						env.states[key]['state'].add_rewards(action, new_value)
						env.states[key]['visit_count'] += 1
						env.current_values[key]['visits'] += 1

						last_key = key

					starting_state = episode[0]['key']
					env.current_values[starting_state]['episode_lengths'].append(N-1)
					env.current_values[starting_state]['last_episode'] = episode
					episode_sizes.append(N-1)

					# Compute state-values and check if converged.
					_converged_ = []
					if epsilon == epsilons[-3]:
						convergence_cutoff = 1e-4
					if epsilon == epsilons[-2]:
						convergence_cutoff = 1e-5
					if epsilon == epsilons[-1]:
						convergence_cutoff = 1e-6
					
					for key in env.current_values:
						_rewards = env.states[key]['state'].rewards
						actions = np.where(np.isfinite(_rewards))[0]

						action_values = _rewards[actions].reshape(actions.shape[0])
						best_action_value = np.max(action_values)
						
						this_state = env.current_values[key]
						this_state['value'].append(best_action_value)
						this_state['action_values'].append(action_values)

						Nbonds = unique_states[key]['bonds']
						if Nbonds == max_bonds: 
							_converged_.append(True)
							continue

						visits = this_state['visits']
						if visits >= N_visit_min:
							ave_value_0 = np.mean(np.array(this_state['action_values'][:-1]), axis = 0)
							ave_value_f = np.mean(np.array(this_state['action_values']), axis = 0)
							#print "DIFF", ave_value_f-ave_value_0
							_converged_.append(all(abs(ave_value_f-ave_value_0) <= convergence_cutoff))

					if len(_converged_) == len(env.current_values):
						if all(_converged_):
							is_converged = True
						else:
							is_converged = False

					# Print progress of v(state)
					if print_verbose or ((episode_no+1 == N_episodes) and write_logger) or is_converged:
						strp  = "=============================================================\n"
						strp += "Epoch {} | Episode {} | epsilon {} | Time per epoch {}\n".format(epoch+1, episode_no+1, float("%0.3f"%epsilon), float("%0.2f"%np.mean(epoch_times)))
						strp += "-------------------------------------------------------------\n"
						strp += "State, Optimal value, N_visits, N_bonds | State-action values\n"
						strp += "-------------------------------------------------------------\n"
						largest_error = 0.0
						for (key,X) in sorted(env.current_values.items(), key = lambda x: (np.mean(x[1]['value']), np.mean(x[1]['action_values'][-1])), reverse = True):
							value = np.mean(X['value'])
							#action_values_0 = X['action_values'][-2]
							#action_values_f = X['action_values'][-1]
							action_values_0 = np.mean(np.array(X['action_values'][:-1]), axis = 0)
							action_values_f = np.mean(np.array(X['action_values']), axis = 0)
							difference = abs(action_values_f-action_values_0)
							largest_error = max(largest_error, max(difference))
							#" ".join([str(float("%0.2f"%x)) for x in action_values_0])
							action_values = " ".join([str(float("%0.3f"%x)) for x in action_values_f])
							#difference = " ".join([str(float("%0.5f"%x)) for x in difference])
							visits = X['visits']#[-1]
							bonds = int(unique_states[key]['bonds'])
							strp += "{} {} {} {} | {}\n".format(key, float("%0.3f"%value), visits, bonds, action_values)
							
						if len(episode_sizes):
							_lengths_ = [env.current_values[key]['episode_lengths'][-1] for key in env.current_values if len(env.current_values[key]['episode_lengths'])]
							_average_length_ = np.mean(_lengths_)
							strp += "--------------------------------------------------------------------\n"
							strp += "Average episode length: {} | max-last-length: {} | max-length: {}\n".format(float("%0.2f"%_average_length_),max(_lengths_),max(episode_sizes))
							strp += "Max error on state-action values: {}".format(largest_error)
							#strp += "-------------------------------------------------------------------------"
					
						if verbose or is_converged:
							print strp

						if write_logger:
							with open(logger_fileName, "a+") as fid:
								fid.write("{}".format(strp))

						if is_converged:
							break

					if is_converged:
						break

				if is_converged:
					break

			if is_converged:
				p.close()
				p.terminate()
				break

	print "--------------------------------------------------------------------"
	print "It took {} s to converge to optimal state-action values".format(time.time()-t_start)
  	print "... making plots and movies"
	
	# Make some plots 

	fig = fig_window(1, scale_y = 1.25)
	plt.subplot(311)
	for key in random.sample(env.current_values, min(len(env.current_values), 50)):
		for xx in range(len(env.current_values[key]['action_values'][-1])):
			y = np.array(env.current_values[key]['action_values'])[:,xx]
			plt.plot(range(len(y)),y)
	plt.ylabel('State-action value')
	
	plt.subplot(312)
	if len(episode_sizes) >= 1000:
		episode_sizes = running_mean(episode_sizes, 1000)
	plt.plot(range(len(episode_sizes)), episode_sizes)
	plt.ylabel('Episode length')
	xmax = 0.8 * len(episode_sizes)
	ymax = 0.7 * max(episode_sizes)
	_lengths_ = [env.current_values[key]['episode_lengths'][-1] for key in env.current_values if len(env.current_values[key]['episode_lengths'])]
	_mean_ = float("%0.3f" % np.mean(_lengths_))
	plt.text(xmax,ymax,'Mean: {}\nMax:   {}'.format(_mean_,max(_lengths_)))

	plt.subplot(313)
	plt.plot(range(len(env.epsilon_decay)),env.epsilon_decay)
	plt.xlabel('Episode')
	plt.ylabel('Epsilon')

	plt.tight_layout()
	
	F = 'images/' + 'L_{}_{}'.format(length,seq)
	pylab.savefig(F+'.pdf', pad_inches=0, transparent=False)
	os.system('pdf-crop-margins -v -s -u %s.pdf 1> /dev/null 2> /dev/null' % F)
	os.system('mv %s_cropped.pdf %s.pdf' % ('L_{}_{}'.format(length,seq),F))

	# Make movies of 50 longest trajectories
	total = 0
	sorted_states = sorted(env.current_values.items(), key = lambda x: x[1]['value'][-1]) 
	for (_key,item) in sorted_states:
		if total >= 50: 
			break 
		episode = item['last_episode']
		if not episode: continue
		N = len(episode)-1
		if N == 2: 
			# N = 2 is 1 move. Don't bother making such short movies. 
			continue
		name = '{}_{}_{}_{}'.format(seq,total,N-1,_key)
		for t in range(N):
			key = episode[t]['key']
			initial_chain = unique_states[key]['chain']
			T = Trajectory()
			A = Actions(Lx = Lx, Ly = Lx, available = available_move_details, use_neural_model = False)
			C = Chain(seq)
			moves = Moves(C)
			C.load_configuration(initial_chain)
			initial_chain = C.chain 
			initial_grid = C.grid
			initial_bonds_dict, initial_bonds_grid, bond_dirs, N_bonds_f = C.compute_energy()
			string_id = '{}_{}'.format(name,t)
			T.draw(initial_chain, initial_grid, initial_bonds_dict, key, string_id, Lx, Ly)
		traj_id = '/tmp/{}'.format(name)
		save_as_movie(traj_id, traj_id)
		os.system('rm {}*.png'.format(traj_id))
		os.system('mv {}.mp4 images/{}/{}.mp4'.format(traj_id,length,name))
		total += 1 

	# Dump unique_states to pickle. 
	with open(save_states_fileName, "wb") as fid:
		for state in unique_states:
			pickle.dump([state,unique_states[state],env.current_values[state]],fid,pickle.HIGHEST_PROTOCOL)
	with open(logger_fileName, "a+") as fid:
		fid.write("Saved state-action dictionary to {}\n".format(save_states_fileName))

		