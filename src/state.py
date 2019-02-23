from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from scipy.sparse import csr_matrix
import scipy.stats as ss
import cPickle as pickle
import numpy as np
import copy 

def softmax(z):
	z_norm = np.exp(z-np.max(z,axis=0,keepdims=True))
	return(np.divide(z_norm,np.sum(z_norm,axis=0,keepdims=True)))

class State(object):
	def __init__(self, pos_grid, bond_grid, bonds_dict, rewards_size):
		self.state = np.array([pos_grid, bond_grid])
		self.bonds_dict = bonds_dict
		self.rewards = -np.ones(rewards_size) * np.inf
		self.counts = np.zeros(rewards_size)
		self.scaler = StandardScaler()

	def add_rewards(self, action, reward):
		self.counts[action] += 1
		self.rewards[action] = reward #self.rewards[action] + (reward - self.rewards[action]) / self.counts[action]

	def save_sparse(self, fid):
		save_arr = [self.bonds_dict]
		grid, bonds = self.state

		idx = np.where(np.isfinite(self.rewards))[0]
		scaled_y = np.zeros(self.rewards.shape[0])
		normalized = softmax(self.rewards)
		scaled_y[idx] = normalized[idx]
		#scaled_y[np.where(self.rewards==np.max(self.rewards))] = 1

		for matrix in [grid, bonds, scaled_y]:
			_matrix = csr_matrix(matrix) 
			_parameters = [_matrix.data, _matrix.indices, _matrix.indptr, _matrix.shape]
			save_arr.append(_parameters)
		
		pickle.dump(save_arr, fid, pickle.HIGHEST_PROTOCOL)

	def load_sparse(self, parameters):
		self.bonds_dict, grid, bond, reward = parameters 
		grid = csr_matrix((grid[0], grid[1], grid[2]), shape = grid[3]).todense() / 4.
		bond = csr_matrix((bond[0], bond[1], bond[2]), shape = bond[3]).todense() / 3.

		#grid = self.scaler.fit_transform(grid)
		#bond = self.scaler.fit_transform(bond)

		self.rewards = csr_matrix((reward[0], reward[1], reward[2]), shape = reward[3]).todense()
		self.state = np.array([grid, bond])
		self.counts = np.zeros(self.rewards.shape[0])

	def available_moves(self):
		return None

	def fps(self, N = 50):
		
		return [self.state.reshape(-1, 50, 50, 2)]
		
		x1 = np.rot90(self.state[0])
		y1 = np.rot90(self.state[1])
		r1 = np.array([x1,y1])

		x2 = np.rot90(r1[0])
		y2 = np.rot90(r1[1])
		r2 = np.array([x2,y2])

		x3 = np.rot90(r2[0])
		y3 = np.rot90(r2[1])
		r3 = np.array([x3,y3])
		
		return_arr = [state_arr.reshape(-1, 50, 50, 2) for state_arr in [self.state,r1,r2,r3]]

		return return_arr

		'''
		grid, bond = self.state
		row, col = np.where(grid>0)
		_grid = np.roll(np.roll(grid, np.int(max(col)/2)-2, axis=1), np.int(max(row)/2)-1, axis = 0)
		_bond = np.roll(np.roll(bond, np.int(max(col)/2)-2, axis=1), np.int(max(row)/2)-1, axis = 0)

		grid = np.zeros((N,N))
		grid[:_grid.shape[0],:_grid.shape[1]] += _grid
		grid = np.roll(np.roll(grid, np.int(N/2)-1, axis=1), np.int(N/2)-1, axis = 0)

		bond = np.zeros((N,N))
		bond[:_bond.shape[0],:_bond.shape[1]] += _bond
		bond = np.roll(np.roll(bond, np.int(N/2)-1, axis=1), np.int(N/2)-1, axis = 0)

		img = np.array([grid,bond]).swapaxes(0,2).swapaxes(0,1)
		return img.reshape(-1, N, N, 2)


		for a,n in enumerate(grid):
			for b,m in enumerate(n):
				element = grid[a,b]
				bond_element = bond[a,b]
				if bond_element == 1.0: c = 204 # red
				if bond_element == 2.0: c = 102 # pink
				if bond_element == 3.0: c = 0   # rose
				# Reds
				if element == 1.0: # H at an end 
					if bond_element > 0.0:
						img[a,b,:] = np.array([255, c, c]) 
					else:
						img[a,b,:] = np.array([153, 0, 0]) # dark red
				# Blues
				elif element == 2.0: # H having 2 neighbors
					if bond_element > 0.0:
						img[a,b,:] = np.array([c, c, 255]) 
					else:
						img[a,b,:] = np.array([0, 0, 153]) # dark blue
				# Greens
				elif element == 3.0: # P at an end
					img[a,b,:] = np.array([0,153,0]) # Light green

				elif element == 4.0: # P having 2 neighbors
					img[a,b,:] = np.array([0,255,0]) # Solid green
				# Blacks
				else:
					img[a,b,:] = np.array([0,0,0]) # unoccupied cell

		return img.reshape(-1,N,N,3) #swapaxes(0,2).swapaxes(1,2)#.reshape(-1,3,N,N)
		'''
