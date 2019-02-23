import numpy as np

class Particle:
	def __init__(self, base, coordinate, num_neighbors, x = None, y = None):
		def base_to_index(base = base):
			if base == "H":
				if num_neighbors == 1:
					return 1
				else:
					return 2
			else:
				if num_neighbors == 1:
					return 3
				else:
					return 4
		self.coordinate = coordinate
		self.base_id = base
		self.base = base_to_index()
		self.x = x
		self.y = y

class Grid(object):
	def __init__(self, Lx = 10, Ly = 10):
		self.Lx = Lx 
		self.Ly = Ly
		self.grid = np.zeros((Ly,Lx), dtype = np.int)