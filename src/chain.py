from RLFold.src.particle import Grid, Particle
from RLFold.src.misc import CustomDictionary
import numpy as np
import math 

class Chain(Grid):
	def __init__(self, seq = '', Lx = 10, Ly = 10):
		Grid.__init__(self, Lx = Lx, Ly = Ly)
		self.seq = seq
		self.set_topology = self.set_chain_topology()

	def set_chain_topology(self):
		self.chain = {}
		for k,base in enumerate(self.seq):
			if k == 0 or k == (len(self.seq)-1):
				self.chain[k] = Particle(base, k, 1)
			else:
				self.chain[k] = Particle(base, k, 2)
			self.chain[k].direction = None

	def generate_random_configuration(self, max_attempts = 10000):
		coor_dict = create_saw_config(n = 2 * len(self.chain), length = len(self.chain), max_attempts = max_attempts)
		if not coor_dict:
			return False
		last_particle = max(self.chain)
		for particle_id in self.chain:
			x, y = coor_dict[particle_id]
			if x < 0:
				x += self.Lx
			elif y < 0:
				y += self.Ly 
			elif x >= self.Lx:
				x = x % self.Lx
			elif y >= self.Ly: 
				y = y % self.Ly
			self.grid[y,x] = self.chain[particle_id].base
			self.chain[particle_id].x = x
			self.chain[particle_id].y = y
		return self.chain, self.grid

	def random_walk_configuration(self, max_attempts = 10000):
		### Try pushing new particles away from current COM of chain.
		last_particle = max(self.chain)
		for particle_id in self.chain:
			if not particle_id:
				x = 0 * self.Lx
				y = 0 * self.Ly
				self.grid[y,x] = self.chain[particle_id].base
				self.chain[particle_id].x = x
				self.chain[particle_id].y = y
				continue
			attempts = 0 
			while attempts <= max_attempts:
				attempts += 1
				last_x = self.chain[particle_id-1].x
				last_y = self.chain[particle_id-1].y
				current_x = last_x + random.choice([0,1,1,1])
				current_y = last_y + random.choice([0,1,1,1])
				if not (abs(current_x - last_x) + abs(current_y - last_y)) == 1:
					continue
				current_x = current_x % self.Lx
				current_y = current_y % self.Ly
				if not self.grid[current_y,current_x]:
					#print current_x, current_y
					self.grid[current_y,current_x] = self.chain[particle_id].base
					self.chain[particle_id].x = current_x
					self.chain[particle_id].y = current_y
					break
			if attempts == max_attempts:
				return 0
			return self.chain, self.grid

	def load_configuration_from_coors(self, config):
		for k, (x,y) in enumerate(config):
			self.grid[y,x] = self.chain[k].base
			self.chain[k].x = x
			self.chain[k].y = y
		self.canonicalize()

	def load_configuration(self, chain):
		for particle_id in chain:
			x = chain[particle_id].x
			y = chain[particle_id].y
			self.grid[y,x] = self.chain[particle_id].base
			self.chain[particle_id].x = x
			self.chain[particle_id].y = y
		self.canonicalize()

	def vec2coords(self, coordinates):
		"""Convert a list of chain vectors to a list of coordinates (duples).""" 
		tmp = [(0,0)]
		x = 0
		y = 0
		for k,i in enumerate(coordinates[1:]):
			if i == 0:
				y = y + 1
			if i == 1:
				x = x + 1
			if i == 2:
				y = y - 1
			if i == 3:
				x = x - 1
			tmp.append((x,y))
		return tmp

	def update_directions(self):
		
		#for particle in self.chain:
		#	x0, y0 = self.chain[particle].x, self.chain[particle].y
		#	print particle, x0, y0 

		for particle in self.chain:
			if not particle:
				self.chain[particle].direction = None
				continue
			x0, y0 = self.chain[particle-1].x, self.chain[particle-1].y
			xf, yf = self.chain[particle].x, self.chain[particle].y
			drx = xf - x0
			dry = yf - y0
			
			#     0     up
			#     1     right
			#     2     down
			#     3     left
			# {0,1,2,3} =  {n,w,s,e} = {U,R,D,L}

			if drx == 0 and dry == 1:
				direction = 0
			elif drx == 1 and dry == 0:
				direction = 1
			elif drx == 0 and dry == -1:
				direction = 2
			elif drx == -1 and dry == 0:
				direction = 3
			else:
				#print particle, drx, dry, x0, y0, xf, yf
				raise Exception('You missed a direction')
		
			self.chain[particle].direction = direction

	def particle_directions(self):
		particle_dirs = []
		for particle in self.chain:
			p_dir = self.chain[particle].direction
			if p_dir is None:
				p_dir = 'X'
			particle_dirs.append(p_dir)
		return particle_dirs

	def canonicalize(self):
		
		self.update_directions()
		
		#seq = [self.chain[particle].direction for particle in self.chain]
		#t = self.vec2coords(seq)
		#print "a", seq, t 

		first_R = False
		updated_seq = [None]
		for k in range(1,len(self.chain)):
			if k == 1:
				if self.chain[k].direction != 1:
					if self.chain[k].direction == 0: # If pointing up
						for j in range(k, len(self.chain)):
							self.chain[j].direction += 1  
							self.chain[j].direction = self.chain[j].direction % 4
					elif self.chain[k].direction == 2: # If pointing down
						for j in range(k, len(self.chain)):
							self.chain[j].direction -= 1  
							self.chain[j].direction = self.chain[j].direction % 4
					else:
						print "HEY", self.chain[k].direction

			else:
				if not first_R: 
					if self.chain[k].direction == 0: # If pointing up, do nothing
						first_R = True 
					elif self.chain[k].direction == 2: # If pointing down, flip ups <--> downs
						first_R = True
						for j in range(k, len(self.chain)):
							if self.chain[j].direction in [0,2]:
								self.chain[j].direction += 2 
								self.chain[j].direction = self.chain[j].direction % 4
					else:
						pass
		 				#print "WHAAAAA", self.chain[k].direction
			updated_seq.append(self.chain[k].direction)

		t = self.vec2coords(updated_seq)
		#print "b", updated_seq, t
		for particle, (xf,yf) in enumerate(t):
			x0 = self.chain[particle].x
			y0 = self.chain[particle].y
			self.grid[y0,x0] = 0
			self.chain[particle].x = xf
			self.chain[particle].y = yf
			self.grid[yf,xf] = self.chain[particle].base
			
	def distance(self,x1,x2,y1,y2):
		dx = abs(x1 - x2)
		dy = abs(y1 - y2)
		if dx > self.Lx / 2:
			dx = self.Lx - dx
		if dy > self.Ly / 2:
			dy = self.Ly - dy
		return dx + dy

	def move_to_origin(self):
		# Chain coordinates and the grid needs to be updated. 
		return 

	def compute_energy(self):
		#     0     up
		#     1     right
		#     2     down
		#     3     left
		# {0,1,2,3} =  {n,w,s,e} = {U,R,D,L}
		bond_orientations = CustomDictionary()
		particles = self.chain.keys()
		self.bonds_dict = {p: [] for p in particles}
		self.bonds_array = Grid(Lx = self.Lx, Ly = self.Ly).grid
		for particle_id in particles:
			bond_orientations.initialize(particle_id)
			if self.chain[particle_id].base > 2: # H ~ 1, 2; P ~ 3, 4
				bond_orientations.add_to_dict(particle_id,None)
				continue
			current_x = self.chain[particle_id].x
			current_y = self.chain[particle_id].y
			if particle_id == 0:
				neighbors = set([particle_id,particle_id+1])
			elif particle_id == max(particles):
				neighbors = set([particle_id-1,particle_id])
			else:
				neighbors = set([particle_id-1,particle_id,particle_id+1])
			potential_binding_partners = list(set(particles) - neighbors)
			for next_particle in potential_binding_partners:
				if self.chain[next_particle].base > 2:
					continue
				next_x = self.chain[next_particle].x
				next_y = self.chain[next_particle].y
				if self.distance(current_x,next_x,current_y,next_y) == 1:
					self.bonds_dict[particle_id].append(next_particle)
					self.bonds_array[current_y,current_x] = 1
					self.bonds_array[next_y,next_x] = 1
		total_bonds = {}
		for particle,partners in self.bonds_dict.items():
			if len(partners):
				for partner in partners:
					if (particle,partner) not in total_bonds and (partner,particle) not in total_bonds:
						total_bonds[(particle,partner)] = 1

						x0, y0 = self.chain[particle].x, self.chain[particle].y
						xf, yf = self.chain[partner].x, self.chain[partner].y
						drx = xf - x0
						dry = yf - y0
		
						# {0,1,2,3} =  {n,w,s,e} = {U,R,D,L}

						if drx == 0 and dry == 1:
							direction = 0
						elif drx == 1 and dry == 0:
							direction = 1
						elif drx == 0 and dry == -1:
							direction = 2
						elif drx == -1 and dry == 0:
							direction = 3
						else:
							print "You missed a direction"
						bond_orientations.add_to_dict(particle,direction)

		total_bonds = float(len(total_bonds))

		return self.bonds_dict, self.bonds_array, bond_orientations, total_bonds

	def check_distances(self):
		valid_configuration = True 
		# First check if we have overlapping particles
		particle_coordinates = list(set([(self.chain[k].x, self.chain[k].y) for k in self.chain]))
		if len(particle_coordinates) != len(self.chain):
			return -1
		for k in range(1,len(self.chain.keys())):
			dist = self.distance(self.chain[k-1].x,self.chain[k].x,self.chain[k-1].y,self.chain[k].y)
			if dist != 1:
				return -dist
		return 1
