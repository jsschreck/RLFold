import random, copy, matplotlib
import matplotlib.pyplot as plt
import pylab
try:
	plt.switch_backend('Qt5Agg')
except:
	plt.switch_backend('agg')

class Trajectory:
	def __init__(self):
		self.loaded_trajectory = {}
		return

	def convert_base_tcl(self, base):
		if base == 'H':
			base = 'white'
		else:
			base = 'black'
		return base

	def save_to_tcl(self, configuration, filename, append = True):
		if append:
			mode = 'a+'
		else:
			mode = 'w'
		with open(filename, mode) as fid:
			num_particles = len(configuration)
			fid.write("color Display Background white\nmol new\n")
			for coordinate in sorted(configuration.keys()):
				_p = configuration[coordinate]
				fid.write("graphics 0 color %s \n" % self.convert_base_tcl(_p.base))
				fid.write("graphics 0 sphere {%s %s 0} radius 0.25\n" % (_p.x,_p.y))
				if coordinate:
					last_x = configuration[coordinate-1].x
					last_y = configuration[coordinate-1].y
					fid.write("graphics 0 color black\n")
					fid.write("graphics 0 line {%s %s 0} {%s %s 0} width 5\n" % (last_x, last_y, _p.x,_p.y))
	
	def set_com_to_zero(self, configuration, Lx = 10, Ly = 10):
		cdm = np.array([0.,0.])
		for particle in configuration:
			p = configuration[particle]
			cm_pos = np.array([p.x,p.y])
			diff = np.rint(cm_pos / Lx) * Lx
			cdm += (cm_pos - diff)
		cdm = cdm / float(len(configuration))
		for particle in configuration:
			p = configuration[particle]
			p.x -= int(cdm[0])
			p.y -= int(cdm[1]) 
		return cdm
 
	def draw(self, chain, grid, bonds, value, i, Lx, Ly):
		fig, ax = plt.subplots()
		lines, circles = [], []

		configuration = copy.deepcopy(chain)
		#com = self.set_com_to_zero(configuration, Lx = 10, Ly = 10)
		for c,v in sorted(configuration.items()):
			color = 'k'
			if v.base == 1 or v.base == 2:
				color = '#FFFFFF'
			if c:
				if abs(last_x - v.x) > 1.5:
					v.x -= Lx
				if abs(last_y - v.y) > 1.5:
					v.y -= Ly
				lines.append([[last_x,v.x],[last_y,v.y]])
			last_x = v.x
			last_y = v.y
			circles.append([c,color,v.x,v.y])
		coordinate, color, xscale, yscale = circles[0]
		for line in lines:
			xs, ys = line 
			xs = [x - xscale for x in xs]
			ys = [y - yscale for y in ys]
			ax.plot(xs,ys, color='k', alpha=1.0, zorder = 1)
		for coor,color,_px,_py in circles:
			if len(bonds[coor]):
				for bond_coor in bonds[coor]:
					__px = circles[bond_coor][2]
					__py = circles[bond_coor][3]
					ax.plot([_px-xscale,__px-xscale],[_py-yscale,__py-yscale], color='r', ls = 'dotted', alpha=1.0, zorder = 1)
			patch = plt.Circle((_px-xscale, _py-yscale), 0.2, fc=color, ec = 'k', lw = 1, alpha=1.0, zorder = 2)
			ax.add_patch(patch)
		plt.xlim(-Lx+2,Lx-2)
		plt.ylim(-Ly+2,Ly-2)

		try:
			ii = i.split("_")
			if ii[0] != 'L':
				# Using L to generate unique configuration files. 
				episode, time = ii[1:]
				value = float("%0.3f"%value)
				plt.title('Episode: {} Time: {} Value: {}'.format(episode,time,value), fontsize=12)
			else:
				plt.title('Configuration: {}'.format(value), fontsize=12)
		except: 
			plt.title('Configuration: {}'.format(value), fontsize=12)
			pass 

		ax.set_aspect('equal')

		plt.tick_params(axis = 'both',          # changes apply to the x-axis
						which = 'both',      # both major and minor ticks are affected
						bottom = False,      # ticks along the bottom edge are off
						top = False,         # ticks along the top edge are off
						left = False, 
						right = False,
						labelleft = False,
						labelbottom = False) # labels along the bottom edge are off
		#plt.show()
		plt.savefig("/tmp/{}.png".format(i))#, transparent=True)  
		plt.close('all')

	def append_to_trajectory(self, seq, initial_chain, initial_grid, bonds_grid, bonds_dict, fileName = ''):
		initial_state_key = np.array2string(np.concatenate((initial_grid, bonds_grid)))
		if initial_state_key not in self.loaded_trajectory:
			self.loaded_trajectory[initial_state_key] = [seq, initial_chain, initial_grid, bonds_grid, bonds_dict]
			if fileName:
				with open(fileName, "a+") as fid:
					pickle.dump([initial_state_key, seq, initial_chain, initial_grid, bonds_grid, bonds_dict],fid,pickle.HIGHEST_PROTOCOL)

	def load_trajectory(self, fileName = '', N = 1e10):
		counter = 0
		keys = []
		with open(fileName, "rb") as fid:
			while True:
				try:
					initial_state_key, seq, chain, grid, bonds_grid, bonds_dict = pickle.load(fid)
					if initial_state_key not in self.loaded_trajectory:
						self.loaded_trajectory[initial_state_key] = [seq, chain, grid, bonds_grid, bonds_dict]
						counter += 1  
				except:
					break

		if N < counter:
			remove = counter - N 
			delete = random.sample(self.loaded_trajectory, remove)
			for key in delete:
				del self.loaded_trajectory[key]
		
		return len(self.loaded_trajectory)

	def load_HPtrajectory(self, fileName = '', N = 1e10):
		with open(fileName, "rb") as fid:
			while True:
				try:
					initial_state_key, seq, chain, grid, bonds_grid, bonds_dict, moves_dict = pickle.load(fid)
					if initial_state_key not in self.loaded_trajectory:
						self.loaded_trajectory[initial_state_key] = [seq, chain, grid, bonds_grid, bonds_dict, moves_dict]
						counter += 1  
				except:
					break

	def save_trajectory(self, fileName = ''):
		with open(fileName, "wb") as fid:
			for state_key, config in self.loaded_trajectory.items():
				seq, chain, grid, bonds_grid, bonds_dict = config 
				pickle.dump([state_key,seq, chain, grid, bonds_grid, bonds_dict], fid, pickle.HIGHEST_PROTOCOL)

	def yield_starting_configuration(self):
		randomly_sorted_keys = self.loaded_trajectory.keys()
		random.shuffle(randomly_sorted_keys)
		while True:
			try:
				for state_key in randomly_sorted_keys:
					seq, chain, grid, bonds_grid, bonds_dict = self.loaded_trajectory[state_key] 
					yield state_key, seq, chain, grid, bonds_grid, bonds_dict
			except Exception as E:
				yield 0 
				break 

	def yield_rand_config(self, c, max_attempts = 100):
		counter = 1 
		while counter < max_attempts:
			if random.random < 0.5:
				config_attempt = c.random_walk_configuration()
			else:
				config_attempt = c.generate_random_configuration()
			if config_attempt:
				chain_0, grid_0 = config_attempt
				break 
			else:
				counter += 1
		if counter == max_attempts:
			print "Having a hard time generating an initial configuration ... loading trajectory file"
			return False
		return chain_0, grid_0