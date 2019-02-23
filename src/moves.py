class Moves(object):
	def __init__(self, Chain):
		self.Chain = Chain
		self.chain = self.Chain.chain 
		
	def available(self):
		moves = {}
		sorted_particle_ids = sorted(self.chain.keys())
		for _id in sorted_particle_ids:
			moves[_id] = {'moves': [], 'weights': []}
			_p = self.chain[_id]
			x0 = _p.x
			y0 = _p.y

			'''
			if x0 < 0:
				x0 += self.Chain.Lx
			if y0 < 0:
				y0 += self.Chain.Ly
			x0 = x0 % self.Chain.Lx
			y0 = y0 % self.Chain.Ly
			'''
			
			# End moves
			if _id == 0 or _id == sorted_particle_ids[-1]:
				# Move particle to left by one
				xf = (x0 + 1) #% self.Chain.Lx
				yf = y0
				moves[_id]['moves'].append([(x0,y0),(xf,yf)])

				# Move particle to right by one 
				xf = (x0 - 1)
				#if xf < 0: xf += self.Chain.Lx
				yf = y0 
				moves[_id]['moves'].append([(x0,y0),(xf,yf)])

				# Move particle up by one
				xf = x0 
				#yf = (y0 + 1) % self.Chain.Ly
				moves[_id]['moves'].append([(x0,y0),(xf,yf)])

				# Move particle down by one
				xf = x0 
				yf = (y0 - 1)
				#if yf < 0: yf += self.Chain.Ly
				moves[_id]['moves'].append([(x0,y0),(xf,yf)])
			
			# Corner move 1
			xf = (x0 + 1) #% self.Chain.Lx
			yf = y0 - 1
			#if yf < 0: yf += self.Chain.Ly
			moves[_id]['moves'].append([(x0,y0),(xf,yf)])

			# Corner move 2
			xf = x0 - 1
			yf = (y0 + 1) #% self.Chain.Ly
			#if xf < 0: xf += self.Chain.Lx
			moves[_id]['moves'].append([(x0,y0),(xf,yf)])

			# Corner move 3
			xf = (x0 + 1) #% self.Chain.Lx
			yf = (y0 + 1) #% self.Chain.Ly
			moves[_id]['moves'].append([(x0,y0),(xf,yf)])

			# Corner move 4
			xf = x0 - 1
			yf = y0 - 1
			#if xf < 0: xf += self.Chain.Lx
			#if yf < 0: yf += self.Chain.Ly
			moves[_id]['moves'].append([(x0,y0),(xf,yf)])

			# Do nothing 
			moves[_id]['moves'].append([(x0,y0),(x0,y0)])

		return moves 

	def perform_one_move(self, particle, move):
		x0, y0 = move[0]
		xf, yf = move[1]
		self.Chain.chain[particle].x = xf
		self.Chain.chain[particle].y = yf
		weight = self.Chain.check_distances()
		if (x0 == xf) and (y0 == yf):
			weight = 0.1
		self.Chain.grid[y0,x0] = 0
		self.Chain.grid[yf,xf] = self.Chain.chain[particle].base
		self.Chain.canonicalize()
		return float(weight), self.Chain.chain, self.Chain.grid



		