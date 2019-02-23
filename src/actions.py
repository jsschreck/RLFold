from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,  BatchNormalization, LeakyReLU
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam

from RLFold.src.particle import Grid
from RLFold.src.state import State
import numpy as np, random 
from scipy.sparse import csr_matrix, vstack as sparse_vstack 

def custom(y_true, y_pred):
	# scale predictions so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip to prevent NaN's and Inf's
	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
	# calc
	loss = y_true * K.log(y_pred) # * weights
	loss = -K.sum(loss, -1)

	mae  = K.mean(K.abs(y_pred - y_true), axis=-1)
	return loss * mae

def focal_loss(y_true, y_pred):
	gamma = 1.0 
	y_true = K.round(y_true)
	p = y_pred + K.epsilon()
	p2 = p * p 
	op2 = (1 - p) * (1 - p)
	loss = y_true * K.log(p) * op2 + (1-y_true) * K.log(1-p) * p2
	return -K.mean(loss)

def available_moves(Lx, Ly):
	moves = {}
	encoder = {}
	decoder = {}
	for i in range(-Lx,Lx):
		for j in range(-Ly,Ly):
			if (i,j) not in moves: moves[(i,j)] = []
			ip1 = i+1 #% Lx 
			im1 = i-1 
			jp1 = j+1 #% Ly 
			jm1 = j-1 
			#if im1 < 0: im1 += Lx
			#if jm1 < 0: jm1 += Ly
			possible_moves = [(im1,jp1), (im1,j), (im1,jm1), (i,jp1), (i,jm1), (ip1,jp1), (ip1,j), (ip1,jm1)]
			for (ip,jp) in possible_moves:
				moves[i,j].append((ip,jp))
	total_actions = 0
	for c0 in sorted(moves, key = lambda x: (x[0],x[1])):
		for c1 in moves[c0]:

			# Decoder dict. contains 4X too many moves.
			# The n.n. output should be equal to the number of unique counters
			# not the length of the decoder dictionary. 

			s_c0 = list(c0)
			s_c1 = list(c1) 
			
			if s_c0[0] < 0:
				s_c0[0] += Lx
			s_c0[0] = s_c0[0] % Lx
			if s_c1[0] < 0:
				s_c1[0] += Lx 
			s_c1[0] = s_c1[0] % Lx
			if s_c0[1] < 0:
				s_c0[1] += Ly 
			s_c0[1] = s_c0[1] % Ly
			if s_c1[1] < 0:
				s_c1[1] += Ly 
			s_c1[1] = s_c1[1] % Ly

			#print c0, c1, s_c0, s_c1
		
			s_c0 = tuple(s_c0)
			s_c1 = tuple(s_c1)
			if (s_c0,s_c1) not in decoder:
				decoder[s_c0,s_c1] = total_actions
				decoder[c0,c1] = total_actions
				total_actions += 1
			else:
				decoder[c0,c1] = decoder[s_c0,s_c1]

			#decoder[c0,c1] = counter
			encoder[total_actions] = [s_c0,s_c1]
			#counter += 1

	return total_actions, moves, encoder, decoder

class LearningRateTracker(Callback):
	def on_epoch_end(self, epoch, logs={}):
		logs = logs or {}
		logs['lr'] = K.get_value(self.model.optimizer.lr)

class Actions(Grid):
	def __init__(self, Lx = 10, Ly = 10, available = None, model_path = '', weights_path = '', use_neural_model = False, lr = 0.001, load = False):
		Grid.__init__(self, Lx = Lx, Ly = Ly)

		if not available:
			self.total_actions, self.all_possible_moves, self.encoder, self.decoder = available_moves(Lx,Ly)
		else:
			self.total_actions, self.all_possible_moves, self.encoder, self.decoder = available
		
		self.states = {}
		self.model_path = model_path 
		self.weights_path = weights_path 
		self.use_neural_model = use_neural_model
		self.lr = lr 
		self.load = load 

	def initialize_state(self, initial_grid, bonds_grid, bonds_dict):
		initial_state_key = np.array_str(np.hstack([initial_grid,bonds_grid]))
		if initial_state_key not in self.states:
			rewards_size = self.total_actions
			#rewards_size = len(self.decoder)
			self.states[initial_state_key] = State(initial_grid, bonds_grid, bonds_dict, rewards_size)
		return self.states[initial_state_key]
		
	def select_move(self, state, available_moves, epsilon = 0.0):
		idx = []
		move_details = {}
		rewards = state.rewards
		for particle in available_moves:
			particle_moves = available_moves[particle]['moves']
			for move in particle_moves:
				c0, c1 = move
				action_vector_index = self.decoder[c0,c1]
				idx.append(action_vector_index)
				move_details[action_vector_index] = [particle,[c0,c1]]

		if random.random() < epsilon:
			selected_action_idx = random.choice(idx)
		else:
			if self.use_neural_model == True:

				'''
					- ONLY use the neural network on allowed moves (assumed that the probabilities of 
					illegal actions are fixed to zero even though after training the network will 
					sometimes pick illegal moves)

					- Remove dependence on encoder. 
				'''
				occupied_sites = np.zeros(len(self.decoder))
				occupied_sites[idx] = 1

				probability_over_actions = self.model.predict(state.fps(), verbose=0)[0]
				probability_over_actions *= occupied_sites
				#selected_action_idx = np.argmax(probability_over_actions)
				degenerate_action_scores = np.where(probability_over_actions==max(probability_over_actions))[0]
				selected_action_idx = random.choice(degenerate_action_scores)
				selected_action = self.encoder[selected_action_idx]
				reward = rewards[selected_action_idx]
				
				'''
				did_we_select_correctly = False
				for particle in available_moves:
					particle_moves = available_moves[particle]['moves']
					for move in particle_moves:
						if move == selected_action:
							did_we_select_correctly = True 
				if did_we_select_correctly:
					print "True"
				else:
					print "False"
				'''
				
				#print selected_action_idx, reward 

				### How to deal with cases when Q selected a move that does not apply to any particle
				
			else: # Use random selection policy
				selector = [[rewards[action_idx], action_idx] for action_idx in idx]
				max_reward, _idx = max(selector)
				other_possible_actions = [x for x in selector if x[0] == max_reward]
				selected_action = random.choice(other_possible_actions)
				reward, selected_action_idx = selected_action
		selected_particle, selected_action = move_details[selected_action_idx]
		return selected_action_idx, selected_particle, selected_action, rewards[selected_action_idx]

	def update_tree(self, state, action_idx, reward):
		state = self.states[state]
		state.add_rewards(action_idx, reward)
	
	def Q(self, Lx = 50, Ly = 50, color_dim = 2):
		if self.load:
			model = model_from_json(open(self.model_path).read())
			adam = Adam(lr=self.lr, clipnorm=1.0)
			model.compile(loss='categorical_crossentropy', 
						  optimizer = adam, 
						  metrics=['categorical_accuracy']
						  )
			model.load_weights(self.weights_path)
		else:
			model = Sequential()

			model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(Lx,Ly,color_dim), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
			
			model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
			
			model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

			'''
			
			#model.add(ZeroPadding2D((1,1), input_shape=(Lx,Ly,color_dim)))
			model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', input_shape=(Lx,Ly,color_dim), use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			#model.add(ZeroPadding2D((1,1)))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
			
			#model.add(ZeroPadding2D((1,1)))
			model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			#model.add(ZeroPadding2D((1,1)))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
			
			#model.add(ZeroPadding2D((1,1)))
			model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			#model.add(ZeroPadding2D((1,1)))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
			
			#model.add(ZeroPadding2D((1,1)))
			model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			#model.add(ZeroPadding2D((1,1)))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

			#model.add(ZeroPadding2D((1,1)))
			model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			#model.add(ZeroPadding2D((1,1)))
			model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
			'''

			model.add(Flatten()) # Feature identification
			
			# Start classification 
			#model.add(Dense(512, activation='linear', use_bias=False))
			#model.add(BatchNormalization())
			#model.add(LeakyReLU(alpha=0.1))
			#model.add(Dropout(0.5))

			model.add(Dense(512, use_bias=False))
			model.add(BatchNormalization())
			model.add(LeakyReLU(alpha=0.1))
			model.add(Dropout(0.1))

			model.add(Dense(len(self.decoder), activation='softmax'))
			
			adam = Adam(lr=self.lr, clipnorm=1.0)
			model.compile(loss = 'categorical_crossentropy',
						  optimizer = adam, 
						  metrics=['categorical_accuracy'])
			
			#model.count_params()
			#model.summary()
			with open("neural/model.json", 'w') as outfile:
				outfile.write(model.to_json())
		return model 

	def train(self, iteration, batch_size, epochs, data = None, split = 0.0, max_bonds = 100):

		def anneal_lr(epoch_no):
			return self.lr * (1.0 / (1 + np.sqrt(epoch_no)))

		def softmax(z):
			z_norm = np.exp(z-np.max(z,axis=0,keepdims=True))
			return(np.divide(z_norm,np.sum(z_norm,axis=0,keepdims=True)))
		
		def get_callbacks():
			from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
			callbacks = [
				ModelCheckpoint('neural/weights.h5', monitor = 'val_loss', save_best_only = True, mode = 'auto'), 
				EarlyStopping(monitor='val_loss', patience = 3),
				#LearningRateScheduler(anneal_lr),
				LearningRateTracker(),
				ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 2, min_lr=1e-7, mode='auto'), 
				CSVLogger('neural/training.log', separator= ' ', append = True),
				]
			return callbacks

		if not data:
			batch_x, batch_y = [], []
			for (key,item) in [x for x in self.states.items() if (x[1]['bonds'] != max_bonds)]:
				State = item['state']
				scaled_x = State.fps(self.Lx).astype(np.float64)
				
				scaled_y = np.zeros(State.rewards.shape[0])
				max_idx = np.where(State.rewards==np.max(State.rewards))[0]
				scaled_y[max_idx] = 1
				
				batch_x.append(scaled_x)
				batch_y.append(scaled_y)

			batch_x = np.vstack(batch_x)
			batch_y = np.vstack(batch_y)
			X_trn, X_tst, y_trn, y_tst = train_test_split(batch_x, batch_y, test_size = split)
		
		else:
			X_trn, X_tst, y_trn, y_tst = data

		#datagen_train = ImageDataGenerator()
		#datagen_train.fit(X_trn)

		#datagen_test  = ImageDataGenerator()
		#datagen_test.fit(X_tst)

		print "... iteration:", iteration 
		print "... total states:", X_trn.shape[0]
		#print "Batch size:", batch_size

		callbacks = get_callbacks()
		lr = anneal_lr(iteration)
		#K.set_value(self.model.optimizer.lr, lr)

		self.model = self.Q(Lx = X_trn.shape[1], 
							Ly = X_trn.shape[2],
							color_dim = X_trn.shape[3])
		'''
		self.model.fit_generator(datagen_train.flow(X_trn, y_trn, batch_size = batch_size),
                    			 steps_per_epoch = len(X_trn) / batch_size, 
                    			 validation_data = datagen_train.flow(X_tst, y_tst, batch_size = batch_size),
        						 validation_steps = len(X_tst) / batch_size,
                    			 epochs = epochs,
                    			 verbose = 1,
                    			 shuffle = True,
                    			 callbacks = callbacks)

		'''
		hist = self.model.fit(X_trn, y_trn, 
					   		  batch_size = batch_size, 
					   		  epochs = epochs, 
					   		  verbose = 1, 
					   		  validation_data=(X_tst, y_tst),
					   		  shuffle = True,
					   		  callbacks = callbacks
					   		  )
		
		#print "... mean trainining loss:", np.mean(hist.history["loss"])

		def evaluate(x,y, _str = 'train'):
			#scores = self.model.predict_generator(datagen_train.flow(x,y, batch_size = batch_size))
			scores = self.model.predict(x, verbose=0) 
			top1 = []
			totals = []
			for c,(y1,y2) in enumerate(zip(y,scores)):
				
				total = len(np.where(y1>1e-6)[0])
				correct_sorted_indices = np.argsort(y1)[::-1][:total]
				predicted_sorted_indices = np.argsort(y2)[::-1][:total]

				if correct_sorted_indices[0] == predicted_sorted_indices[0]:
					top1.append(100)
				else:
					top1.append(0)

				#correct_sorted_indices = correct_sorted_indices))
				#predicted_sorted_indices = set(list(predicted_sorted_indices))
				correct = 0
				for idx in predicted_sorted_indices:
					if idx in correct_sorted_indices:
						correct += 1
				totals.append(100.0 * correct/float(total))
			print "... {}: Percent choosing a legal / best move: {} / {}".format(_str,np.mean(totals),np.mean(top1))

		evaluate(X_trn,y_trn,'train')
		
		if split:
			evaluate(X_tst,y_tst,'val')

		self.model.save_weights("neural/weights.h5", overwrite = True)

	def shift(self,x,cutoff):
		if x >= 0:
			if x >= cutoff: 
				x = x % cutoff 
			else:
				x < cutoff
		else:
			x += cutoff 
		return x 