#!/rigel/cheme/users/jss2278/env/rdkit/bin/python
import os, sys, subprocess, itertools, time 

def print_and_save(string):
	return 

length = int(sys.argv[1])
all_combos = map(list, itertools.product(['H', 'P'], repeat=length))

print "Total combinations of length {}: {}".format(length,len(all_combos))

total = 0
for k,combo in enumerate(all_combos):
	t0 = time.time()
	seq = "".join(combo)

	if k == 0:
		print 'Generating configs + moves {} for L = {} ...'.format(k,length)
		launch_arg = 'python config_generator.py {}'.format(length)
		x = subprocess.Popen(launch_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

	#####

	print 'Generating trajectory labeled {} for sequence {} ...'.format(k,seq)

	if combo.count('H') <= 1:
		print '... no states with bonds found'
		continue
	else:
		launch_arg = 'python value_iteration.py {}'.format(seq)
		x = subprocess.Popen(launch_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

		if x[0].find('cannot') > 0:
			print '... no states with bonds found'
			continue

	print '... finished in {} s'.format(time.time()-t0)
	total += 1
	
	#if total >= 10:
	#	break