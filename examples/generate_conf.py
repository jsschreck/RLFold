import random, sys

if len(sys.argv) < 2:
    print 'Usage: python generate_conf.py [seq]'
    sys.exit(1)

sequence = sys.argv[1]

conf = {}
conf["HPSTRING"] = sequence

svec = ",".join(['0' for x in range(len(conf["HPSTRING"])-1)])

conf["INITIALVEC"] = "["+svec+"]"
conf["RESTRAINED_STATE"] = '[]'
conf['randseed'] = random.randint(1,10000)
conf["eps"] = 0.0
conf["NREPLICAS"] = 1
conf["REPLICATEMPS"] = [0.0]
conf["EXPDIR"] = "./{}".format(len(conf["HPSTRING"]))
conf["PRINTEVERY"] = 1000
conf["TRJEVERY"] = 1
conf["ENEEVERY"] = 1000
conf["NATIVEDIR"] = "{}".format(len(conf["HPSTRING"]))

with open("{}.conf".format(sequence), "w") as fid:
    for x in ["HPSTRING", "INITIALVEC", "RESTRAINED_STATE", 'randseed', 'eps', "NREPLICAS", "REPLICATEMPS", "EXPDIR", "PRINTEVERY", "TRJEVERY", "ENEEVERY", "NATIVEDIR"]:
        y = conf[x]
        fid.write("{} {}\n".format(x,y))
