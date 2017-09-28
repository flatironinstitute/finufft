#!/usr/bin/python3

from accuracy_tests import *

class Processor:
	name='finufft.accuracy_tests'
	inputs=[]
	outputs=[]
	parameters=[
		{"name":"num_nonuniform_points","optional":True,"default_value":100},
		{"name":"eps","optional":True,"default_value":"1e-6"}
	]
	opts={"cache_output":False}
	def run(self,args):
		accuracy_tests(int(args['num_nonuniform_points']),float(args['eps']))
		return True