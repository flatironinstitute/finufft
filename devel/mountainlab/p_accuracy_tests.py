#!/usr/bin/python3

from accuracy_tests import *

class Processor:
	name='finufft.accuracy_tests'
	inputs=[]
	outputs=[]
	parameters=[]
	def run(self,args):
		accuracy_tests()
		return True