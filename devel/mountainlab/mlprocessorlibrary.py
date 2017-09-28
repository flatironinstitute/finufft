import sys
import os
import json

class MLProcessorLibrary:
	_processors=[]
	def addProcessor(self,processor):
		self._processors.append(processor)
	def run(self,argv):
		if (len(sys.argv) < 2):
			print("At least one argument expected")
			return False
		arg1=argv[1]
		if arg1 == 'spec':
			spec=self.getSpec(argv)
			print(json.dumps(spec, sort_keys=True, indent=4))
			return True
		P=self.findProcessor(arg1)
		if P is None:
			print("Unable to find processor: {}".format(arg1))
			return False
		args=self._get_args_from_argv(sys.argv)
		if not self._check_args(P,args):
			return False
		return P.run(args)
	def getSpec(self,argv):
		spec={"processors":[]}
		for j in range(0,len(self._processors)):
			obj=self.getProcessorSpec(self._processors[j])
			program=os.path.abspath(argv[0])
			obj["exe_command"]="{} {} $(arguments)".format(program,self._processors[j].name)
			spec["processors"].append(obj)
		return spec
	def getProcessorSpec(self,P):
		spec={"name":P.name}
		spec["inputs"]=P.inputs
		spec["outputs"]=P.outputs
		spec["parameters"]=P.parameters
		spec["opts"]=P.opts
		return spec
	def findProcessor(self,processor_name):
		for j in range(0,len(self._processors)):
			if (self._processors[j].name == processor_name):
				return self._processors[j]
		return None
	def _get_args_from_argv(self,argv):
		args={}
		for j in range(2,len(argv)):
			arg0=argv[j]
			if (arg0.startswith("--")):
				tmp=arg0[2:].split("=")
				if (len(tmp)==2):
					args[tmp[0]]=tmp[1]
				else:
					print("Warning: problem with argument: {}".format(arg0))
					exit(-1)
			else:
				print("Warning: problem with argument: {}".format(arg0))
				exit(-1)
		return args
	def _check_args(self,P,args):
		valid_params={}
		for j in range(0,len(P.inputs)):
			valid_params[P.inputs[j]["name"]]=1
			if not P.inputs[j]["name"] in args:
				print("Missing input path: {}".format(P.inputs[j]["name"]))
				return False
		for j in range(0,len(P.outputs)):
			valid_params[P.outputs[j]["name"]]=1
			if not P.outputs[j]["name"] in args:
				print("Missing output path: {}".format(P.outputs[j]["name"]))
				return False
		for j in range(0,len(P.parameters)):
			valid_params[P.parameters[j]["name"]]=1
			if not P.parameters[j]["optional"]:
				if not P.parameters[j]["name"] in args:
					print("Missing required parameter: {}".format(P.parameters[j]["name"]))
					return False
		for key in args:
			if not key in valid_params:
				if not key.startswith("_"):
					print("Invalid parameter: {}".format(key))
					return False
		return True
