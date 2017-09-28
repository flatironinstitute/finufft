#!/usr/bin/python3

import sys
from mlprocessorlibrary import MLProcessorLibrary

from p_accuracy_tests import Processor as p_accuracy_tests

PL=MLProcessorLibrary()
PL.addProcessor(p_accuracy_tests())
if not PL.run(sys.argv):
	exit(-1)

