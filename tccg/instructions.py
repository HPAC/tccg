# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
from register import Register

class Instruction:
   def __init__(self, code, latency):
       self.latency = latency
       self.dependencies = []
       self.code = code

   def __str__(self):
       return self.code
   
