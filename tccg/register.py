# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy

class Register:
    def __init__(self, name, numElements):
        self.name = name
        self.numElements = numElements
        self.content = ["" for i in range(numElements)]

    def setzero(self):
        self.content = ["0" for i in range(self.numElements)]

    def __str__(self):
        ret = "%s = ["%self.name
        for s in self.content:
            ret += s + ";"
        return ret[:-1]+"]"

    def setContent(self, content):
        self.content = copy.deepcopy(content)

