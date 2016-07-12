# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
class index:
    def __init__(self, label, size):
        self.label = label
        self.size  = size

    def __eq__(self, other):
        return (self.label == other.label)
    def __ne__(self, other):
        return not(self.label == other.label)
    
    def __str__(self):
        return "(%s,%d)"%(self.label,self.size)


