# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy
import tccg_util
from Index import *


class Tensor:
    def __init__(self, label, indices):
        self.indices = copy.deepcopy(indices)
        self.label = label
        self.ld = [1]
        for i in range(1,len(self.indices)):
            self.ld.append(self.ld[i-1] * self.indices[i-1].size)

    def transposeRequired(self,indices):
        for i in range(len(indices)):
            if indices[i] != self.indices[i]:
                return 1

    def countContiguousStrideOneElements(self):
        count = self.indices[0].size
        for i in range(1, len(self.indices)):
            if( self.ld[i] == count ):
                count *= self.indices[i].size
            else: # break if the indices are not contiguous anymore
                break
        return count

    def getDim(self):
        return len(self.indices)

    def getSize(self):
        size = 1
        for idx in self.indices:
            size *= idx.size
        return size

    def indicesContiguous(self, indices):
        pos = self.getPos(indices[0])
        l = 0
        while l < len(indices):
            if(self.indices[pos] != indices[l]):
                print self.indices[pos].label , indices[l].label
                return 0
            pos += 1
            l += 1
        return 1

    def getSubTensor(self, indices):
        A = copy.deepcopy(self)

        newIndices = []
        for i in range(len(self.indices)): #keep the original order of indices of the input tensor
            if( tccg_util.hasItem(indices, self.indices[i]) ):
                newIndices.append(self.indices[i])
        A.setIndices(newIndices)

        leadingDim = []
        for idx in A.indices:
            leadingDim.append(self.getLd(idx))
        A.setLd(leadingDim)

        return A
    
    def setIndices(self, indices):
        self.indices = copy.deepcopy(indices)

        self.ld = [1]
        for i in range(1,len(self.indices)):
            self.ld.append(self.ld[i-1] * self.indices[i-1].size)
    def getIndexStr(self):
        ret = ""
        for idx in self.indices:
            ret += idx.label
        return ret

    def hasIndex(self, index):
        for id in self.indices:
            if( id == index ):
                return 1
        return 0
    
    def replaceIndex(self, idx, indices):
        pos = self.getPos(idx)
        if( pos == -1 ):
            print "Tensor::replaceIndex(): index not found"
            exit(-1)

        size = 1
        for idxt in indices:
            size *= idxt.size
        if( size != idx.size ):
            print "Tensor::replaceIndex(): size does not match"
            exit(-1)

        del self.indices[pos]
        # insert indices into the proper location
        for idxt in reversed(indices):
            self.indices.insert(pos, copy.deepcopy(idxt))
        
        # change leading dimension accordingly
        for i in range(1,len(indices)):
            self.ld.insert(pos+i, self.ld[pos+i-1] *
                    self.indices[pos+i-1].size)


    #TODO: deprecated!
    def resize(self, indices, BLOCKING_SIZE):
        size = 1
        i = 0
        while size < BLOCKING_SIZE:
            pos = self.getPos(indices[i])
            if( size * self.indices[pos].size > BLOCKING_SIZE ):
                print "ERROR: this function should not do anything anymore (since the splitting and resizing happens earlier)"
                exit(-1)
                if(BLOCKING_SIZE % size != 0):
                    print "ERROR: size is not divisible by %d"%BLOCKING_SIZE 
                    for idx in indices:
                        print str(idx)
                    exit(-1)

                self.indices[pos].size = BLOCKING_SIZE / size
            size *= self.indices[pos].size
            i+=1

    #split index idx into two indices 
    def split(self, idxPos, size):
        idx = self.indices[idxPos]
        if( self.indices[idxPos].size % size != 0 ):
            print "ERROR: cannot split index %s because its size is not divisible by %d."%(idx.label, size)
            exit(-1)
        
        newIdx = index(idx.label + "_1", idx.size/size)
        self.indices[idxPos].label += "_0"
        self.indices[idxPos].size = size

        self.indices.insert(idxPos+1, newIdx) #add new index
        self.ld.insert(idxPos + 1, self.getLd(idx) * size) #update its ld

    def myPrint(self):
        ids = ""
        for id in self.indices:
            ids += id.label 
            if( id != self.indices[-1] ):
                ids += ","

        size = []
        for i in self.indices:
            size.append(i.size)

        print self.label + "_" + ids, size, self.ld

    def __ne__(self, other):
        return self.label != other.label
    def __eq__(self, other):
        return self.label == other.label

    def getPos(self, index):
        pos = 0
        for i in range(len(self.indices)):
            idx = self.indices[i]
            if(idx == index):
                return pos
            pos += 1
        return -1

    def setLd(self, ld):
        if( len(self.indices) != len(ld) ):
            print "ERROR: leading dimension: dimension of tensors does not match."
            exit(-1)
        self.ld = copy.deepcopy(ld)

    def getLd(self, index):
        pos = 0
        for i in range(len(self.indices)):
            idx = self.indices[i]
            if(idx == index):
                return self.ld[pos]
            pos += 1
        return -1

    def getLdVariable(self, index):
        ret = ""
        for idx in self.indices: # traverse from left to right
            if( idx == index ):
                if(ret[-2:] == "* "):
                    return ret[0:-2]
                else:
                    return ret
            ret += "%s_upper * "%idx.label

    def getOffset(self, indices, fixedSize = 1):
        ret = ""
        for i in range(len(indices)):
            idx = indices[i]
            pos = 0
            #find position of idx in self.indices
            for idxA in self.indices:
                if( idxA == idx ):
                    break
                pos += 1

            if( pos >= len(self.indices) ): #not found
                continue

            if( pos == 0):
                ret += "%s"%idx.label
            else:
                if( fixedSize == 1 ):
                    ret += "%s * %d"%(idx.label, self.ld[pos])
                else:
                    ret += "%s * (%s)"%(idx.label, self.getLdVariable(idx))
            if( i != len(indices) -1 ):
                ret += " + "
        if( ret == "" ):
            return "0"
        else:
            if( ret[-3:] == " + " ):
                return ret[:-3]
            else:
                return ret

    def getOffsetIdx(self, idx, fixedSize = 1):
        offset = self.getOffset([idx],fixedSize)
        offset = offset.replace("%s * "%idx.label, "")
        return offset

    def __str__(self, clean = 1):
        string = "%s"%self.label
        string += "("
        for idx in self.indices:
            string += str(idx)
            string += ","
        string = string[:-1]
        string += ")"
        if( clean == 0 ):
            string += " ld("
            for idx in self.ld:
                string += str(idx)
                string += ","
            string = string[:-1]
            string += ")"
        return str(string)


