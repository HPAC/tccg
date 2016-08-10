# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
from register import Register
from instructions import Instruction
import tccg_util

def maskToHex(self, mask):
    string = ""
    for s in mask:
        string += str(s)
    return hex(int(string,2))

class cuda:
    def __init__(self, floatType):
        self.architectureName = "cuda"
        self.axpyBandwidth = 150. * 2**30# B/s

        self.floatType = floatType
        self.floatSize = tccg_util.getFloatSize(floatType)


class avx2:
    def __init__(self, floatType):
        self.architectureName = "avx2"
        self.axpyBandwidth = 13. * 2**30# B/s

        self.floatType = floatType
        self.floatSize = tccg_util.getFloatSize(floatType)
        self.registerSize = 256 / 8 / self.floatSize

        self.numRegisters = 16
        self.verbose = 1

        self.frequency = 2.5
        self.numFMAcycle = 2
        self.L1_SIZE = 32* 1024.
        self.L2_SIZE = 256* 1024.

        self.L1_LATENCY = 7
        self.L1_ASSOCIATIVITY = 8
        self.L2_ASSOCIATIVITY = 8

        self.packedPostfix = "ps"
        self.scalarPostfix = "ss"
        self.registerType = "__m256"
        if( self.floatType == "double" ):
            self.packedPostfix = "pd"
            self.scalarPostfix = "sd"
            self.registerType = "__m256d"

    def getPeakGFLOPS(self):
        return self.frequency * self.numFMAcycle * self.registerSize * 2

    def setzero(self,reg, indent, define):
        if( self.floatType != "float" and self.floatType != "double"):
           print "setzero not supported for selected precision yet."
           exit(-1)

        if( define ):
          defineStr = "%s "%(self.registerType)
        else:
          defineStr = ""
        reg.setzero()
        return Instruction("%s%s%s = _mm256_setzero_%s();\n"%(indent, defineStr, reg.name, self.packedPostfix), 1)

    def store(self, dst , offset, reg, indent):
        if( self.floatType != "float" and self.floatType != "double"):
           print "store not supported for selected precision yet."
           exit(-1)

        return Instruction("%s_mm256_store_%s(%s + %d, %s);\n"%(indent, self.packedPostfix, dst, offset, reg.name), 0)

    def load_l1(self,src, offset, reg, indent, define): #load from L1
        if( self.floatType != "float" and self.floatType != "double"):
           print "load_l1 not supported for selected precision yet."
           exit(-1)

        if( define ):
          defineStr = "%s "%(self.registerType)
        else:
          defineStr = ""
        content = []
        for i in range(self.registerSize):
          content.append("%s%d"%(src,i+offset))

        ins = Instruction("%s%s%s = _mm256_load_%s(%s + %d);\n"%(indent,defineStr, reg.name, self.packedPostfix, src, offset), self.L1_LATENCY)
        reg.setContent(content)
        return ins

    #c = a * b
    #src needs to be a memory location
    def broadcast(self, src, offset, dst, indent, define):
        if( self.floatType != "float" and self.floatType != "double"):
           print "bcast not supported for selected precision yet."
           exit(-1)

        if( define ):
          defineStr = "%s "%(self.registerType)
        else:
          defineStr = ""
        latency = 1
        content = []
        for i in range(self.registerSize):
            content.append(src + str(offset))
        dst.setContent(content)
        return Instruction("%s%s%s = _mm256_broadcast_%s(%s + %d);\n"%(indent, defineStr, dst.name, self.scalarPostfix, src, offset), latency)
   

    #c = a * b
    def mul(self, a, b, c, indent):
        if( self.floatType != "float" and self.floatType != "double"):
           print "mul not supported for selected precision yet."
           exit(-1)

        latency = 5
        content = []
        for i in range(self.registerSize):
            content.append(a.content[i] + b.content[i])
        c.setContent(content)
        return Instruction("%s%s = _mm256_mul_%s(%s, %s);\n"%(indent, c.name, self.packedPostfix, a.name, b.name), latency)
   

    #c = a + b
    def add(self, a, b, c, indent):
        if( self.floatType != "float" and self.floatType != "double"):
           print "add not supported for selected precision yet."
           exit(-1)
           
        latency = 5
        content = []
        for i in range(self.registerSize):
            content.append(a.content[i] + b.content[i])
        c.setContent(content)
        return Instruction("%s%s = _mm256_add_%s(%s, %s);\n"%(indent, c.name, self.packedPostfix, a.name, b.name), latency)
   

    #d = a * b + c
    def fma(self, a, b, c, d, indent):
        if( self.floatType != "float" and self.floatType != "double"):
           print "fma not supported for selected precision yet."
           exit(-1)

        latency = 5
        content = []
        for i in range(self.registerSize):
            content.append(a.content[i] + b.content[i])
        d.setContent(content)
        return Instruction("%s%s = _mm256_fmadd_%s(%s, %s, %s);\n"%(indent, d.name, self.packedPostfix, a.name, b.name, c.name), latency)
   
    def duplicate(self, perm, src, dst, indent):
       if( self.floatType != "float" ):
           print "unpack not supported for selected precision yet."
           exit(-1)
       content = []
       for i in perm:
          content.append( src.content[i] )
       if(perm == [0,1,2,3,0,1,2,3]):
           ins = Instruction("%s%s = _mm256_permute2f128_%s(%s, 0x00);\n"%(indent, dst.name, self.packedPostfix, src.name), 3)

       dst.setContent(content)
       return ins

    def blend(self, mask, src1, src2, dst, indent, define):
       if( define ):
          defineStr = "%s "%(self.registerType)
       else:
          defineStr = ""
       if( self.floatType != "float" ):
           print "unpack not supported for selected precision yet."
           exit(-1)
       content = []
       verb = "  //"
       for i in range(self.registerSize):
           if(mask[i]):
               content.append(src2.content[i])
           else:
               content.append(src1.content[i])
           verb += str(content[-1])
       if( self.verbose == 0):
           verb = ""

       dst.setContent(content)
       ins = Instruction("%s%s%s = _mm256_blend_%s(%s, %s, %s);%s\n"%(indent, defineStr, dst.name, self.packedPostfix, src1.name, src2.name, self.maskToHex(mask), verb), 1)
       return ins

    def unpack_hi32(self, src1, src2, dst, indent, define):
       if( self.floatType != "float" ):
           print "unpack not supported for selected precision yet."
           exit(-1)
       content = []
       content.append(src1.content[2])
       content.append(src2.content[2])
       content.append(src1.content[3])
       content.append(src2.content[3])
       content.append(src1.content[6])
       content.append(src2.content[6])
       content.append(src1.content[7])
       content.append(src2.content[7])
       dst.setContent(content)
       if( define ):
           ins = Instruction("%s__m256 %s = _mm256_unpackhi_%s(%s,%s);\n"%(indent, dst.name, src1.name, src2.name), 1)
       else:
           ins = Instruction("%s%s = _mm256_unpackhi_%s(%s,%s);\n"%(indent, dst.name, src1.name, src2.name), 1)
       return ins

    def unpack_lo32(self, src1, src2, dst, indent, define):
       if( self.floatType != "float" ):
           print "unpack not supported for selected precision yet."
           exit(-1)
       content = []
       content.append(src1.content[0])
       content.append(src2.content[0])
       content.append(src1.content[1])
       content.append(src2.content[1])
       content.append(src1.content[4])
       content.append(src2.content[4])
       content.append(src1.content[5])
       content.append(src2.content[5])
       dst.setContent(content)
       if( define ):
           ins = Instruction("%s__m256 %s = _mm256_unpacklo_%s(%s,%s);\n"%(indent, dst.name, src1.name, src2.name), 1)
       else:
           ins = Instruction("%s%s = _mm256_unpacklo_%s(%s,%s);\n"%(indent, dst.name, src1.name, src2.name), 1)
       return ins

    def permute(self, perm, src, dst, indent, define):
       if( self.floatType != "float" ):
           print "unpack not supported for selected precision yet."
           exit(-1)
       if( define ):
          defineStr = "%s "%(self.registerType)
       else:
          defineStr = ""
       content = []
       for i in perm:
          content.append( src.content[i] )
       if(perm == [1,0,3,2,5,4,7,6]):
           ins = Instruction("%s%s%s = _mm256_permute_%s(%s, 0xB1);\n"%(indent, defineStr, dst.name, src.name), 1)
       if(perm == [2,3,0,1,6,7,4,5]):
           ins = Instruction("%s%s%s = _mm256_castpd_%s(_mm256_permute_pd( _mm256_castps_pd(%s), 0x5));\n"%(indent, defineStr, dst.name, src.name), 1)
       elif(perm == [4,5,6,7,0,1,2,3]):
           ins = Instruction("%s%s%s = _mm256_permute2f128_%s(%s, %s, 0x01);\n"%(indent, defineStr,  dst.name, src.name, src.name), 3)
          
       dst.setContent(content)
       return ins

class avx512:
    def __init__(self, floatType):
        self.architectureName = "avx512"
        self.axpyBandwidth = 13. * 2**30# B/s
        self.floatType = floatType
        self.floatSize = tccg_util.getFloatSize(floatType)
        self.registerSize = 512 / 8 / self.floatSize

        self.numRegisters = 32
        self.verbose = 1

        self.frequency = 1.3
        self.numFMAcycle = 2
        self.L1_SIZE = 32* 1024.
        self.L2_SIZE = 512 * 1024.

        self.L1_LATENCY = 7
        self.L1_ASSOCIATIVITY = 8
        self.L2_ASSOCIATIVITY = 8

        self.packedPostfix = "ps"
        self.scalarPostfix = "ss"
        self.registerType = "__m512"
        if( self.floatType == "double" ):
            self.packedPostfix = "pd"
            self.scalarPostfix = "sd"
            self.registerType = "__m512"

    def getPeakGFLOPS(self):
        return self.frequency * self.numFMAcycle * self.registerSize * 2

    def setzero(self,reg, indent, define):
        if( define ):
           defineStr = "%s "%(self.registerType)
        else:
           defineStr = ""
        reg.setzero()
        return Instruction("%s%s%s = _mm512_setzero_%s();\n"%(indent, defineStr, reg.name, self.packedPostfix), 1)

    def store(self, dst , offset, reg, indent):
       ins = Instruction("%s_mm512_store_%s(%s + %d, %s);\n"%(indent, self.packedPostfix, dst, offset, reg.name), 0)
       return ins

    def load_l1(self,src, offset, reg, indent, define): #load from L1
       if( define ):
          defineStr = "%s "%(self.registerType)
       else:
          defineStr = ""
       content = []
       for i in range(self.registerSize):
          content.append("%s%d"%(src,i+offset))

       ins = Instruction("%s%s%s = _mm512_load_%s(%s + %d);\n"%(indent, defineStr, reg.name, self.packedPostfix, src, offset), self.L1_LATENCY)
       reg.setContent(content)
       return ins

    #c = a * b
    #src needs to be a memory location
    def broadcast(self, src, offset, dst, indent, define):
        if( define ):
           defineStr = "%s "%(self.registerType)
        else:
           defineStr = ""
        latency = 1
        content = []
        for i in range(self.registerSize):
            content.append(src + str(offset))
        dst.setContent(content)
        return Instruction("%s%s%s = _mm512_set1_%s(*(%s + %d));\n"%(indent, defineStr, dst.name, self.packedPostfix, src, offset), latency)
   

    #c = a * b
    def mul(self, a, b, c, indent):
        latency = 5
        content = []
        for i in range(self.registerSize):
            content.append(a.content[i] + b.content[i])
        c.setContent(content)
        return Instruction("%s%s = _mm512_mul_%s(%s, %s);\n"%(indent, c.name, self.packedPostfix, a.name, b.name), latency)
   

    #c = a + b
    def add(self, a, b, c, indent):
        latency = 5
        content = []
        for i in range(self.registerSize):
            content.append(a.content[i] + b.content[i])
        c.setContent(content)
        return Instruction("%s%s = _mm512_add_%s(%s, %s);\n"%(indent, c.name, self.packedPostfix, a.name, b.name), latency)
   

    #d = a * b + c
    def fma(self, a, b, c, d, indent):
        latency = 5
        content = []
        for i in range(self.registerSize):
            content.append(a.content[i] + b.content[i])
        d.setContent(content)
        return Instruction("%s%s = _mm512_fmadd_%s(%s, %s, %s);\n"%(indent, d.name, self.packedPostfix, a.name, b.name, c.name), latency)

    def blend(self, mask, src1, src2, dst, indent, define):
       if( define ):
          defineStr = "%s "%(self.registerType)
       else:
          defineStr = ""
       content = []
       verb = "  //"
       for i in range(self.registerSize):
           if(mask[i]):
               content.append(src2.content[i])
           else:
               content.append(src1.content[i])
           verb += str(content[-1])
       if( self.verbose == 0):
           verb = ""

       dst.setContent(content)
       ins = Instruction("%s%s%s = _mm512_blend_%s(%s, %s, %s);%s\n"%(indent, defineStr, dst.name, self.packedPostfix, src1.name, src2.name, self.maskToHex(mask), verb), 1)
       return ins

    def unpack_hi32(self, src1, src2, dst, indent, define):
       if( define ):
          defineStr = "%s "%(self.registerType)
       else:
          defineStr = ""
  
       content = []
       for i in [0,1]:
           content.append(src1.content[i*8 + 2])
           content.append(src2.content[i*8 + 2])
           content.append(src1.content[i*8 + 3])
           content.append(src2.content[i*8 + 3])
           content.append(src1.content[i*8 + 6])
           content.append(src2.content[i*8 + 6])
           content.append(src1.content[i*8 + 7])
           content.append(src2.content[i*8 + 7])
       dst.setContent(content)
       ins = Instruction("%s%s%s = _mm512_unpackhi_%s(%s,%s);\n"%(indent, defineStr, dst.name, self.packedPostfix, src1.name, src2.name), 1)
       return ins

    def unpack_lo32(self, src1, src2, dst, indent, define):
       if( define ):
          defineStr = "%s "%(self.registerType)
       else:
          defineStr = ""
       content = []
       for i in [0,1]:
           content.append(src1.content[i*8 + 0])
           content.append(src2.content[i*8 + 0])
           content.append(src1.content[i*8 + 1])
           content.append(src2.content[i*8 + 1])
           content.append(src1.content[i*8 + 4])
           content.append(src2.content[i*8 + 4])
           content.append(src1.content[i*8 + 5])
           content.append(src2.content[i*8 + 5])
       dst.setContent(content)
       ins = Instruction("%s%s%s = _mm512_unpacklo_%s(%s,%s);\n"%(indent, defineStr, dst.name, self.packedPostfix, src1.name, src2.name), 1)
       return ins

    def permute(self, perm, src, dst, indent, define):
       print "ERROR: arch::permute() is not implemented yet."
       exit(-1)
       content = []
       for i in perm:
          content.append( src.content[i] )
       if(perm == [1,0,3,2,5,4,7,6]):
           ins = Instruction("%s%s%s = _mm256_permute_%s(%s, 0xB1);\n"%(indent, define, dst.name, self.packedPostfix, src.name), 1)
       if(perm == [2,3,0,1,6,7,4,5]):
           ins = Instruction("%s%s%s = _mm256_castpd_%s(_mm256_permute_pd( _mm256_castps_pd(%s), 0x5));\n"%(indent, define, dst.name, self.packedPostfix, src.name), 1)
       elif(perm == [4,5,6,7,0,1,2,3]):
           ins = Instruction("%s%s%s = _mm256_permute2f128_%s(%s, %s, 0x01);\n"%(indent, define,  dst.name, self.packedPostfix, src.name, src.name), 3)
          
       dst.setContent(content)
       return ins
