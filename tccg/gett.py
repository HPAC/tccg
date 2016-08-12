# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy
import re
from register import Register
from arch import *
from tensor import *
from tccg_util import *
import os
import traceback


OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

class Gett:
    def __init__(self, A, B, C, alpha, beta,
                 numThreads, arch, floatType, 
                 maxImplementations, useDynamicMemory, fast, generateOnly ):

        self.generateOnly = generateOnly
        self.useDynamicMemory = useDynamicMemory  
        self.maxImplementations = maxImplementations
        self.implementations = {}
        self.indent = "   "
        self.numImpl = 0
        self.alpha = alpha
        self.beta = beta
        self.numThreads = numThreads
        self.fastMeasurements = fast # setting this variable to one removes the two outermost loops around the macro kernel, thus it speedsup measuring significantly but it yields wrong results (only during the code-gen phase)

        self.floatType = floatType
        self.floatSize = tccg_util.getFloatTypeSize(self.floatType)

        self.unrollFactor = 2

        self.arch = arch

        self.loopIndices = getLoopIndices(A, B, C)
        if( len(self.loopIndices) > 0 ):
            print FAIL + "ERROR: loop indices are not yet supported by GETT" + ENDC
            exit(-1)
        for idx in self.loopIndices:
            if( idx == A.indices[0] or idx == B.indices[0] or idx == C.indices[0] ):
                print FAIL + "ERROR: loop-indices (i.e., indices that appear in all tensors) are not allowed to be the stride-1 index of any input." + ENDC
                exit(-1)

        # encode the loop-order and the position of the packing routines. (Leftmost == outermost loop)
        self.gemmVariants = [ 
                ['n','k','B','m','A',"kernel_MC_NC_NC1",'C',"}m","}k","}n"], # panel-matrix
                ['m','k','A','n','B',"kernel_NC_MC_NC1",'C',"}n","}k","}m"]  # matrix-panel
                ] 
        # this variant doesn't seem to be efficient (uncomment, if desired)
        # self.gemmVariants.append(['m','n','k','A','B',"kernel_NC_MC_NC1","}k",'C',"}n","}m"]) # panel-panel

        ##########################
        # rewrite tensor contraction in terms of GETT
        #
        # i.e., find m-, n- and k-dim indices (similar to a typical GEMM)
        ##########################
        self.mInd = getMindices(A, B, C) # paper: equivalent to $I_m$ 
        self.nInd = getNindices(A, B, C) # paper: equivalent to $I_n$ 
        self.kInd = getContractedIndices(A, B, C) # paper: equivalent to $I_k$ 

        # store tensors
        self.C = copy.deepcopy(C)
        if( A.hasIndex(self.nInd[0])):    #if A uses an index of 'n' then we know that
                                              #C needs a transpose (i.e., the small GEMM will be of size NC x KC \times MC x KC = NC x MC
            self.A = copy.deepcopy(B)
            self.B = copy.deepcopy(A)
        else:
            self.A = copy.deepcopy(A)
            self.B = copy.deepcopy(B)

        ####################################
        # determine M, N and K
        ####################################
        self.sizeM = 1  # paper: equivalent to $M$ 
        for idx in self.mInd:
           self.sizeM *= idx.size
        self.sizeN = 1 # paper: equivalent to $N$
        for idx in self.nInd:
           self.sizeN *= idx.size
        self.sizeK = 1 # paper: equivalent to $K$
        for idx in self.kInd:
           self.sizeK *= idx.size

    def getNumRegsA(self, mr):
        return mr / self.arch.registerSize 
    def getNumRegsB(self):
        return 1
    def getNumRegsC(self, mr, nr):
        return self.getNumRegsA(mr) * nr

    def getNumRegs(self, mr, nr):
        return self.getNumRegsA(mr) + self.getNumRegsB() + self.getNumRegsC(mr,nr)

    def getMultiplyKernelBcast(self, Cregs, indent, mc1, nc1, mr, nr):
        numRegsA = self.getNumRegsA(mr) 
        arch = self.arch
        if( self.getNumRegs(mr, nr) > arch.numRegisters):
            print WARNING + "WARNING: too many registers used: %d/%d."%(self.getNumRegs(mr, nr), arch.numRegisters) + ENDC
        if(mr % self.arch.registerSize != 0):
            print "ERROR: mr needs to be a multiple of %d"%self.arch.registerSize
            exit(-1)

        code = []

        Aregs = []
        #load A into registers
        for i in range(numRegsA):
            name = "a%d"%(i)
            reg = Register(name, self.arch.registerSize)
            define = 1
            code.append( arch.load_l1("A",(i*self.arch.registerSize),
                reg, indent, define) )
            Aregs.append(reg)


        #load B into registers
        Bregs = []
        for i in range(nr):
            name = "b%d"%i
            reg = Register(name, self.arch.registerSize)
            Bregs.append(reg)


        for j in range(nr):
            define = 1
            # broadcast B
            code.append( self.arch.broadcast("B",(j), Bregs[j], indent, define)) 
            for i in range(len(Aregs)):
                code.append( arch.fma(Aregs[i], Bregs[j], Cregs[j * numRegsA + i],
                    Cregs[j * numRegsA + i], indent) ) 

        text = ""
        for ins in code:
            text += str(ins)
        return text


    def getMicroKernel(self, mc, mc1, nc, nc1, mr, nr):
       numRegsA = self.getNumRegsA(mr) 
       level = 1
       indent = self.indent
       code = "template<int update, int kc>\n"
       code += "static void microKernel(const %s *A, const %s *B, %s *C)\n"%(self.floatType,self.floatType,self.floatType)
       code += "{\n"

       if(mr % self.arch.registerSize != 0):
           print "ERROR: mr must be divisible by %d"%self.arch.registerSize
           exit(-1)

       #initialize C to zero
       Cregs = []
       for j in range(nr):
          for i in range(numRegsA):
            name = "C%d%d"%(i * self.arch.registerSize,j)
            reg = Register(name, self.arch.registerSize)
            code += str( self.arch.setzero(reg, level*indent, 1) )
            Cregs.append( reg )
       
       if( self.unrollFactor > 1 ):
           code += "#pragma unroll (%d)\n"%(self.unrollFactor)
       code += "%sfor( int k_ = kc-1; k_ >= 0; k_-= 1 ){\n"%(level*indent)
       level += 1
       code += self.getMultiplyKernelBcast(Cregs, level*indent, mc1, nc1, mr, nr)
       code += "%sA += %d;\n"%(level*indent, mc1)
       code += "%sB += %d;\n"%(level*indent, nc1)
       level -= 1
       code += "%s} //KC\n"%(level*indent)
       code += "\n"

       #################################
       #update C
       #################################
       code += "%sif(update){\n"%(level*indent)
       level += 1

       tmpRegs = []
       for j in range(nr):
          for i in range(numRegsA):
            name = "tmp%d%d"%(i * self.arch.registerSize,j)
            reg = Register(name, self.arch.registerSize)
            tmpRegs.append(reg)
       for j in range(nr):
          for i in range(numRegsA):
             code += str(self.arch.load_l1("C", i * self.arch.registerSize + j * mc1, tmpRegs[i + j * numRegsA], level*indent, 1)) #TODO L2
             code += str( self.arch.add(tmpRegs[i + j * numRegsA], Cregs[i + j * numRegsA], Cregs[i + j * numRegsA], level*indent) ) 

       level -= 1
       code += "%s}\n"%(level*indent)
       code += "%s//update C\n"%(level*indent)
       for j in range(nr):
          for i in range(numRegsA):
             code += str(self.arch.store("C", i * self.arch.registerSize + j * mc1,
                 Cregs[i + j * numRegsA], level*indent))
       level -=1
       
       code += "}\n"
       return code

    def _getIncAndLoop(self, upperBound, offsetA, offsetB, offsetC):
       preA = ""
       preB = ""
       preC = ""
       if( len(offsetA) != 0):
           preA += " + "
       if( len(offsetB) != 0):
           preB += " + "
       if( len(offsetC) != 0):
           preC += " + "

       if( upperBound == "NC" ):
           offsetB += preB + "in_ * NC1 * KC" 
           offsetC += preC + "in_ * NC1 * MC" 
           return ("NC1", "in_", offsetA, offsetB, offsetC)
       elif( upperBound == "MC" ):
           offsetA += preA + "im_ * MC1 * KC" 
           offsetC += preC + "im_ * MC1 * NC1" 
           return ("MC1", "im_", offsetA, offsetB, offsetC)
       elif( upperBound == "NC1" ):
           offsetB += preB + "in1_ * NR" 
           offsetC += preC #+ "in1_ * MC1" 
           return ("NR", "in1_", offsetA, offsetB, offsetC)
       elif( upperBound == "MC1" ):
           offsetA += preA + "im1_ * MR" 
           offsetC += preC + "im1_ * MR"
           return ("MR", "im1_", offsetA, offsetB, offsetC)
       else:
           print "[TCC] ERROR: cannot decode upperbound:", upperBound
           exit(-1)

    #at this point A is already packed in column-major format: mc x kc 
    #and B is already packed in row-major format: nc x kc
    def getMacroKernel(self, mc, mc1, nc, nc1, variant, C_hat,
          microTileC,Atilde, Btilde,  
          mIndRemainder, nIndRemainder,transposeNameC_bz,transposeNameC,
          sizeC, ldc, ldcOut):
       indent = self.indent
       offsetA = "" 
       offsetB = "" 
       offsetC = "" 

       level = 1
       code = "template<int update, int kc>\n"
       code += "static void macroKernel(const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n"%(self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
       code += "{\n"
       code += "%s//updating %s <- %s %s\n"%(level*indent, C_hat,Atilde, Btilde)

       if(self.numThreads > 1):
          code += "#pragma omp parallel for collapse(2)\n"
       for upperBound in variant:
           (inc, loopVar, offsetA, offsetB, offsetC) = self._getIncAndLoop(upperBound, offsetA, offsetB, offsetC)
           if( loopVar == 'in1_'):
              code += "%s%s C_reg[%d] __attribute__((aligned(%d)));\n"%(level*indent, self.floatType, mc1 * nc1,4096)

           if(self.numThreads > 1 and loopVar == 'in1_'): 
               # in the parallel case this has to come later in order to resolve conflicts with collpase(2)
               code += self._declareMindices(level, indent, mIndRemainder)
               code += self._declareNindices(level, indent, nIndRemainder)
           code += "%sfor( int %s = 0; %s < (%s/%s); %s ++){\n"%(level*indent, loopVar, loopVar, upperBound, inc, loopVar)
           level += 1
           if(self.numThreads == 1): 
               if( loopVar == 'im_'):
                   code += self._declareMindices(level, indent, mIndRemainder)
               if( loopVar == 'in_'):
                   code += self._declareNindices(level, indent, nIndRemainder)

       #if( mc1 > self.mr ):
       #    code += "%sfor( int im1_ = 0; im1_ < MC1; im1_+= MR ){\n"%(level*indent)
       #    level += 1

       code += "%s// blocking for registers\n"%(level*indent)
       code += "%smicroKernel<0, kc>( &A[%s], &B[%s], &C_reg[in1_ * NR * MC1]);\n"%(level*indent,offsetA, offsetB) #TODO merge transposeC into micro-kernel
       level -= 1
       code += "%s}\n"%(level*indent)

       offsetC = C_hat.getOffset(nIndRemainder + mIndRemainder) # extract subtensor
       size_str = ""
       for s in sizeC:
           size_str += "%d, "%s
       size_str = size_str[:-2]
       code += "%sint lda[] = {"%(indent * level)
       for s in ldc:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       code += "%sint ldb[] = {"%(indent * level)
       for s in ldcOut:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       # unpack C
       code += "%s//updating %s\n"%(level*indent,microTileC)
       code += "%sif( update ){\n"%(level*indent)
       level += 1
       code += "%s%s<%s>(C_reg, &C[%s], alpha, beta, lda, ldb);\n"%(level*indent,transposeNameC, size_str, offsetC)
       level -= 1
       code += "%s}else{\n"%(level*indent)
       level += 1
       code += "%s%s<%s>(C_reg, &C[%s], alpha, lda, ldb);\n"%(level*indent,transposeNameC_bz, size_str, offsetC)
       level -= 1
       code += "%s}\n"%(level*indent)
       for upperBound in variant[1:]:
           level -= 1
           code += "%s}\n"%(level*indent)

       code += "}\n"

       return code

    def getPackIndices(self, indices, BLOCKING_SIZE):
        indPack = []
        size = 1
        i = 0
        while size < BLOCKING_SIZE:
            indPack.append(indices[i])
            size *= indices[i].size
            i+=1

        if( size % BLOCKING_SIZE != 0):
            print size, BLOCKING_SIZE
            print "TODO: MC does not divide sizeM"
            exit(-1)

        return indPack
    
    def declareVariables(self):
       level = 1
       indent = self.indent
       code = "%sconst int M_ = %d;\n"%(indent,self.sizeM)
       code += "%sconst int N_ = %d;\n"%(indent,self.sizeN)
       code += "%sconst int K_ = %d;\n"%(indent,self.sizeK)
       code += "%s%s *C_L2, *A_L2, *B_L2;\n"%(level*indent, self.floatType)
       pageSize = 4*1024 #4kb
       if( self.useDynamicMemory ):
           code += "%s%s *L2 = new %s[MC*NC + MC*KC + NC*KC];\n"%(level*indent,self.floatType,self.floatType)
       else:
           code += "%s%s L2[(MC*NC + MC*KC + NC*KC)] __attribute__((aligned(%d)));\n"%(level*indent,self.floatType,pageSize)
       code += "%sC_L2 = L2;\n"%(level*indent)
       code += "%sA_L2 = C_L2 + MC*NC;\n"%(level*indent) #TODO: alignment for each temporary array
       code += "%sB_L2 = A_L2 + MC*KC;\n"%(level*indent)
       code += "\n"
       return code

    def getHeader(self, name):
        return "void %s(%s * __restrict__ %s, %s * __restrict__ %s, %s * __restrict__ %s, const %s alpha, const %s beta)"%(
            name, self.floatType, self.A.label, self.floatType, self.B.label, self.floatType, self.C.label, self.floatType, self.floatType)

    def unpackC(self, level, indent, protectUpdateC, mInd1, nInd1, kInd1, transposeNameC, transposeNameC_bz, tensorC, size, lda, ldb):
       offsetC = tensorC.getOffset(nInd1 + mInd1) # extract subtensor

       code = "%s{ // unpack C\n"%(level*indent)
       level+=1

       size_str = ""
       for s in size:
           size_str += "%d, "%s
       size_str = size_str[:-2]
       
       code += "%sint lda[] = {"%(indent * level)
       for s in lda:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       code += "%sint ldb[] = {"%(indent * level)
       for s in ldb:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       if( self.beta != 0 ):
           if( protectUpdateC ):
               code += "%sif( ik_ == 0 )\n"%(level*indent)
               code += "   %s%s<%s>(C_L2, &C[%s], alpha, beta, lda, ldb);\n"%(level*indent,transposeNameC, size_str, offsetC)
               code += "%selse\n"%(level*indent)
               code += "   %s%s<%s>(C_L2, &C[%s], alpha, 1.0, lda, ldb);\n"%(level*indent,transposeNameC, size_str, offsetC)
           else:
               code += "%s%s<%s>(C_L2, &C[%s], alpha, beta, lda, ldb);\n"%(level*indent,transposeNameC, size_str, offsetC)
       else:
           if( protectUpdateC ):
               code += "%sif( ik_ == 0 )\n"%(level*indent)
               code += "   %s%s<%s>(C_L2, &C[%s], alpha, lda, ldb);\n"%(level*indent,transposeNameC_bz, size_str, offsetC)
               code += "%selse\n"%(level*indent)
               code += "   %s%s<%s>(C_L2, &C[%s], alpha, 1.0, lda, ldb);\n"%(level*indent,transposeNameC, size_str, offsetC)
           else:
               code += "%s%s<%s>(C_L2, &C[%s], alpha, lda, ldb);\n"%(level*indent,transposeNameC_bz, size_str, offsetC)
       level-=1
       code += "%s}\n"%(level*indent)
       return code

    def packA(self, level, indent, mInd1, nInd1, kInd1, transposeName, tensorA, size, lda, ldb):
       offsetAk = tensorA.getOffset(mInd1 + kInd1) # extract subtensor

       code = "%s{ // pack A\n"%(level*indent)
       level+=1
       size_str = ""
       for s in size:
           size_str += "%d, "%s
       size_str = size_str[:-2]
       code += "%sint lda[] = {"%(indent * level)
       for s in lda:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       code += "%sint ldb[] = {"%(indent * level)
       for s in ldb:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       code += "%s%s<%s>(&A[%s], A_L2, 1.0, lda, ldb);\n"%(level*indent,transposeName,size_str,offsetAk)
       level-=1
       code += "%s}\n"%(level*indent)
       return code


    def packB(self, level, indent, mInd1, nInd1, kInd1, transposeName, tensorB, size, lda, ldb):
       offsetBk = tensorB.getOffset(nInd1 + kInd1) # extract subtensor

       code = "%s{ // pack B\n"%(level*indent)
       level+=1
       size_str = ""
       for s in size:
           size_str += "%d, "%s
       size_str = size_str[:-2]
       code += "%sint lda[] = {"%(indent * level)
       for s in lda:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       code += "%sint ldb[] = {"%(indent * level)
       for s in ldb:
           code += "%s,"%s
       code = code[:-1] + "};\n"
       code += "%s%s<%s>(&B[%s], B_L2, 1.0, lda, ldb);\n"%(level*indent,transposeName, size_str,offsetBk)
       level-=1
       code += "%s}\n"%(level*indent)
       return code

    def _pack(self, ABC, level, indent, protectUpdateC, mInd1, nInd1, kInd1, transposeNameA, transposeNameB, transposeNameC, transposeNameC_bz, tensorA, tensorB, tensorC, sizeA, lda, ldaOut, sizeB, ldb, ldbOut, sizeC, ldc, ldcOut):
      if( ABC == 'A'):
         return self.packA(level, indent, mInd1, nInd1, kInd1, transposeNameA, tensorA, sizeA, lda, ldaOut)
      elif( ABC == 'B'):
         return self.packB(level, indent, mInd1, nInd1, kInd1, transposeNameB, tensorB, sizeB, ldb, ldbOut)
      elif( ABC == 'C'):
         return ""
      #   return self.unpackC( level, indent, protectUpdateC, mInd1, nInd1, kInd1, transposeNameC, transposeNameC_bz, tensorC, sizeC, ldc, ldcOut)
      else:
         print FAIL + "ERROR: packing cannot be decoded.", ABC +ENDC
         exit(-1)

    def _declareIndices(self, mnk, level, indent, mInd1, nInd1, kInd1):
      """ Converts loop indices into tensor indices """
      if( mnk == 'm'):
         return self._declareMindices(level, indent, mInd1)
      elif( mnk == 'n'):
         return self._declareNindices(level, indent, nInd1)
      elif( mnk == 'k'):
         return self._declareKindices(level, indent, kInd1)
      else:
         print FAIL + "ERROR: loop cannot be decoded." +ENDC
         exit(-1)

    def _declareKindices(self, level, indent, kInd1):
       ###############################
       # K: determine all tensor indices that are affected by 'ik' (this index iterates along the k-dimension
       ###############################
       code = ""
       if( len(kInd1) > 0):
          code += "%sconst int %s = ik_ %% %d;\n"%(level*indent,kInd1[0].label, kInd1[0].size)
          if( len(kInd1) > 1):
              code += "%sint tmpIdx = ik_ / %d;\n"%(level*indent, kInd1[0].size)
          for i in range(1,len(kInd1)):
              idx = kInd1[i]
              code += "%sconst int %s = tmpIdx %% %d;\n"%(level*indent, idx.label, idx.size)
              if( i != len(kInd1) -1 ):
                  code += "%stmpIdx = tmpIdx / %d;\n"%(level*indent,idx.size)
       return code

    def _declareNindices(self, level, indent, nInd1):
       ###############################
       # N: determine all tensor indices that are affected by 'in' (this index iterates along the n-dimension
       ###############################
       code = ""
       if( len(nInd1) > 0):
           code += "%sconst int %s = in_ %% %d;\n"%(level*indent,nInd1[0].label, nInd1[0].size)
           if( len(nInd1) > 1):
               code += "%sint tmpIdx = in_ / %d;\n"%(level*indent,nInd1[0].size)
           for i in range(1,len(nInd1)):
               idx = nInd1[i]
               code += "%sconst int %s = tmpIdx %% %d;\n"%(level*indent,idx.label, idx.size)
               if( i != len(nInd1) -1 ):
                   code += "%stmpIdx = tmpIdx / %d;\n"%(level*indent,idx.size)
       return code

    def _declareMindices(self, level, indent, mInd1):
       ###############################
       # M: determine all tensor indices that are affected by 'im' (this index iterates along the m-dimension
       ###############################
       code = ""
       if( len(mInd1) > 0):
           code += "%sconst int %s = im_ %% %d ;\n"%(level*indent,mInd1[0].label, mInd1[0].size)
           if( len(mInd1) > 1):
               code += "%sint tmpIdx = im_ / %d;\n"%(level*indent, mInd1[0].size)
           for i in range(1,len(mInd1)):
               idx = mInd1[i]
               code += "%sconst int %s = tmpIdx %% %d;\n"%(level*indent, idx.label, idx.size)
               if( i != len(mInd1) -1 ):
                   code += "%stmpIdx = tmpIdx / %d;\n"%(level*indent,idx.size)
       return code


    def generateLoop(self, mnk,level, indent):
      if( mnk == 'm'):
         code = "%sfor( int im_ = 0; im_ < (M_ / MC); ++im_ )\n%s{\n"%(level*indent,level*indent)
      elif( mnk == 'n'):
         code = "%sfor( int in_ = 0; in_ < (N_ / NC); ++in_ )\n%s{\n"%(level*indent,level*indent)
      elif( mnk == 'k'):
         code = "%sfor( int ik_ = 0; ik_ < (K_ / KC); ++ik_ )\n%s{\n"%(level*indent,level*indent)
      else:
         print FAIL + "ERROR: loop cannot be decoded." +ENDC
         exit(-1)

      return code

    def decodeVariant(self, variant, mInd0, mInd1, mIndRemainder, nInd0, nInd1, nIndRemainder, kInd0, kInd1,
          transposeNameA, transposeNameB, transposeNameC, transposeNameC_bz,
          tensorA, tensorB, tensorC, kc, sizeA, lda, ldaOut, sizeB, ldb,
          ldbOut, sizeC, ldc, ldcOut, Ahat, Bhat, Chat ):
        code = ""
        level = 1
        indent = self.indent

        for i in range(len(variant)):
            token = variant[i]

            if ( token[0] == "A" or token[0] == "B" or token[0] == "C" ):     # PACKING
                posK = variant.index("}k")
                posC = variant.index("C")
                protectUpdateC = posK > posC
                code += self._pack(token[0], level, indent, protectUpdateC,
                      mInd1, nInd1, kInd1, transposeNameA, transposeNameB,
                      transposeNameC, transposeNameC_bz, tensorA, tensorB,
                      tensorC, sizeA, lda, ldaOut, sizeB, ldb, ldbOut, sizeC,
                      ldc, ldcOut)

            elif ( token[0:6] == "kernel" ): # MacroKernel
                posK = variant.index("}k")
                posC = variant.index("C")
                offsetC = tensorC.getOffset(mInd1 + nInd1) # extract subtensor, this select the portion of C which corresponds to the macro-tile of C
                code += "%s//%s <- %s %s\n"%(level*indent,Chat, Ahat, Bhat)
                code += "%sif( ik_ == 0 )\n"%(level*indent)
                code += "%sif( beta == 0 )\n"%((level+1)*indent)
                code += "%smacroKernel<0, %d>(A_L2, B_L2, &C[%s],alpha, beta);\n"%((level+2)*indent, kc,offsetC)
                code += "%selse\n"%((level+1)*indent)
                code += "%smacroKernel<1, %d>(A_L2, B_L2, &C[%s],alpha, beta);\n"%((level+2)*indent, kc,offsetC)
                code += "%selse\n"%(level*indent)
                code += "%s   macroKernel<1, %d>(A_L2, B_L2, &C[%s],alpha, 1.0);\n"%(level*indent, kc,offsetC)

            elif ( token[0] == "}" and len(token) == 2 ): #closing braces
                if( self.fastMeasurements and i == len(variant)-1 ):
                    code += "%sreturn;\n"%(level*indent)
                level -=1
                code += "%s} //%s\n"%(level*indent, token[1])

            else:                       # LOOPS
                code += self.generateLoop(token ,level, indent) 
                level+=1
                code += self._declareIndices(token, level, indent, mInd1, nInd1, kInd1)

        if( self.useDynamicMemory ):
            code += "%sdelete[] L2;\n"%(indent)
        code += "} //gett\n"

        return code

    def getFastFlopsFromGettName(self, gettName):
        tokens = gettName.split('_')
        mc = int(tokens[2][2:])
        nc = int(tokens[3][2:])
        kc = int(tokens[4][2:])
        flops = 2 * mc * nc * kc
        if( int(tokens[1][3:]) == 0): #pack C
            flops *= self.sizeK / kc * self.sizeN / nc
        elif( int(tokens[1][3:]) == 1): #pack B
            flops *= self.sizeM / mc * self.sizeK / kc
        elif( int(tokens[1][3:]) == 2): #pack A
            flops *= self.sizeN / nc * self.sizeK / kc
        return flops

    def decodeName(self, variant):
        tokens = variant.split("_")

        loopVar = int(tokens[1][3:])
        mc = int(tokens[2][2:])
        nc = int(tokens[3][2:])
        kc = int(tokens[4][2:])
        mc1 = int(tokens[5][3:])
        nc1 = int(tokens[6][3:])
        mr = int(tokens[7][2:])
        nr = int(tokens[8][2:])
        indices = ""
        for i in range(9,len(tokens)):
            indices += tokens[i] + "_"
        indices = indices[:-1]
        return (loopVar, mc, nc, kc, mc1, nc1, mr, nr, indices)

    def getName(self,key):
        return "gett_var%d_mc%d_nc%d_kc%d_mc1%d_nc1%d_mr%d_nr%d_%s"%(key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],key[8])
    
    def getFlopsPerFMA(self):
        if( self.floatType == "float" or self.floatType == "double"):
            return 2
        elif( self.floatType.find("omplex") != -1 ):
            return 8
        else:
            print "ERROR float type unknown"
            exit(-1)

    def estimateGFLOPS(self, mc, nc, kc, mc1, nc1, mr, nr, permA, permB, permC, variant):
        # estimate transpose overhead

        numIter = {}
        numIter['m'] = self.sizeM / mc
        numIter['n'] = self.sizeN / nc
        numIter['k'] = self.sizeK / kc
        size = {}
        size['A'] = mc * kc
        size['B'] = nc * kc
        size['C'] = mc * nc
        perm = {}
        perm ['A'] = permA
        perm ['B'] = permB
        perm ['C'] = permC

        ##################################
        # estimate packing time
        ##################################
        packingTime = 0
        for tensor in ['A','B','C']:
            openLoops = []
            # get all loops surrounding 'tensor'
            for token in variant:
                if(token == tensor):
                    break
                if( token.startswith("}") ):
                    openLoops.remove(token[1])
                elif( token == 'm' or token == 'n' or token == 'k' ):
                    openLoops.append(token)

            bytesMoved = size[tensor] * float(self.floatSize)
            for it in openLoops:
                bytesMoved *= numIter[it]

            bandwidth = self.arch.axpyBandwidth * (1 - (len(perm) - 1) * 0.015 ) # slightly favor smaller dimensional transpositions
            packingTimeTensor= bytesMoved / bandwidth
            if( perm[tensor][0] != 0 ): # we assume that transpositions for which the first index does not change are more efficient
                packingTimeTensor = bytesMoved / (bandwidth * 0.71) 

            if( tensor == 'C' and self.beta != 0 ):
                packingTime += 3 * packingTimeTensor 
            else:
                packingTime += 2 * packingTimeTensor 

        ##################################
        # decrease GEMM perf if we exceed the cache size
        ##################################
        gemmEfficiency = 0.95 # we assume that we can reach xx% of peak (once everything is packed)
        ###### L2 ######
        if( variant[0] == 'm' ): # pack B in L2
            # stream A through L2, assume we have both the current and old copy in L2
            if( (nc * kc + 2 * mc1 * kc) * self.floatSize > ((self.arch.L2_ASSOCIATIVITY - 1.0)/self.arch.L2_ASSOCIATIVITY) * self.arch.L2_SIZE ):
                gemmEfficiency *= 0.85
        else:                    # pack A in L2
            # stream B through L2, assume we have both the current and old copy in L2
            if( (mc * kc + 2 * nc1 * kc) * self.floatSize > ((self.arch.L2_ASSOCIATIVITY - 1.0)/self.arch.L2_ASSOCIATIVITY) * self.arch.L2_SIZE ):
                gemmEfficiency *= 0.85

        ###### L1 ######
        outerLoop = "" # of macro kernel
        for token in variant:
            if token.startswith("kernel"):
                outerLoop = token.split("_")[1]
                break
        copiesA = 1
        copiesB = 1
        if( outerLoop == "MC" ):
            copiesA = 2 # stream A through L1, keep two copies
        else:
            copiesB = 2 # stream B through L1, keep two copies
        reqL1 = (copiesB * nc1 * kc + copiesA * mc1 * kc) * self.floatSize
        if( reqL1 > ((self.arch.L1_ASSOCIATIVITY - 1.0)/self.arch.L1_ASSOCIATIVITY) * self.arch.L1_SIZE ):
            gemmEfficiency *= 0.85
        ###############

        ##################################
        # estimate GEMM time
        ##################################
        flopsTotal = self.sizeM * self.sizeN * self.sizeK * self.getFlopsPerFMA()
        GEMMTime   = flopsTotal / 1e9 / (self.arch.getPeakGFLOPS() * gemmEfficiency)

        estimatedGflops = self.arch.getPeakGFLOPS() * GEMMTime / (packingTime + GEMMTime )
        return estimatedGflops 

    def getPossibleValues(self, blockSize, mustDivide, targetValue, maxNumValues):
        allowedValues = []
        a = targetValue / blockSize
        while( len(allowedValues) < maxNumValues and a >= 1):
            while ( mustDivide % (a * blockSize) != 0 and a > 1):
                a -= 1
            if( a * blockSize > 0 and mustDivide % (a * blockSize) == 0):
                allowedValues.append(a * blockSize)
            a -= 1 
        return allowedValues

    def genCode(self, fastestKey = ()):
       ##############################################
       # This function generates all versions, unless 'fastestKey' is provided.
       #
       # split index sets $I_m, I_n$ and $I_k$ if necessary into 
       # $I_m^0, I_m^1, I_n^0, I_n^1, I_k^1$ and $I_k^1$
       ##############################################

       if( self.maxImplementations == 0 or self.arch.architectureName == "cuda"):
           return

       codeHpp = ""
       counter = 0

       numMCvalues = 4
       numNCvalues = 4
       numKCvalues = 4

       self.implementations = {}
       self.numImpl = 0

       transpositionToGenerate = {}

       estimatedGflops = {}
       done = {} # stores the information if a certain variant has already been generated

       for estimate_or_generate in [0,1]: # if estimate_or_generate == 0 : estimate performance
                                          # if estimate_or_generate == 1 : generate code for the best x implementations
           sortedEstimatedGflops = []
           if( estimate_or_generate == 1 ):
               for key in estimatedGflops:
                   sortedEstimatedGflops.append(estimatedGflops[key])
               sortedEstimatedGflops.sort(reverse=True)

           for variant_id in range(len(self.gemmVariants)):
               variant = self.gemmVariants[variant_id]

               # 1) search through possible mc,nc,kc values
               #TODO add search for mr, nr and mc1
               if( self.arch.architectureName == "avx2" ):
                   if( self.floatType == "float" ):
                       mr = 24
                       nr = 4
                   elif( self.floatType == "double" ):
                       mr = 12
                       nr = 4
               elif( self.arch.architectureName == "avx512" ):
                   if( self.floatType == "float" ):
                       mr = 16
                       nr = 30
                   elif( self.floatType == "double" ):
                       mr = 24
                       nr = 4
               mc1 = mr
               targetMc = 192
               if( self.floatType == "double" ):
                   targetMc /= 2
               mcValues = self.getPossibleValues(mr, self.sizeM, targetMc, 3)

               if( variant[0] == 'm' ): # make MC large such that the packing of B doesn't cost too much
                   if( self.sizeM < 4600 ):
                       mcValues = [self.sizeM]
                   else:
                       # choose mc s.t. it divides M _and_ s.t. nc is roughly 4600 (or smaller in size)
                       mcValues = self.getPossibleValues(mr, self.sizeM, 4600, 3)

               for mc in mcValues:
                   if( self.sizeM % mc != 0 or mc % mr != 0): 
                       continue
                   nc1Values = [nr] # the analysis of the performance results indicates that mr = nr is optimal in almost all cases (feel free to try different values for nc1 if you feel fit)

                   for nc1 in nc1Values:
                       if( self.sizeN % nc1 != 0 or nc1 % nr != 0): 
                           continue
                       targetNc = 192
                       if( self.floatType == "double" ):
                           targetNc /= 2
                       ncValues = self.getPossibleValues(nc1, self.sizeN, targetNc, 3)

                       if( variant[0] == 'n' ): # make NC large such that the packing of A doesn't cost too much
                           if( self.sizeN < 4600 ):
                               ncValues = [self.sizeN]
                           else:
                               # choose nc s.t. it divides N _and_ s.t. nc is roughly 4600 (or smaller in size)
                               ncValues = self.getPossibleValues(nc1, self.sizeN, 4600, 3)

                       for nc in ncValues:
                           if( self.sizeN % nc != 0 ):
                               continue
                           kcValues = self.getPossibleValues(1, self.sizeK, 256, 4)

                           for kc in kcValues:
                               if( self.sizeK % kc != 0 ):
                                   continue
               # 2) search for different permutations
                               # split indices to fit L2 cache
                               splitsAC = splitIndexSet(copy.deepcopy(self.mInd), mc, self.A, self.C)
                               for (mInd0, mInd1, tensorA, tensorC) in splitsAC: #mInd0 is if size mc
                                   splitsBC = splitIndexSet(copy.deepcopy(self.nInd), nc, self.B, tensorC)
                                   for (nInd0, nInd1, tensorB, tensorC2) in splitsBC: #nInd0 is if size nc
                                       splitsAB = splitIndexSet(copy.deepcopy(self.kInd), kc, tensorA, tensorB)
                                       for (kInd0, kInd1, tensorA2, tensorB2) in splitsAB:
                                       
                                           # Split indices again to fit L1 cache and to achieve the desired BLIS-like packing format
                                           splitsAC_L1 = splitIndexSet(copy.deepcopy(mInd0), mc1, tensorA2, tensorC2) 
                                           for (mIndL1, mIndRemainder, tensorA3, tensorC3) in splitsAC_L1:
                                               splitsBC_L1 = splitIndexSet(copy.deepcopy(nInd0), nc1, tensorB2, tensorC3)
                                               for (nIndL1, nIndRemainder, tensorB3, tensorC4) in splitsBC_L1:

               # 3) generate current candidate/implementation
                                                   mIndHat = copy.deepcopy(mIndL1 + mIndRemainder) 
                                                   nIndHat = copy.deepcopy(nIndL1 + nIndRemainder)
                                                   kIndHat = copy.deepcopy(kInd0) 
                                               
                                                   # extract Sub-tensors A and B
                                                   indAhat = mIndHat + kIndHat 
                                                   indBhat = nIndHat + kIndHat
                                                   indChat = mIndHat + nIndHat
                                                   Ahat = tensorA3.getSubTensor(indAhat) # non-packed subtensor, see paper
                                                   Bhat = tensorB3.getSubTensor(indBhat) # non-packed subtensor, see paper
                                                   Chat = tensorC4.getSubTensor(indChat) # non-packed 4D subtensor
                                                   ChatMicro = Chat.getSubTensor(mIndL1 + nIndL1) # non-packed 2D subtensor
                                               
                                                   indAtilde = mIndL1 + kIndHat + mIndRemainder
                                                   indBtilde = nIndL1 + kIndHat + nIndRemainder
                                                   indABtilde = mIndL1 + nIndL1# + mIndRemainder + nIndRemainder
                                               
                                                   Atilde  = Tensor("A~",  indAtilde)  #packed subtensor, see paper
                                                   Btilde  = Tensor("B~",  indBtilde)  #packed subtensor, see paper
                                                   ABtilde = Tensor("AB~", indABtilde) #packed subtensor, see paper
                                                   microTileC = Tensor("Cmicro",mIndL1 + nIndL1) # packed 2D subtensor
                                                                                                 # The macro-kernel iterates over Chat in steps of the microTileC
                                               
                                                   ########################################
                                                   # generate Transpositions using TTC
                                                   ########################################
                                                   permA = tccg_util.getPerm(Ahat, Atilde)
                                                   permB = tccg_util.getPerm(Bhat, Btilde)
                                                   permC = tccg_util.getPerm(ABtilde, Chat)

                                                   key = (variant_id, mc,nc,kc,mc1, nc1, mr, nr)# tensorA3.getIndexStr(), tensorB3.getIndexStr(), tensorC4.getIndexStr())
                                                   indicesStr = tensorA3.getIndexStr()+"_"+ tensorB3.getIndexStr()+"_"+ tensorC4.getIndexStr()
                                                   if( len(fastestKey) > 0 ): # fastestKey has been provided
                                                       if( variant_id != fastestKey[0]
                                                        or mc != fastestKey[1]
                                                        or nc != fastestKey[2]
                                                        or kc != fastestKey[3]
                                                        or mc1 != fastestKey[4]
                                                        or nc1 != fastestKey[5]
                                                        or mr != fastestKey[6]
                                                        or nr != fastestKey[7]
                                                        or indicesStr != fastestKey[8]):
                                                           continue
                                                   estFlops = self.estimateGFLOPS(mc, nc, kc, mc1, nc1, mr, nr, permA, permB, permC, variant)
                                                   if( estimatedGflops.has_key(key) ): 
                                                       estimatedGflops[key] = max(estimatedGflops[key], estFlops) #only consider the best-rated permutation per variant
                                                   else:
                                                       estimatedGflops[key] = estFlops
                                                   if( estFlops < estimatedGflops[key] or done.has_key(key) ): # only consider the best-rated permutation per variant, this helps to sample the 
                                                      continue                              # search space in a coarser fashion and avoids to sample very similar candidates redundantly.

                                                   maxImpl = min(self.maxImplementations-1, len(sortedEstimatedGflops)-1)
                                                   if( estimate_or_generate == 0 or estimatedGflops[key] < sortedEstimatedGflops[maxImpl] or self.numImpl >= self.maxImplementations):
                                                      continue; #skip the code generation process in the first phase (we first estimate the performance)

                                                   done[key] = 1

                                                   #print "var: %d mc: %d, nc: %d, kc: %d, nc1: %d %f"%(variant_id,mc,nc,kc,nc1, estimatedGflops )
                                               
                                                   ########################################
                                                   # generate Transpositions using TTC
                                                   ########################################
                                                   (transposeNameA, bandwidthA, permA, sizeA, lda, ldaOut) = generateTranspose(Ahat,
                                                           Atilde,
                                                           self.floatType, 1.0,
                                                           0.0, self.numThreads,
                                                           0, 1, 0,
                                                           self.arch.architectureName,
                                                           self.generateOnly, 0)
                                                   (transposeNameB, bandwidthB, permB, sizeB, ldb, ldbOut) = generateTranspose(Bhat,
                                                           Btilde,
                                                           self.floatType, 1.0,
                                                           0.0, self.numThreads,
                                                           0, 1, 0,
                                                           self.arch.architectureName,self.generateOnly, 0)
                                                   (transposeNameC, bandwidthC, permC, sizeC, ldc, ldcOut) = generateTranspose(microTileC,
                                                           ChatMicro,
                                                           self.floatType,
                                                           self.alpha, 1.0,
                                                           self.numThreads, 1,
                                                           0, 0,
                                                           self.arch.architectureName,self.generateOnly, 0)
                                                   (transposeNameC_bz, bandwidthC_bz, permC, sizeC, ldc, ldcOut) = generateTranspose(microTileC,
                                                           ChatMicro,
                                                           self.floatType,
                                                           self.alpha, 0.0,
                                                           self.numThreads, 1,
                                                           0, 0,
                                                           self.arch.architectureName,self.generateOnly, 0)

                                                   # include headers
                                                   code = "#include \"ttc_transpositions/%s.h\"\n"%transposeNameC 
                                                   code += "#include \"ttc_transpositions/%s.h\"\n"%transposeNameC_bz 
                                                   code += "#include \"ttc_transpositions/%s.h\"\n"%transposeNameA
                                                   code += "#include \"ttc_transpositions/%s.h\"\n"%transposeNameB
                                                   code += "#include <immintrin.h>\n"
                                                   code += "#include <stdlib.h>\n"
                                                   code += "#include <stdio.h>\n"

                                                   code += "#define MR (%d)\n"%mr
                                                   code += "#define NR (%d)\n"%nr
                                                   code += "#define MC1 (%d)\n"%mc1
                                                   code += "#define NC1 (%d)\n"%nc1
                                                   code += "#define MC (%d)\n"%mc
                                                   code += "#define NC (%d)\n"%nc
                                                   code += "#define KC (%d)\n"%kc
                                                   code += "\n"

                                                   ###############################
                                                   # emit micro kernels
                                                   ###############################
                                                   code += self.getMicroKernel(mc, mc1, nc, nc1, mr, nr)
                                                   code += "\n"

                                                   ###############################
                                                   # emit macro kernels
                                                   ###############################
                                                   macroVariant = ""
                                                   for token in variant:
                                                       if( token[0:6] == "kernel" ):
                                                           macroVariant = token[7:].split("_")
                                                           break
                                                   if( macroVariant == "" ):
                                                       print "[TTC] ERROR: macro variant could not be decoded."
                                                       exit(0)
                                                   code += self.getMacroKernel(mc, mc1,
                                                         nc, nc1, macroVariant,
                                                         Chat, microTileC,Atilde, Btilde,
                                                         mIndRemainder,
                                                         nIndRemainder,transposeNameC_bz,transposeNameC,
                                                         sizeC, ldc, ldcOut)
                                                   ##############################

                                                   key = (variant_id, mc,nc,kc,mc1, nc1, mr, nr, tensorA3.getIndexStr() +"_"+ tensorB3.getIndexStr()+"_"+ tensorC4.getIndexStr())
                                                   gettName = self.getName(key)
                                                   code += self.getHeader(gettName) + "\n{\n"
                                                   
                                                   tmpCode = self.declareVariables()
                                                   code += tmpCode

                                                   ###############################
                                                   # Generate M, N and K Loops as well as packing routines
                                                   ###############################
                                                   code += "   //%s <- %s %s\n"%(tensorC4, tensorA3, tensorB3)
                                                   tmpCode = self.decodeVariant(variant,
                                                         mInd0, mInd1, mIndRemainder,
                                                         nInd0, nInd1, nIndRemainder,
                                                         kInd0, kInd1,
                                                         transposeNameA,
                                                         transposeNameB,
                                                         "",
                                                         "",
                                                         tensorA3, tensorB3,
                                                         tensorC4, kc, 
                                                         sizeA, lda, ldaOut, 
                                                         sizeB, ldb, ldbOut, 
                                                         sizeC, ldc, ldcOut,
                                                         Ahat, Bhat, Chat)
                                                   code += tmpCode

                                                   ####################################
                                                   # print to file
                                                   ####################################
                                                   codeHpp += self.getHeader(gettName) + ";\n"
                                                   fgett = open("gett%d.cpp"%self.numImpl,"w")
                                                   fgett.write(code)
                                                   fgett.close()

                                                   if( self.implementations.has_key(key) ):
                                                       self.implementations[key].append((gettName,estFlops))
                                                   else:
                                                       self.implementations[key] = [(gettName,estFlops)]

                                                   counter += 1
                                                   print "%d gett versions generated so far."%counter
                                                   self.numImpl += 1

       fgett = open("gett.hpp","w")
       fgett.write(codeHpp)
       fgett.close()




