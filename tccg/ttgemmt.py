# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy
import traceback
import itertools
from arch import *
from tccg_util import *
from transpose import *

class TTGEMMT:
    """
    param gemm is only used for internal purposes; it is used to estimate an upper bound on performance by omitting the transpositions
    """
    def __init__ (self, A, B, C, alpha, beta, numThreads, arch, floatType, gemm,
            generateOnly, maxCandidates ):
        self.generateOnly = generateOnly
        self.arch = arch
        self.A = A
        self.B = B
        self.maxCandidates = maxCandidates
        if( not A.hasIndex(C.indices[0]) ): #adhere to the definition of mInd and nInd
           self.A = B
           self.B = A
        self.C = C
        self.alpha = alpha
        self.beta = beta 
        self.numThreads = numThreads

        self.floatType = floatType
        self.useAsGEMMName = getGemmFunctionName(floatType, floatType)

        self.candidates = {}
        self.numImpl = 0
        self.useAsGEMM = gemm

        self.loopIndices = getLoopIndices(self.A, self.B, self.C)
        for idx in self.loopIndices:
            print "ERROR: TTGEMMT: loops not supported yet."
            exit(-1)
            if( idx == A.indices[0] or idx == B.indices[0] or idx == C.indices[0] ):
                print FAIL + "ERROR: loop-indices (i.e., indices that appear in all tensors) are not allowed to be the stride-1 index of any input." + ENDC
                exit(-1)
        ##########################
        # rewrite tensor contraction in terms of GETT
        #
        # i.e., find m-, n- and k-dim indices (similar to a typical GEMM)
        ##########################
        mInd = getMindices(self.A, self.B, self.C)
        out = ""
        for idx in mInd:
            out += idx.label + ","

        nInd = getNindices(self.A, self.B, self.C)
        out = ""
        for idx in nInd:
            out += idx.label + ","

        kInd = getContractedIndices(self.A, self.B, self.C)
        out = ""
        for idx in kInd:
            out += idx.label + ","

        self.mInd = copy.deepcopy(mInd) 
        self.nInd = copy.deepcopy(nInd) 
        self.kInd = copy.deepcopy(kInd) 
        self.sizeK = 1
        for idx in self.kInd:
            self.sizeK *= idx.size
        self.sizeM = 1
        for idx in self.mInd:
            self.sizeM *= idx.size
        self.sizeN = 1
        for idx in self.nInd:
            self.sizeN *= idx.size
   
        self.sizeM_var = "1 * "
        for idx in mInd:
           self.sizeM_var += idx.label + "_upper * "
        self.sizeM_var = self.sizeM_var[:-3]
        self.sizeN_var = "1 * "
        for idx in nInd:
           self.sizeN_var += idx.label + "_upper * "
        self.sizeN_var = self.sizeN_var[:-3]
        self.sizeK_var = "1 * "
        for idx in kInd:
           self.sizeK_var += idx.label + "_upper * "
        self.sizeK_var = self.sizeK_var[:-3]

    def numCandidates(self):
        return len(self.candidates)

    def selectBestCandidate(self, transpositions):
        # get minimum cost among all candidates
        minTransposeTime = 1e100
        for candidate in transpositions:
            minTransposeTime = min(minTransposeTime , transpositions[candidate])

        goodCandidates = []
        for candidate in transpositions:
            if( transpositions[candidate] == minTransposeTime ):
                goodCandidates.append(candidate)

        goodCandidatesStr= ""
        goodCandidatesCount = 0
        for candidate in goodCandidates:
            goodCandidatesStr += candidate + ", "
            goodCandidatesCount += 1
            if( goodCandidatesCount >= self.maxCandidates ):
                break
        return goodCandidatesStr[0:-2]


    def skipCandidate(self, candidate, transpositions):
        goodCandidatesStr = self.selectBestCandidate( transpositions )
        if( goodCandidatesStr.find(candidate) == -1 ):
            return True
        else:
            return False

    def getCandidateName(self, mIndices, nIndices, kIndices, swapAB, opA, opB):
        candidate = ""
        for idx in mIndices:
            candidate += idx.label
        candidate += "_"
        for idx in nIndices:
            candidate += idx.label
        candidate += "_"
        for idx in kIndices:
            candidate += idx.label
        candidate += "_"+str(swapAB)+opA+opB # it is important that opA and opB are the last two symbols!!!
        return candidate

#we assume that indices have been fused beforehand
    def genCode(self, maxWorkspaceLimit):
        C = self.C
        alpha = self.alpha
        beta = self.beta
        if( self.useAsGEMM == 0 ):
            code = "#include \"ttgemmt.hpp\"\n"
        else:
            code = "#include \"gemm.hpp\"\n"
        code += "#include <stdlib.h>\n"
        code += "#include <hptt.h>\n"
        codeHpp = ""
        if( self.arch.architectureName  == "cuda" ):
            code += "#include <cublas_v2.h>\n"
            codeHpp += "#include <cublas_v2.h>\n"
        else:
            codeHpp = """extern \"C\"
{
void %s(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const %s *alpha, const %s *a,
            const int *lda, const %s *b, const int *ldb,
            const %s *beta, %s *c, const int *ldc);
}\n\n"""%(self.useAsGEMMName, self.floatType, self.floatType, self.floatType, self.floatType, self.floatType)

        transpositions = {}
        for estimate_or_generate in [0,1]: # if estimate_or_generate == 0 : estimate performance
                                           # if estimate_or_generate == 1 : generate code for the best x implementations
            if( estimate_or_generate ):
                print "Total amount of TTGT implementations: %d"%len(transpositions)
                print "Best TTGT candidates: %s"%self.selectBestCandidate(transpositions)
            self.numImpl = 0
            maxWork = 0
            #all these permutations are valid
            ##########################################
            # To cast the contraction as a single GEMM, we have to ensure that all m-, n- and
            # k-indices are contiguous in the participating tensors.
            ##########################################
            for mIndices in itertools.permutations(self.mInd):
                for nIndices in itertools.permutations(self.nInd):
                    for kIndices in itertools.permutations(self.kInd):
                        A = self.A
                        B = self.B
                        mIndices_ = mIndices
                        nIndices_ = nIndices
                        sizeM_var_ =  self.sizeM_var
                        sizeN_var_ =  self.sizeN_var
                        for swapAB in [0,1]:
                            if( swapAB ):
                                A = self.B
                                B = self.A
                                mIndices_ = nIndices
                                nIndices_ = mIndices
                                sizeM_var_ =  self.sizeN_var
                                sizeN_var_ =  self.sizeM_var

                            indicesC = mIndices_ + nIndices_
                            for opA in ["N","T"]:
                                if( opA == "N" ):
                                    lda = sizeM_var_
                                    indicesA = mIndices_ + kIndices
                                else:
                                    lda = self.sizeK_var
                                    indicesA = kIndices + mIndices_
                                for opB in ["N","T"]:
                                    if( opB == "N" ):
                                        ldb = self.sizeK_var
                                        indicesB = kIndices + nIndices_
                                    else:
                                        ldb = sizeN_var_
                                        indicesB = nIndices_ + kIndices

                                    candidateName = self.getCandidateName(mIndices_, nIndices_, kIndices,swapAB, opA, opB)
                                    self.candidates[candidateName] = candidateName

                                    ##########################################
                                    # generate NT case
                                    ##########################################
                                    if( self.useAsGEMM == 0 ):
                                        if( self.arch.architectureName  == "cuda" ):
                                            header = "int ttgemmt_%s(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta, %s *work_)"%(candidateName,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                                        else:
                                            header = "int ttgemmt_%s(const %s *A, const %s *B, %s *C, %s alpha, %s beta, %s *work_)"%(candidateName,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                                    else:
                                        if( self.arch.architectureName  == "cuda" ):
                                            header = "int gemm_%s(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta, %s *work_)"%(candidateName,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                                        else:
                                            header = "int gemm_%s(const %s *A, const %s *B, %s *C, %s alpha, %s beta, %s *work_)"%(candidateName,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                                    codeblocks = [] #stores all the building blocks (e.g., transpose, gemm)
                                    tmpA = A
                                    tmpB = B
                                    transposeTime = 0
                                    # Transpose A
                                    if( A.transposeRequired( indicesA ) ):
                                        transposeA = Transpose(A, indicesA,
                                                self.floatType, 1.0, 0.0,
                                                self.numThreads, self.arch,
                                                self.generateOnly, 1)
                                        codeblocks.append(transposeA)
                                        if( self.useAsGEMM == 0 ):
                                            tmpA = transposeA.OUT
                                        transposeTime += transposeA.getTransposeTime(self.arch.axpyBandwidth)
                                    # Transpose B
                                    if( B.transposeRequired( indicesB ) ):
                                        transposeB = Transpose(B, indicesB,
                                                self.floatType, 1.0, 0.0,
                                                self.numThreads, self.arch,
                                                self.generateOnly, 1)
                                        codeblocks.append(transposeB)
                                        if( self.useAsGEMM == 0 ):
                                            tmpB = transposeB.OUT
                                        transposeTime += transposeB.getTransposeTime(self.arch.axpyBandwidth)

                                    # GEMM
                                    gemm = GEMM(opA, opB, mIndices_, nIndices_, kIndices, tmpA,
                                            lda, tmpB, ldb, "0.0", self.floatType, self.arch)
                                    codeblocks.append(gemm)

                                    # Transpose C
                                    if( C.transposeRequired( indicesC ) ):
                                        transposeC = Transpose(gemm.OUT,
                                                self.C.indices, self.floatType,
                                                1.0, self.beta, self.numThreads,
                                                self.arch,self.generateOnly, 1)
                                        transposeC.renameOutput(self.C.label)
                                        transposeC.setGenericBeta()
                                        codeblocks.append(transposeC)
                                        transposeTime += transposeC.getTransposeTime(self.arch.axpyBandwidth)
                                        if( self.useAsGEMM != 0 ):
                                            gemm.setBeta("beta")
                                            gemm.renameOutput(self.C.label)
                                    else:
                                        gemm.setBeta("beta")
                                        gemm.renameOutput(self.C.label)

                                    
                                    transpositions[candidateName] = transposeTime
                                    if( estimate_or_generate == 0 or self.skipCandidate(candidateName, transpositions) ): 
                                        continue

                                    # Allocate working memory
                                    (tmpCode, workspace) = allocateMemory(codeblocks, self.floatType)
                                    if( workspace / 1024.**3 > maxWorkspaceLimit ): # only consider this candidate if it operates within the workspace limit
                                       continue

                                    # emit code
                                    implementation = ""
                                    #declare variables
                                    for idx in self.mInd + self.nInd + self.kInd:
                                        implementation += "   int %s_upper = %d;\n"%(idx.label, idx.size)
                                        
                                    #implementation = "   for(int i = 0; i < %d; i++) %s[i] = 0.0;\n"%(C.getSize(),C.label)  #TODO remove
                                    for block in codeblocks:
                                       try:
                                           (include, cpp) = block.genCode()
                                           if( self.useAsGEMM == 0 or 
                                              (self.useAsGEMM == 1 and cpp.lower().find("gemm") != -1)): #skip transpositions for GEMM
                                               code = include + code
                                               implementation += cpp
                                       except:
                                           print "ERROR in TTGT:",A, "->",tmpA, B, "->",tmpB,C, candidateName
                                           traceback.print_stack()   
                                           exit(-1)

                                    workspaceCode = "   //REQUIRED WORKSPACE: %d bytes\n"%workspace
                                    workspaceCode += "   if( work_ == NULL )\n"
                                    workspaceCode += "      return %d;\n"%workspace
                                    implementation = header + "\n{\n" + workspaceCode + tmpCode + implementation
                                    implementation += "   return 0;\n"
                                    implementation += "}\n"
                                    self.numImpl += 1
                                    if( self.useAsGEMM == 0 ):
                                        print "%d ttgt-based versions generated so far: %s.                                 "%(self.numImpl, candidateName)
                                    else:
                                        print "%d gemm-based versions generated so far: %s.                                 "%(self.numImpl, candidateName)
                                    code += "//candidate: %s\n"%candidateName + implementation
                                    codeHpp += header + ";\n"

                                    # determine maximum workspace across all implementations
                                    maxWork = max(workspace, maxWork)

        if( self.useAsGEMM == 0 ):
            fgett = open("ttgemmt.cpp","w")
            fgett.write(code)
            fgett.close()
            fgett = open("ttgemmt.hpp","w")
            fgett.write(codeHpp)
            fgett.close()
        else:
            fgett = open("gemm.cpp","w")
            fgett.write(code)
            fgett.close()
            fgett = open("gemm.hpp","w")
            fgett.write(codeHpp)
            fgett.close()

        # remove all unwanted candidates
        goodCandidatesStr = self.selectBestCandidate( transpositions )
        selectedCandidates = {}
        for candidate in self.candidates:
            if( goodCandidatesStr.find(candidate) != -1 ):
                selectedCandidates[candidate] = self.candidates[candidate]
        self.candidates = selectedCandidates 

        return (maxWork)
