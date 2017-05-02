# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy
import itertools
from tccg_util import *

class GemmLoop:
    def __init__ (self, A, B, C, alpha, beta, numThreads, floatType, arch, batchedGEMM, maxCandidates):
        self.arch = arch
        self.batchedGEMM = batchedGEMM
        self.candidates = {}
        self.maxCandidates = maxCandidates
        self.A = A
        self.B = B
        self.C = C
        self.alpha = alpha
        self.beta = beta 
        self.parallelize = numThreads > 1

        self.floatType = floatType
        self.gemmName = getGemmFunctionName(floatType, arch.architectureName)
        self.useStridedBatchedGemm = 0
        self.batchedGemmName = getBatchedGemmFunctionName(floatType,
                arch.architectureName, self.useStridedBatchedGemm )

        self.numImpl = 0

    def numCandidates(self):
        return len(self.candidates)

    def getGemmOps(self, transA, transB, mIdx, nIdx, kIdx, A, B):
        opA = "X"
        opB = "X"
        lda = -1 
        ldb = -1 

        if( transA ):
            opA = "T"
            lda = A.getLdVariable(mIdx)
        else:
            lda = A.getLdVariable(kIdx)
            opA = "N"
        if( transB ):
            opB = "T"
            ldb = B.getLdVariable(kIdx)
        else:
            opB = "N"
            ldb = B.getLdVariable(nIdx)

        if( self.arch.architectureName  == "cuda" ):
            if( opA == "T" ):
                opA = "CUBLAS_OP_T"
            else:
                opA = "CUBLAS_OP_N"
            if( opB == "T" ):
                opB = "CUBLAS_OP_T"
            else:
                opB = "CUBLAS_OP_N"

        return opA, lda, opB, ldb

    def getCandidateName(self, loopOrderFreeIndices, loopOrderContractedIndices, batchedIdx, mIdx, nIdx, kIdx):
        name = "LoG_free_"
        for idx in loopOrderFreeIndices:
            name += idx.label
        name += "_contracted_"
        for idx in loopOrderContractedIndices:
            name += idx.label
        name += "_batch_"
        name += batchedIdx.label
        name += "_"
        name += mIdx.label
        name += nIdx.label
        name += kIdx.label
        return name

    def genLoGBody(self, loopOrderFreeIndices, loopOrderContractedIndices, batchedIdx,
            mIdx, nIdx, kIdx, A, B, C):

        mIndices = intersect( A.indices, C.indices )
        nIndices = intersect( B.indices, C.indices )
        kIndices = intersect( A.indices, B.indices )
        allIndices = concatenate(mIndices, concatenate(nIndices, kIndices))

        transA =  hasItem(kIndices, A.indices[0])
        transB =  not hasItem(kIndices, B.indices[0])

        opA, lda, opB, ldb = self.getGemmOps(transA, transB, mIdx, nIdx, kIdx, A, B)

        indentLevel = 0
        indent = "   "
        candidateName = self.getCandidateName(loopOrderFreeIndices, loopOrderContractedIndices, batchedIdx, mIdx, nIdx, kIdx)
        self.candidates[candidateName] = [batchedIdx.size, mIdx.size, nIdx.size, kIdx.size]
        code = "// %s = %s * %s\n"%(str(C), str(A), str(B))
        include =  "int %s(const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(candidateName, self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
        if( self.arch.architectureName == "cuda" ):
            include =  "int %s(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(candidateName, self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
        code += "%s\n{\n"%(include[0:-2])
        indentLevel +=1

        # declare variables
        for idx in allIndices:
            code += "%sconst int %s_upper = %d;\n"%(indentLevel * indent, idx.label, idx.size)
        code += "%sconst int lda_ = %s;\n"%(indentLevel * indent, lda)
        code += "%sconst int ldb_ = %s;\n"%(indentLevel * indent, ldb)
        code += "%sconst int ldc_ = %s;\n"%(indentLevel * indent, C.getLdVariable(nIdx))
        code += "%sconst int m_ = %s_upper;\n"%(indentLevel * indent, mIdx.label)
        code += "%sconst int n_ = %s_upper;\n"%(indentLevel * indent, nIdx.label)
        code += "%sconst int k_ = %s_upper;\n"%(indentLevel * indent, kIdx.label)
        code += "%s%s beta_ = beta;\n"%(indentLevel * indent, self.floatType)

        if( self.arch.architectureName == "cuda" and batchedIdx.label != "dummy" and not self.useStridedBatchedGemm):
            code += "%sconst %s **Aarray_d, **Barray_d;\n"%(indentLevel * indent, self.floatType)
            code += "%s%s **Carray_d;\n"%(indentLevel * indent, self.floatType)
            code += "%scudaMalloc((void**) &Aarray_d, sizeof(%s*) * %s_upper);\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
            code += "%scudaMalloc((void**) &Barray_d, sizeof(%s*) * %s_upper);\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
            code += "%scudaMalloc((void**) &Carray_d, sizeof(%s*) * %s_upper);\n"%(indentLevel * indent, self.floatType, batchedIdx.label)

        # generate Loops
        for idx in loopOrderFreeIndices:
            if( idx.label != "dummy" ):
                code += "%s//free indices\n"%(indentLevel * indent)
                code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, idx.label,idx.label,idx.label,idx.label)
                indentLevel += 1 
        betaPredicate = ""
        for idx in loopOrderContractedIndices:
            if( idx.label != "dummy" ):
                code += "%s//contracted indices\n"%(indentLevel * indent)
                code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, idx.label,idx.label,idx.label,idx.label)
                betaPredicate += "%s > 0 || "%idx.label
                indentLevel += 1 
        if(len(betaPredicate) > 0):
            code += "%sif( %s )\n"%(indentLevel * indent, betaPredicate[0:-4])
            indentLevel += 1 
            code += "%sbeta_ = 1.0;\n"%(indentLevel * indent)
            indentLevel -= 1 
            code += "%selse\n"%(indentLevel * indent)
            indentLevel += 1 
            code += "%sbeta_ = beta;\n"%(indentLevel * indent)
            indentLevel -= 1 

        if( batchedIdx.label != "dummy" ): # use batched GEMM
            offsetIndices = setMinus(allIndices, [batchedIdx, mIdx, nIdx, kIdx])
            # generate batched arrays
            code += "%sconst %s *A_tmp = %s+(%s);\n"%(indentLevel * indent, self.floatType, A.label, A.getOffset(offsetIndices,0))
            code += "%sconst %s *B_tmp = %s+(%s);\n"%(indentLevel * indent, self.floatType, B.label, B.getOffset(offsetIndices,0))
            code += "%s%s *C_tmp = %s+(%s);\n"%(indentLevel * indent, self.floatType, C.label, C.getOffset(offsetIndices,0))
            if ( self.arch.architectureName != "cuda" or not self.useStridedBatchedGemm ):
                code += "%sconst %s *Aarray[%s_upper];\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
                code += "%sconst %s *Barray[%s_upper];\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
                code += "%s%s *Carray[%s_upper];\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
                code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, batchedIdx.label,batchedIdx.label,batchedIdx.label,batchedIdx.label)
                indentLevel += 1
                code += "%sAarray[%s] = A_tmp+(%s);\n"%(indentLevel * indent, batchedIdx.label, A.getOffset([batchedIdx],0))
                code += "%sBarray[%s] = B_tmp+(%s);\n"%(indentLevel * indent, batchedIdx.label, B.getOffset([batchedIdx],0))
                code += "%sCarray[%s] = C_tmp+(%s);\n"%(indentLevel * indent, batchedIdx.label, C.getOffset([batchedIdx],0))
                indentLevel -= 1
                code += "%s}\n"%(indentLevel * indent)

            if( self.arch.architectureName == "cuda" ):
                if ( self.useStridedBatchedGemm ):
                    code += "%slong long int strideA  = %s;\n"%(indentLevel * indent, A.getOffsetIdx(batchedIdx,0))
                    code += "%slong long int strideB  = %s;\n"%(indentLevel * indent, B.getOffsetIdx(batchedIdx,0))
                    code += "%slong long int strideC  = %s;\n"%(indentLevel * indent, C.getOffsetIdx(batchedIdx,0))
                    code += "%scublasStatus_t status = %s(cublas_handle, %s, %s, m_, n_, k_, &alpha, A_tmp, lda_, strideA, B_tmp, ldb_, strideB, &beta_, C_tmp, ldc_, strideC, %s_upper);\n"%(indentLevel*indent, self.batchedGemmName, opA, opB, batchedIdx.label)
                else:
                    code += "%scudaMemcpy(Aarray_d, Aarray, sizeof(%s*) * %s_upper, cudaMemcpyHostToDevice);\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
                    code += "%scudaMemcpy(Barray_d, Barray, sizeof(%s*) * %s_upper, cudaMemcpyHostToDevice);\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
                    code += "%scudaMemcpy(Carray_d, Carray, sizeof(%s*) * %s_upper, cudaMemcpyHostToDevice);\n"%(indentLevel * indent, self.floatType, batchedIdx.label)
                    code += "%scublasStatus_t status = %s(cublas_handle, %s, %s, m_, n_, k_, &alpha, Aarray_d, lda_, Barray_d, ldb_, &beta_, Carray_d, ldc_, %s_upper);\n"%(indentLevel*indent, self.batchedGemmName, opA, opB, batchedIdx.label)
                code += "%sif( status != CUBLAS_STATUS_SUCCESS ) return -1;\n"%(indentLevel*indent)
            else:
                code += "%schar transA_array[] = {'%s'};\n"%(indentLevel*indent, opA)
                code += "%schar transB_array[] = {'%s'};\n"%(indentLevel*indent, opB)
                code += "%sint m_array[] = {m_};\n"%(indentLevel*indent)
                code += "%sint n_array[] = {n_};\n"%(indentLevel*indent)
                code += "%sint k_array[] = {k_};\n"%(indentLevel*indent)
                code += "%sint lda_array[] = {lda_};\n"%(indentLevel*indent)
                code += "%sint ldb_array[] = {ldb_};\n"%(indentLevel*indent)
                code += "%sint ldc_array[] = {ldc_};\n"%(indentLevel*indent)
                code += "%s%s alpha_array[] = {alpha};\n"%(indentLevel*indent, self.floatType)
                code += "%s%s beta_array[] = {beta_};\n"%(indentLevel*indent, self.floatType)
                code += "%sint batch_size[] = {%s_upper};\n"%(indentLevel*indent, batchedIdx.label)
                code += "%sint grp_count = 1;\n"%(indentLevel*indent)
                code += "%s%s(transA_array, transB_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, &grp_count, batch_size);\n"%(indentLevel*indent, self.batchedGemmName)
        else: # use normal, non-batched GEMM
            offsetIndices = setMinus(allIndices, [mIdx, nIdx, kIdx])
            if( self.arch.architectureName == "cuda" ):
                code += "%s%s(cublas_handle, %s, %s, m_, n_, k_, &alpha, &%s[%s], lda_, &%s[%s], ldb_, &beta_, &%s[%s], ldc_);\n"%(indentLevel * indent,
                    self.gemmName, opA, opB, A.label, A.getOffset(offsetIndices,0), B.label, B.getOffset(offsetIndices,0), C.label, C.getOffset(offsetIndices,0))
            else:
                code += "%s%s(\"%s\", \"%s\", &m_, &n_, &k_, &alpha, &%s[%s], &lda_, &%s[%s], &ldb_, &beta_, &%s[%s], &ldc_);\n"%(indentLevel * indent,
                    self.gemmName, opA, opB, A.label, A.getOffset(offsetIndices,0), B.label, B.getOffset(offsetIndices,0), C.label, C.getOffset(offsetIndices,0))
        #closing braces
        while ( indentLevel > 1 ):
            indentLevel -= 1
            code += "%s}\n"%(indentLevel * indent)

        if( self.arch.architectureName == "cuda" and batchedIdx.label != "dummy" and not self.useStridedBatchedGemm):
            code += "%scudaFree(Aarray_d);\n"%(indentLevel * indent)
            code += "%scudaFree(Barray_d);\n"%(indentLevel * indent)
            code += "%scudaFree(Carray_d);\n"%(indentLevel * indent)
        code += "%sreturn 0;\n"%(indentLevel * indent)
        indentLevel -= 1
        code += "%s}\n"%(indentLevel * indent)
        return code, include


#we assume that indices have been fused beforehand
    def genCode(self):
        code = "#include \"loopOverGemm.hpp\"\n"
        codeHpp = ""
        if( self.arch.architectureName  == "cuda" ):
            code += "#include <cublas_v2.h>\n"
            code += "#include <cuda_runtime.h>\n"
            codeHpp += "#include <cublas_v2.h>\n"
        else:
            code += "#include <mkl.h>\n"
            codeHpp = """extern \"C\"
        {
        void %s(const char *transa, const char *transb,
                    const int *m, const int *n, const int *k,
                    const %s *alpha, const %s *a,
                    const int *lda, const %s *b, const int *ldb,
                    const %s *beta, %s *c, const int *ldc);
        }\n\n"""%(self.gemmName, self.floatType, self.floatType, self.floatType, self.floatType, self.floatType)


        A = self.A
        B = self.B
        C = self.C

        # this is the set of indices that _must_ participate in the GEMM
        requiredGEMMindices = concatenate([A.indices[0]], concatenate([B.indices[0]], [C.indices[0]]))

        # swap A, B such that AA is always the tensor which has the first index of C
        AA = A
        BB = B
        if( not hasItem(A.indices, C.indices[0]) ):
            AA = B
            BB = A

        # all indices of C (the so called free indices) either belong to A or B
        mIndices = intersect( AA.indices, C.indices )
        nIndices = intersect( BB.indices, C.indices )
        kIndices = intersect( AA.indices, BB.indices )

        if( len(intersect(requiredGEMMindices, mIndices)) != 1 or
            len(intersect(requiredGEMMindices, nIndices)) >= 2 or
            len(intersect(requiredGEMMindices, kIndices)) >= 2 ):
            print "LoG not possible"
            return -1

        mIdx = C.indices[0]
        nIdx = index("dummy",-1)
        if( len(intersect(requiredGEMMindices, nIndices)) == 1):
            nIdx = intersect(requiredGEMMindices, nIndices)[0]
        kIdx = index("dummy",-1)
        if( len(intersect(requiredGEMMindices, kIndices)) == 1):
            kIdx = intersect(requiredGEMMindices, kIndices)[0]

        allIndices = concatenate(mIndices, concatenate(nIndices, kIndices))

        counter = 1
        nIndicesChoices = copy.deepcopy(nIndices)
        if( nIdx.label != "dummy" ):
            nIndicesChoices = [nIdx]
        kIndicesChoices = copy.deepcopy(kIndices)
        if( kIdx.label != "dummy" ):
            kIndicesChoices = [kIdx]
           
        generatedCode = {}
        for nIdx in nIndicesChoices: # freely choose the n-index of the GEMM
           for kIdx in kIndicesChoices: # freely choose the k-index of the GEMM
              gemmIndices = [mIdx, nIdx, kIdx]
              loopIndices = setMinus(allIndices, gemmIndices)
              freeLoopIndices = setMinus( loopIndices, kIndices )
              contractedLoopIndices = intersect( loopIndices, kIndices )
              if( len(freeLoopIndices) == 0 ):
                  freeLoopIndices.append(index("dummy",-1))
              if( len(contractedLoopIndices) == 0 ):
                  contractedLoopIndices.append(index("dummy",-1))
              for batchedIdx in freeLoopIndices: # freely choose the batched index of the batched GEMMs
                  freeLoopIndicesTmp = setMinus( freeLoopIndices, [batchedIdx])

                  if( len(freeLoopIndicesTmp) == 0 ):
                      freeLoopIndicesTmp.append(index("dummy",-1))
                  for loopOrderFreeIndices in itertools.permutations(freeLoopIndicesTmp ): # freely choose any loop order
                      for loopOrderContractedIndices in itertools.permutations(contractedLoopIndices): # freely choose any loop order
                          codeTmp, include = self.genLoGBody(loopOrderFreeIndices,
                                  loopOrderContractedIndices,
                                  batchedIdx, mIdx, nIdx, kIdx, AA, BB, C)

                          candidateName = self.getCandidateName(loopOrderFreeIndices, loopOrderContractedIndices, batchedIdx, mIdx, nIdx, kIdx)
                          generatedCode[candidateName] = [codeTmp, include]
                          counter += 1

        # only generate good candidates
        goodCandidatesStr = self.selectBestCandidate()
        genCount = 0
        selectedCandidates = {}
        for candidate in self.candidates:
            if( genCount < self.maxCandidates and goodCandidatesStr.find(candidate) != -1 ):
                selectedCandidates[candidate] = self.candidates[candidate]
                code += generatedCode[candidate][0]
                codeHpp += generatedCode[candidate][1]
                genCount += 1
        self.candidates = selectedCandidates 

        fgett = open("loopOverGemm.cpp","w")
        fgett.write(code)
        fgett.close()
        fgett = open("loopOverGemm.hpp","w")
        fgett.write(codeHpp)
        fgett.close()

    def selectBestCandidate(self):
        print "Total amount of LoG implementations: %d"%len(self.candidates)
        maxGemmSize = 0
        for candidate in self.candidates:
            m_size = self.candidates[candidate][1]
            n_size = self.candidates[candidate][2]
            k_size = self.candidates[candidate][3]
            gemmSize = m_size * n_size * k_size
            maxGemmSize = max( maxGemmSize, gemmSize)

        maxBatchSize = -1
        for candidate in self.candidates:
            batcheSize = self.candidates[candidate][0]
            m_size = self.candidates[candidate][1]
            n_size = self.candidates[candidate][2]
            k_size = self.candidates[candidate][3]
            gemmSize = m_size * n_size * k_size
            if( gemmSize == maxGemmSize ):
                maxBatchSize = max(maxBatchSize, batcheSize)

        # 1) only consider candidates with maximum gemm size
        # 2) only consider candidates with maximum batch size
        goodCandidatesStr = ""
        for candidate in self.candidates:
            batchSize = self.candidates[candidate][0]
            m_size = self.candidates[candidate][1]
            n_size = self.candidates[candidate][2]
            k_size = self.candidates[candidate][3]
            gemmSize = m_size * n_size * k_size
            if( gemmSize == maxGemmSize and batchSize == maxBatchSize ):
                goodCandidatesStr += candidate + ", "
        print "Best LoG candidates: ", goodCandidatesStr[0:-2]
        return goodCandidatesStr 

