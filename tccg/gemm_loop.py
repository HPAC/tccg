# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy
from tccg_util import *

class GemmLoop:
    def __init__ (self, A, B, C, alpha, beta, numThreads, floatType, arch, batchedGEMM):
        self.arch = arch
        self.batchedGEMM = batchedGEMM
        self.A = A
        self.B = B
        self.C = C
        self.alpha = alpha
        self.beta = beta 
        self.parallelize = numThreads > 1

        self.floatType = floatType
        self.gemmName = getGemmFunctionName(floatType, arch.architectureName)
        self.batchedGemmName = getBatchedGemmFunctionName(floatType, arch.architectureName)

        self.numImpl = 0

    def getBatchedIndex(self, nonContractedLoopIndices):
        batchedIndex = nonContractedLoopIndices[0]
        for idx in nonContractedLoopIndices:
            if( idx.size > batchedIndex.size ): #largest index
               batchedIndex = idx #this will be the batched index (exposing the most parallelism)

        return batchedIndex

    def generateGemmCall(self, indentLevel, indent, opA, opB, AA, BB, beta, C, loopIndices, nonContractedLoopIndices):
        if( len(nonContractedLoopIndices) == 0 or (not self.batchedGEMM) ):
            code = ""
            if( len(nonContractedLoopIndices) > 0 ):
                code += "%s// free indices\n"%(indentLevel * indent)
            for loopIdx in nonContractedLoopIndices: 
                code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, loopIdx.label,loopIdx.label,loopIdx.label,loopIdx.label)
                indentLevel += 1

            #########################
            # single GEMM
            #########################
            loopIndices_ = copy.deepcopy(loopIndices)
            if( not self.batchedGEMM ):
                loopIndices_ = loopIndices_ + nonContractedLoopIndices
            if( self.arch.architectureName  == "cuda" ):
                code += "%s%s(cublas_handle, %s, %s, m_, n_, k_, &alpha, &%s[%s], lda_, &%s[%s], ldb_, &%s, &%s[%s], ldc_);\n"%(indentLevel * indent, self.gemmName, opA, opB, AA.label,
                    AA.getOffset(loopIndices_,0), 
                    BB.label, BB.getOffset(loopIndices_,0), 
                    beta,
                    C.label, C.getOffset(loopIndices_,0) )
            else:
                code += "%s%s(\"%s\", \"%s\", &m_, &n_, &k_, &alpha, &%s[%s], &lda_, &%s[%s], &ldb_, &%s, &%s[%s], &ldc_);\n"%(indentLevel * indent, self.gemmName, opA, opB, AA.label,
                    AA.getOffset(loopIndices_,0), 
                    BB.label, BB.getOffset(loopIndices_,0), 
                    beta,
                    C.label, C.getOffset(loopIndices_,0) )

            #closing braces
            for loopIdx in nonContractedLoopIndices:
                indentLevel -= 1
                code += "%s}\n"%(indentLevel * indent)
            return code
        else: 
            #########################
            # batched GEMM
            #########################
            code = ""
            batchedIndex = self.getBatchedIndex( nonContractedLoopIndices )
            nonContractedLoopIndices.remove(batchedIndex)

            if( len(nonContractedLoopIndices) > 0 ):
                code += "%s// free indices\n"%(indentLevel * indent)
            for loopIdx in nonContractedLoopIndices: 
                code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, loopIdx.label,loopIdx.label,loopIdx.label,loopIdx.label)
                indentLevel += 1

            # generate batched arrays
            code += "%sconst %s *Aarray[%s_upper];\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
            code += "%sconst %s *Barray[%s_upper];\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
            code += "%s%s *Carray[%s_upper];\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
            code += "%sconst %s *A_tmp = %s+(%s);\n"%(indentLevel * indent, self.floatType, AA.label, AA.getOffset(loopIndices + nonContractedLoopIndices,0))
            code += "%sconst %s *B_tmp = %s+(%s);\n"%(indentLevel * indent, self.floatType, BB.label, BB.getOffset(loopIndices + nonContractedLoopIndices,0))
            code += "%s%s *C_tmp = %s+(%s);\n"%(indentLevel * indent, self.floatType, C.label, C.getOffset(loopIndices + nonContractedLoopIndices,0))
            code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, batchedIndex.label,batchedIndex.label,batchedIndex.label,batchedIndex.label)
            indentLevel += 1
            code += "%sAarray[%s] = A_tmp+(%s);\n"%(indentLevel * indent, batchedIndex.label, AA.getOffset([batchedIndex],0))
            code += "%sBarray[%s] = B_tmp+(%s);\n"%(indentLevel * indent, batchedIndex.label, BB.getOffset([batchedIndex],0))
            code += "%sCarray[%s] = C_tmp+(%s);\n"%(indentLevel * indent, batchedIndex.label, C.getOffset([batchedIndex],0))
            indentLevel -= 1
            code += "%s}\n"%(indentLevel * indent)

            if( self.arch.architectureName == "cuda" ):
                code += "%scudaMemcpy(Aarray_d, Aarray, sizeof(%s*) * %s_upper, cudaMemcpyHostToDevice);\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
                code += "%scudaMemcpy(Barray_d, Barray, sizeof(%s*) * %s_upper, cudaMemcpyHostToDevice);\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
                code += "%scudaMemcpy(Carray_d, Carray, sizeof(%s*) * %s_upper, cudaMemcpyHostToDevice);\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
                code += "%s%s(cublas_handle, %s, %s, m_, n_, k_, &alpha, Aarray_d, lda_, Barray_d, ldb_, &%s, Carray_d, ldc_, %s_upper);\n"%(indentLevel*indent, self.batchedGemmName, opA, opB, beta, batchedIndex.label)
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
                code += "%s%s beta_array[] = {%s};\n"%(indentLevel*indent, self.floatType, beta)
                code += "%sint batch_size[] = {%s_upper};\n"%(indentLevel*indent, batchedIndex.label)
                code += "%sint grp_count = 1;\n"%(indentLevel*indent)
                code += "%s%s(transA_array, transB_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, &grp_count, batch_size);\n"%(indentLevel*indent, self.batchedGemmName)

            #closing braces
            for loopIdx in nonContractedLoopIndices:
                indentLevel -= 1
                code += "%s}\n"%(indentLevel * indent)

            return code


    def getGemm(self,gemmIndices, allIndices, contracted, A, AA, B, BB, C, indentLevel, indent,mIdx, nIdx, kIdx, alpha, beta):
        ############
        # all non-contracted indices will be delt with via a
        # batched-gemm call. Thus those dimensions don't show up
        # explicitly as loops.
        ############

        code = "%s// %s <- %s x %s;\n"%(indentLevel* indent,C, AA, BB)
        #at this point we know that we can rewrite this contractions as a series of GEMMs
        contractedLoopIndices = [] # these are the contracted indices which are looped-over
        nonContractedLoopIndices = [] #these are the free indices which are looped-over
        for idx in allIndices:
            code += "%sconst int %s_upper = %d;\n"%(indentLevel * indent, idx.label, idx.size)
            if( hasItem(gemmIndices, idx) == 0 ): #neglect thos indices which participate in the GEMM
                if( hasItem(contracted, idx) ):
                    contractedLoopIndices.append(idx)
                else:
                    nonContractedLoopIndices.append(idx)

        transA =  hasItem(contracted, AA.indices[0])
        transB =  not hasItem(contracted, BB.indices[0])
        if( transA ):
            opA = "T"
            lda = AA.getLdVariable(mIdx)
        else:
            lda = AA.getLdVariable(kIdx)
            opA = "N"
        if( transB ):
            opB = "T"
            ldb = BB.getLdVariable(kIdx)
        else:
            opB = "N"
            ldb = BB.getLdVariable(nIdx)

        if( self.arch.architectureName  == "cuda" ):
            if( opA == "T" ):
                opA = "CUBLAS_OP_T"
            else:
                opA = "CUBLAS_OP_N"
            if( opB == "T" ):
                opB = "CUBLAS_OP_T"
            else:
                opB = "CUBLAS_OP_N"

        code += "%sconst int m_ = %s_upper;\n"%(indentLevel * indent, mIdx.label)
        code += "%sconst int n_ = %s_upper;\n"%(indentLevel * indent, nIdx.label)
        code += "%sconst int k_ = %s_upper;\n"%(indentLevel * indent, kIdx.label)
        code += "%sconst int lda_ = %s;\n"%(indentLevel * indent, lda)
        code += "%sconst int ldb_ = %s;\n"%(indentLevel * indent, ldb)
        code += "%sconst int ldc_ = %s;\n"%(indentLevel * indent, C.getLdVariable(nIdx))
        code += "%sconst %s one = 1.0;\n"%(indentLevel * indent, self.floatType)

        if( self.arch.architectureName == "cuda" and len(nonContractedLoopIndices) > 0 ):
            
            code += "%sconst %s **Aarray_d, **Barray_d;\n"%(indentLevel * indent, self.floatType)
            code += "%s%s **Carray_d;\n"%(indentLevel * indent, self.floatType)
            batchedIndex = self.getBatchedIndex( nonContractedLoopIndices )
            code += "%scudaMalloc((void**) &Aarray_d, sizeof(%s*) * %s_upper);\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
            code += "%scudaMalloc((void**) &Barray_d, sizeof(%s*) * %s_upper);\n"%(indentLevel * indent, self.floatType, batchedIndex.label)
            code += "%scudaMalloc((void**) &Carray_d, sizeof(%s*) * %s_upper);\n"%(indentLevel * indent, self.floatType, batchedIndex.label)

        #########################
        # Generate the loops
        #########################
        beta = "beta"
        if( len(contractedLoopIndices) > 0): #deal with BETA
            code += self.generateGemmCall( indentLevel, indent, opA, opB, AA, BB, beta, C, [], copy.deepcopy(nonContractedLoopIndices) )
            beta = "one"

        if( len(contractedLoopIndices) > 0 ):
            code += "%s// contracted indices\n"%(indentLevel * indent)
        for loopIdx in contractedLoopIndices: 
            if( loopIdx == contractedLoopIndices[-1] ):
                code += "%sfor(int %s=1; %s < %s_upper; %s++){\n"%(indentLevel * indent, loopIdx.label,loopIdx.label,loopIdx.label,loopIdx.label)
            else:
                code += "%sfor(int %s=0; %s < %s_upper; %s++){\n"%(indentLevel * indent, loopIdx.label,loopIdx.label,loopIdx.label,loopIdx.label)
            indentLevel += 1


        code += self.generateGemmCall( indentLevel, indent, opA, opB, AA, BB, beta, C, contractedLoopIndices, copy.deepcopy(nonContractedLoopIndices)  )

        #closing braces
        for loopIdx in contractedLoopIndices:
                indentLevel -= 1
                code += "%s}\n"%(indentLevel * indent)

        if( self.arch.architectureName == "cuda" and len(nonContractedLoopIndices) > 0):
            code += "%scudaFree(Aarray_d);\n"%(indentLevel * indent)
            code += "%scudaFree(Barray_d);\n"%(indentLevel * indent)
            code += "%scudaFree(Carray_d);\n"%(indentLevel * indent)
        return code

    def skipCandidate(self, candidate, gemmSize):
        # only generate gemm-kernel with largest size; this heuristic can be refined at a later stage
        maxGemmSize = 0
        for key in gemmSize:
            maxGemmSize = max( maxGemmSize, gemmSize[key] )

        if( gemmSize[candidate] == maxGemmSize and self.numImpl == 0):
            return False
        else:
            return True

#we assume that indices have been fused beforehand
    def genCode(self):
        A = self.A
        B = self.B
        C = self.C
        alpha = self.alpha
        beta = self.beta
        code = "#include \"loopGemm.hpp\"\n"
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

        self.numImpl = 0

        gemmIndices = [A.indices[0]] #the list of indices that will participate within the GEMM call
        if( not hasItem(gemmIndices, B.indices[0]) ):
            gemmIndices.append(B.indices[0])
        if( not hasItem(gemmIndices, C.indices[0]) ):
            # at this point we know that c[0] != b[0] and c[0] != a[0]

            # it is not possible to cast this contraction as a loop over GEMMs
            # (without alowing non-stride-1 accesses) iff two (different) free
            # indices of A or B would have to participate in the GEMM
            if( ( hasItem(A.indices, C.indices[0]) and hasItem(C.indices, A.indices[0]) ) or #tests if a[0] is a free index _and_ that c[0] also belongs to a
                ( hasItem(B.indices, C.indices[0]) and hasItem(C.indices, B.indices[0]) )):
                return 0
            else:
                gemmIndices.append(C.indices[0])

        allIndices = copy.deepcopy(A.indices)
        for idx in B.indices:
            if( not hasItem(allIndices, idx)):
                allIndices.append(idx)
        for idx in C.indices:
            if( not hasItem(allIndices, idx)):
                allIndices.append(idx)

        contracted = getContractedIndices(self.A, self.B, self.C)

        indentLevel = 1
        indent = "   "
        #count number of contracted indices within GEMM
        count = 0
        kIdx = 0 #find contracted index for GEMM
        for cont in contracted:
            for idx in gemmIndices:
                if( cont == idx ):
                    kIdx = idx  #find k index for GEMM <<<<<<<<<<<<<
                    count += 1

        if( count > 1 ):
            print "We cannot rewrite this contraction as a series of GEMM calls."
            return 0

        gemmSize = {} # used for the heueristic to choose among the available candidates
                      # always choose kernel with largest m*n*k

        for estimate_or_generate in [0,1]: # if estimate_or_generate == 0 : estimate performance
                                          # if estimate_or_generate == 1 : generate code for the best x implementations
            if( count == 1 ):
                mIdx = C.indices[0] #m index for GEMM <<<<<<<<<<<<<
                AA = A
                BB = B
                if( not hasItem(A.indices, mIdx) ):
                    AA = B
                    BB = A

                nIdx = 0
                nFound = 0
                #find n index of gemm
                for idx in BB.indices:
                    if( hasItem(contracted, idx) ):
                        continue
                    if( hasItem(gemmIndices, idx) ):
                        nIdx = idx
                        nFound = 1
                        break

                #find n dindex for GEMM
                if( nFound == 0): #at this point we have more options, we can choose any of BB's indices for nIdx
                    for idx in BB.indices:
                        if( self.arch.architectureName  == "cuda" ):
                            implementation = "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n{\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                        else:
                            implementation = "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n{\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                        if( hasItem(contracted, idx) ):
                            continue
                        if( hasItem(C.indices, idx) ): #we found a possible candidate for nIdx
                            nIdx = idx  #n index for GEMM  <<<<<<<<<<<<<
                        else:
                            continue

                        gemmIndicesCopy = copy.deepcopy(gemmIndices)
                        gemmIndicesCopy.append(nIdx)
                        implementation += self.getGemm(gemmIndicesCopy, allIndices, contracted, A, AA, B, BB, C, indentLevel, indent,mIdx, nIdx, kIdx,alpha , beta)
                        implementation += "}\n"
                        
                        candidate = ""
                        candidate += str(mIdx)
                        candidate += str(nIdx)
                        candidate += str(kIdx)
                        gemmSize[candidate] = mIdx.size * nIdx.size * kIdx.size
                        if( estimate_or_generate == 0 or self.skipCandidate(candidate, gemmSize) ):
                            continue
                        code += implementation
                        if( self.arch.architectureName  == "cuda" ):
                            codeHpp += "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                        else:
                            codeHpp += "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                        self.numImpl += 1
                        print "%d loop-based versions generated so far."%self.numImpl
                else:
                    if( self.arch.architectureName  == "cuda" ):
                        implementation = "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n{\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                    else:
                        implementation = "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n{\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                    implementation += self.getGemm(gemmIndices, allIndices, contracted, A, AA, B, BB, C, indentLevel, indent,mIdx, nIdx, kIdx,alpha , beta)
                    implementation += "}\n"
                    candidate = ""
                    candidate += str(mIdx)
                    candidate += str(nIdx)
                    candidate += str(kIdx)
                    gemmSize[candidate] = mIdx.size * nIdx.size * kIdx.size
                    if( estimate_or_generate == 0 or self.skipCandidate(candidate, gemmSize) ):
                        continue
                    code += implementation
                    if( self.arch.architectureName  == "cuda" ):
                        codeHpp += "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                    else:
                        codeHpp += "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                    self.numImpl += 1
                    print "%d loop-based versions generated so far."%self.numImpl

            elif( count == 0 ): #we need to add exactly one index of contracted 
                for cont in contracted: #we evaluate all possibilities

                    if( self.arch.architectureName  == "cuda" ):
                        implementation = "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n{\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                    else:
                        implementation = "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta)\n{\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)

                    kIdx = cont
                    gemmIndicesk = copy.deepcopy(gemmIndices)
                    contractedCopy = copy.deepcopy(contracted)
                    contractedCopy.append(kIdx)
                    gemmIndicesk.append(kIdx)

                    mIdx = C.indices[0] #m index for GEMM <<<<<<<<<<<<<
                    AA = A
                    BB = B
                    if( not hasItem(A.indices, mIdx) ):
                        AA = B
                        BB = A

                    nIdx = 0
                    nFound = 0
                    #find n index of gemm
                    for idx in BB.indices:
                        if( hasItem(contractedCopy, idx) ):
                            continue
                        if( hasItem(gemmIndicesk, idx) ):
                            nFound = 1
                            nIdx = idx
                            break

                    #find n dindex for GEMM
                    if( nFound == 0): #at this point we have more choice, we can choose any of BB's indices for nIdx
                        for idx in BB.indices:
                            if( hasItem(contractedCopy, idx) ):
                                continue
                            if( hasItem(C.indices, idx) ): #we found a possible candidate for nIdx
                                nIdx = idx  #n index for GEMM  <<<<<<<<<<<<<
                            else:
                                continue

                            gemmIndicesCopy = copy.deepcopy(gemmIndicesk)
                            gemmIndicesCopy.append(nIdx)
                            implementation += self.getGemm(gemmIndicesCopy, allIndices, contractedCopy, A, AA, B, BB, C, indentLevel, indent,mIdx, nIdx, kIdx,alpha , beta)
                            implementation += "}\n"
                            candidate = ""
                            candidate += str(mIdx)
                            candidate += str(nIdx)
                            candidate += str(kIdx)
                            gemmSize[candidate] = mIdx.size * nIdx.size * kIdx.size
                            if( estimate_or_generate == 0 or self.skipCandidate(candidate, gemmSize) ):
                                continue
                            code += implementation
                            if( self.arch.architectureName  == "cuda" ):
                                codeHpp += "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                            else:
                                codeHpp += "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                            self.numImpl += 1
                            print "%d loop-based versions generated so far."%self.numImpl
                    else:
                        implementation += self.getGemm(gemmIndicesk, allIndices, contractedCopy, A, AA, B, BB, C, indentLevel, indent,mIdx, nIdx, kIdx,alpha , beta)
                        implementation += "}\n"
                        candidate = ""
                        candidate += str(mIdx)
                        candidate += str(nIdx)
                        candidate += str(kIdx)
                        gemmSize[candidate] = mIdx.size * nIdx.size * kIdx.size
                        if( estimate_or_generate == 0 or self.skipCandidate(candidate, gemmSize) ):
                            continue
                        code += implementation
                        if( self.arch.architectureName  == "cuda" ):
                            codeHpp += "void gemmLoop_%d(cublasHandle_t cublas_handle, const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                        else:
                            codeHpp += "void gemmLoop_%d(const %s *A, const %s *B, %s *C, %s alpha, %s beta);\n"%(self.numImpl,self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
                        self.numImpl += 1
                        print "%d loop-based versions generated so far."%self.numImpl

        fgett = open("loopGemm.cpp","w")
        fgett.write(code)
        fgett.close()
        fgett = open("loopGemm.hpp","w")
        fgett.write(codeHpp)
        fgett.close()

        return self.numImpl
