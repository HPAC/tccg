# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import copy
from tccg_util import *
from arch import *
from tensor import *

class GEMM:
    def __init__ (self,opA, opB, mInd, nInd, kInd, A, lda, B, ldb, beta_str,
            floatType, arch):
        # create new output tensor
        self.arch = arch
        self.OUT= copy.deepcopy(A)
        self.OUT.label = self.OUT.label + "_"
        self.OUT.setIndices(copy.deepcopy(mInd) + copy.deepcopy(nInd))

        self.floatType = floatType
        self.gemmName = getGemmFunctionName(floatType, arch.architectureName)
        self.opA = opA
        self.opB = opB 
        self.lda = lda
        self.ldb = ldb
        self.beta_str = beta_str
        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)

        self.sizeM = "1 * "
        for idx in mInd:
           self.sizeM += idx.label + "_upper * "
        self.sizeM = self.sizeM[:-3]
        self.sizeN = "1 * "
        for idx in nInd:
           self.sizeN += idx.label + "_upper * "
        self.sizeN = self.sizeN[:-3]
        self.sizeK = "1 * "
        for idx in kInd:
           self.sizeK += idx.label + "_upper * "
        self.sizeK = self.sizeK[:-3]

        self.dependencies = []

    def setBeta(self, beta_str):
        self.beta_str = beta_str

    def genCode(self):
        indent = "   "
        indentLevel = 1
        code = "   { // %s = GEMM(%s, %s)\n"%(str(self.OUT), str(self.A), str(self.B))
        indentLevel += 1
        code += "%sint m_ = %s;\n"%(indent * indentLevel, self.sizeM)
        code += "%sint n_ = %s;\n"%(indent * indentLevel, self.sizeN)
        code += "%sint k_ = %s;\n"%(indent * indentLevel, self.sizeK)
        code += "%sint lda_ = %s;\n"%(indent * indentLevel, self.lda)
        code += "%sint ldb_ = %s;\n"%(indent * indentLevel, self.ldb)
        code += "%sint ldc_ = %s;\n"%(indent * indentLevel, self.sizeM)
        code += "%s%s beta_ = %s;\n"%(indent * indentLevel, self.floatType, self.beta_str)
        if( self.arch.architectureName == "cuda" ):
            opA = "CUBLAS_OP_N"
            opB = "CUBLAS_OP_N"
            if( self.opA == "T" ):
                opA = "CUBLAS_OP_T"
            if( self.opB == "T" ):
                opB = "CUBLAS_OP_T"
            code += "%scublasStatus_t status = %s(cublas_handle, %s, %s, m_, n_, k_, &alpha, %s, lda_, %s, ldb_, &beta_, %s, ldc_);\n"%(indent * indentLevel, self.gemmName, opA, opB, self.A.label,self.B.label, self.OUT.label)
            code += "%sif( status != CUBLAS_STATUS_SUCCESS ) return -1;\n"%(indentLevel*indent)
        else:
            code += "%s%s(\"%s\", \"%s\", &m_, &n_, &k_, &alpha, %s, &lda_, %s, &ldb_, &beta_, %s, &ldc_);\n"%(indent * indentLevel, self.gemmName, self.opA, self.opB, self.A.label,self.B.label, self.OUT.label)
        code += "   }\n"
        include = ""
        return (include, code)

    def renameOutput(self, label):
        self.OUT.label = label

class Transpose:
    def __init__ (self, A, indices, floatType, alpha, beta, numThreads, arch, generateOnly, useStreamingStores  ):
        self.generateOnly = generateOnly
        self.arch = arch
        self.IN = copy.deepcopy(A)
        self.floatType = floatType
        self.alpha = alpha
        self.beta = beta
        self.numThreads = numThreads
        self.useGenericBeta = 0
        self.streamingStores = useStreamingStores 

        # create new output tensor
        self.OUT= copy.deepcopy(A)
        self.OUT.setIndices(copy.deepcopy(indices))
        self.OUT.label= self.OUT.label +"_"

        self.dependencies = []

    def getMovedBytes(self):
        floatSize = getFloatTypeSize(self.floatType)
        return floatSize * self.IN.getSize()

    def getTransposeTime(self, axpyBandwidth):
        perm = getPerm(self.IN, self.OUT)
        if( perm[0] != 0 ): # we assume that transpositions for which the first index does not change are more efficient
            return self.getMovedBytes() / (axpyBandwidth * 0.71)
        else:
            return self.getMovedBytes() / (axpyBandwidth)

    def renameOutput(self, label):
        self.OUT.label = label

    def setGenericBeta (self):
        self.useGenericBeta = 1

    def genCode(self):
        indent = "   "
        indentLevel = 1
        code = "   { // %s = TRANSPOSE(%s)\n"%(str(self.OUT), str(self.IN))
        indentLevel += 1

        (transposeName, bandwidth, perm, size, lda, ldb) = generateTranspose(self.IN, self.OUT, self.floatType, self.alpha,
                self.beta, self.numThreads,0,0,0,self.arch.architectureName,
                self.generateOnly, self.streamingStores  )
        size_str = ""
        for s in size:
            size_str += "%d, "%s
        size_str = size_str[:-2]
        code += "%sint lda[] = {"%(indent * indentLevel)
        for s in lda:
            code += "%s,"%s
        code = code[:-1] + "};\n"
        code += "%sint ldb[] = {"%(indent * indentLevel)
        for s in ldb:
            code += "%s,"%s
        code = code[:-1] + "};\n"
        code += "%s%s alpha_ = %f;\n"%(indent * indentLevel, self.floatType, self.alpha)
        if( self.beta != 0 ):

            if( self.useGenericBeta ):
                code += "%s%s beta_ = beta;\n"%(indent * indentLevel, self.floatType)
            else:
                code += "%s%s beta_ = %f;\n"%(indent * indentLevel, self.floatType, self.beta)
            if( self.arch.architectureName == "cuda" ):
                code += "%sint size[] = { %s };\n"%(indent * indentLevel, size_str)
                code += "%s%s(%s, %s, alpha_, beta_, size, lda, ldb);\n"%(indentLevel*indent,transposeName, self.IN.label, self.OUT.label)
            else:
                code += "%s%s<%s>(%s, %s, alpha_, beta_, lda, ldb);\n"%(indentLevel*indent,transposeName, size_str, self.IN.label, self.OUT.label)
        else:
            if( self.arch.architectureName == "cuda" ):
                code += "%sint size[] = { %s };\n"%(indent * indentLevel, size_str)
                code += "%s%s(%s, %s, alpha_, size, lda, ldb);\n"%(indentLevel*indent,transposeName, self.IN.label, self.OUT.label)
            else:
                code += "%s%s<%s>(%s, %s, alpha_, lda, ldb);\n"%(indentLevel*indent,transposeName, size_str, self.IN.label, self.OUT.label)

        code += "   }\n"
        include = "#include \"ttc_transpositions/%s.h\"\n"%transposeName 
        return (include ,code)














