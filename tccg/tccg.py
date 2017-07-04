# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved

import argparse
import traceback
from gett import Gett
import sqlite3
import time
import shutil
from arch import *
import sql_util
import sys
import re
import os
import subprocess
import multiprocessing
import copy
from tccg_util import *
from tensor import *
from gemm_loop import *
from ttgemmt import *
import time
import socket

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'
DEVNULL = open(os.devnull, 'wb')

class TccgArgs:
    def __init__ (self, filename):
        self.filename = filename
        self.numThreads = 1
        self.generateOnly = 1
        self.workingDir = "./"
        self.gettOnly = 0
        self.maxWorkspace = 10000000   #in GB
        self.ignoreDatabase = 0
        self.useTimings = 0
        self.maxImplementations = 16
        self.maxImplementationsLoG = 1
        self.maxImplementationsTTGT = 1
        self.compiler = "icpc"
        self.blasLib = "-mkl"
        self.cudaLib = "-L${CUDA_ROOT}/lib64 -lcublas"
        self.cudaInclude = "-I${CUDA_ROOT}/include"
        self.cudaArch= "CUDA_ARCH=-arch=sm_30"
        self.database = "tccg.db"
        self.affinity = "compact,1"
        self.verbose = 0
        self.architecture = "avx2"
        self.floatTypeA = "float"
        self.floatTypeB = "float"
        self.floatTypeC = "float"
        self.tmpDirectory = "tmp"
        self.testing = 0
        self.useDynamicMemory = 0
        self.fastMeasurements = 0
        self.batchedGEMM = 1


    def getSettingsStr(self,escape):
       if( escape ):
           newline = "\\n"
       else:
           newline = "\n"
       ret = "-----------------SETTINGS-----------------"+newline
       ret += "Version".ljust(20)+"v0.1.1"+newline
       ret += "inputfile".ljust(20)+"%s"%self.filename+newline
       ret += "#threads".ljust(20)+"%d"%self.numThreads+newline
       if(self.compiler == "g++"):
            ret += "thread affinity".ljust(20)+"GOMP_CPU_AFFINITY=%s"%self.affinity+newline
       else:
            ret += "thread affinity".ljust(20)+"KMP_AFFINITY=%s"%self.affinity+newline
       ret += "maxImplementations".ljust(20)+"%d"%self.maxImplementations+newline
       ret += "compiler".ljust(20)+ tccg_util.getCompilerVersion(self.compiler)+newline
       ret += "architecture".ljust(20)+self.architecture+newline
       ret += "CPU".ljust(20)+"%s"%tccg_util.getCPUarchitecture()+newline
       ret += "floatType".ljust(20)+self.floatTypeA+newline
       ret += "tmp directory".ljust(20)+"${TCCG_ROOT}/tccg/"+self.tmpDirectory+newline
       ret += "hostname".ljust(20)+ socket.gethostname()+newline
       ret += "batched GEMM".ljust(20)+ "%d"%self.batchedGEMM+newline
       ret += "------------------------------------------"+newline
       return ret

class Tccg:
    def __init__ (self, tccgArgs):

        self.args = tccgArgs
        print self.args.getSettingsStr(0)

        (C,A,B,alpha,beta) = self.parseFile(tccgArgs.filename)

        self.A = A
        self.B = B
        self.C = C
        self.alpha = alpha
        self.beta = beta 

        self.floatType = tccgArgs.floatTypeA #todo mixed precision
        self.alignmentRequirement = 64

        outStr = str(C) + " = " + "%.2f"%alpha + " * " + str(A) +" "+ str(B) + " + " + "%.2f"%beta + " * " + str(C)
        self.fuseIndices()
        outStrAfter = str(C) + " = " + "%.2f"%alpha + " * " + str(A) +" "+ str(B) + " + " + "%.2f"%beta + " * " + str(C)
        if( outStr != outStrAfter ):
            print "Contration before fusing indices:"
            print outStr
            print "Contration after fusing indices:"
        print outStrAfter

        if( self.args.architecture == "avx2" ):
            arch = avx2(self.floatType)
        elif( self.args.architecture == "avx512" ):
            arch = avx512(self.floatType)
        elif( self.args.architecture == "cuda" ):
            arch = cuda(self.floatType)
        else:
            print "[GETT] Error: architecture unknown"
            exit(-1)
        self.ttgemmt = TTGEMMT(self.A, self.B, self.C, self.alpha, self.beta,
                self.args.numThreads, arch, self.floatType, 0, self.args.generateOnly, self.args.maxImplementationsTTGT)
        self.gemm = TTGEMMT(self.A, self.B, self.C, self.alpha, self.beta,
                self.args.numThreads, arch, self.floatType, 1, self.args.generateOnly, self.args.maxImplementationsTTGT) #only used for internal measurements
        self.gett = Gett(self.A, self.B, self.C, self.alpha, self.beta,
                self.args.numThreads, arch, self.floatType,
                self.args.maxImplementations, self.args.useDynamicMemory,
                self.args.fastMeasurements, self.args.generateOnly, self.args.useTimings, self.args.verbose )
        self.gemmLoop = GemmLoop(self.A, self.B, self.C, self.alpha, self.beta,
                self.args.numThreads, self.floatType, arch, self.args.batchedGEMM, self.args.maxImplementationsLoG)


        if( self.floatType == "float" or  self.floatType == "double") :
            self.fmul = 1
            self.fadd = 1
        elif( self.floatType == "complex" or self.floatType == "double complex") :
            self.fmul = 6
            self.fadd = 2

        mInd = getMindices(self.A, self.B, self.C)
        nInd = getNindices(self.A, self.B, self.C)
        kInd = getContractedIndices(self.A, self.B, self.C)

        self.sizeM = 1
        for idx in mInd:
           self.sizeM *= idx.size
        self.sizeN = 1
        for idx in nInd:
           self.sizeN *= idx.size
        self.sizeK = 1
        for idx in kInd:
           self.sizeK *= idx.size

        print "m: %d"%self.sizeM
        print "n: %d"%self.sizeN
        print "k: %d"%self.sizeK

    def emitFinalCode(self, maxGettFlops, maxLoGFlops, maxTTGTFlops, key = ()):

       directory = self.args.workingDir+"/tccg_implementations"
       if( not os.path.exists(directory) ):
            os.makedirs(directory)
       filename = ""

       if( maxGettFlops >= max(maxLoGFlops, maxTTGTFlops) ):
           # generate GETT
           self.gett.genCode(key)
           #filename = self.gett.getName(key)+".cpp"
           filename = "gett.cpp"
           shutil.copyfile("./gett0.cpp",directory+"/"+filename)
       elif( maxLoGFlops >= maxTTGTFlops ):
           # generate LoG
           self.gemmLoop.genCode()
           filename = "loopOverGemm.hpp"
           shutil.copyfile("./loopOverGemm.hpp",directory+"/"+filename)
           filename = "loopOverGemm.cpp"
           shutil.copyfile("./loopOverGemm.cpp",directory+"/"+filename)
       else:
           # generate TTGT
           workspace = self.ttgemmt.genCode(self.args.maxWorkspace)
           filename = "ttgemmt.hpp"
           shutil.copyfile("./ttgemmt.hpp",directory+"/"+filename)
           filename = "ttgemmt.cpp"
           shutil.copyfile("./ttgemmt.cpp",directory+"/"+filename)

       print "The generated code is available at:",directory+"/"+filename

    def codeGen(self):
        tccg_root = os.environ['TCCG_ROOT']

        # open/create SQL database
        connection = sqlite3.connect(tccg_root +"/tccg/" + self.args.database)
        cursor = connection.cursor()

        #create tables, if necessary
        sql_util.createTables(cursor)

        ###########################################
        # check if a solution already exists
        ###########################################
        if( self.args.ignoreDatabase == 0 ):
            measurement_id = sql_util.getMeasurementId(cursor,
                              str(self.A),
                              str(self.B),
                              str(self.C),
                              tccg_util.getSizeStr(self.A, self.B, self.C),
                              tccg_util.getCPUarchitecture(), #self.args.architecture,
                              self.args.numThreads,
                              self.args.floatTypeA,
                              self.args.floatTypeB,
                              self.args.floatTypeC,
                              self.beta)
            if( measurement_id != -1 ):
                (maxGettFlops, variant, mc, nc, kc , mc1 , nc1 , mr , nr, indices) =  sql_util.getFastestGETT(cursor, measurement_id)
                (maxLoGFlops) =  sql_util.getFastestLoop(cursor, measurement_id)
                (maxTTGTFlops) =  sql_util.getFastestTTGT(cursor, measurement_id)
                key = ()
                if(maxGettFlops != -1):
                    key = (int(variant), mc, nc, kc , mc1 , nc1 , mr , nr, indices)
                self.emitFinalCode(maxGettFlops, maxLoGFlops, maxTTGTFlops, key)
                return

        ###########################################
        # generate versions
        ###########################################
        workspace = 0
        startTime = time.time()
        if( not self.args.gettOnly ):
            if( not self.args.noTTGT ):
                workspace = self.ttgemmt.genCode(self.args.maxWorkspace)
            if( not self.args.noLoG ):
                self.gemmLoop.genCode()
        if( not self.args.noGETT ):
            self.gett.genCode()
        if( not self.args.noGEMM ):
            self.gemm.genCode(self.args.maxWorkspace)
        self.genMemoryBroker()
        self.printMain(workspace)

        if( self.args.generateOnly ):
            shutil.copyfile("../Makefile","./Makefile")
            return
        codeGenTime = time.time() - startTime

        ###########################################
        # compile versions
        ###########################################
        # copy makefile to tmp directory
        startTime = time.time()
        fr = open("../Makefile","r")
        fw = open("./Makefile","w")
        fw.write("BLAS_LIB=%s"%self.args.blasLib)
        fw.write("CUDA_LIB=%s"%self.args.cudaLib)
        fw.write("CUDA_INCLUDE=%s"%self.args.cudaInclude)
        fw.write("%s"%self.args.cudaArch)
        for l in fr:
            fw.write(l)
        fw.close()
        fr.close()

        my_env = os.environ.copy()
        my_env["CXX"] = self.args.compiler
        arch = "CPU"
        if( self.args.architecture  == "cuda"):
            arch = "GPU"
            my_env["CXX"] = "g++" 
        print OKGREEN + "[make] Compile ...                          ", ENDC
        numThreadsCompile = max(2, multiprocessing.cpu_count()/2)
        if( self.args.verbose ):
            ret = subprocess.call(["make", arch, "-j%d"%numThreadsCompile], env=my_env)
        else:
            ret = subprocess.call(["make", arch, "-j%d"%numThreadsCompile], stdout=DEVNULL, stderr=subprocess.STDOUT, env=my_env)
        if ret != 0 :
            print FAIL+"[TCC Error] compilation failed." + ENDC
            print "use '--verbose' for debugging purposes."
            raise
        compTime = time.time() - startTime

        ###########################################
        # run versions
        ###########################################

        startTime = time.time()
        #set environment variables
        my_env = os.environ.copy()
        my_env["OMP_NUM_THREADS"] = str(self.args.numThreads)
        my_env["KMP_AFFINITY"] = self.args.affinity 

        print OKGREEN + "[running] Measuring the runtime of all variants ..." + ENDC
        proc = subprocess.Popen(["./gett.exe"],stdout=subprocess.PIPE, env=my_env)

        ###########################################
        # parse output
        ###########################################
        failCount = 0
        referenceFlops = -1
        gettVersions = {}
        gettVersionsEstimated = {}
        ttgtVersions = {}
        gemmVersions = {}
        loopVersions = {}
        for line in proc.stdout:
            Line = line
            line = line.lower()
            if( line.find(":") != -1 ):
                implementationType = line.split(":")[0]
                try:
                    flops = line.split(":")[1]
                    flops = float(flops.split(" ")[1])
                except :      
                    print "some error has occurred: ", line
                    continue
                
                if( implementationType == "reference" ):
                    referenceFlops = flops
                elif( implementationType.startswith("gett") ):
                    gettVersions[implementationType] = flops
                    tokens = line.split(" ")
                    pos = tokens.index("estimated:")
                    gettVersionsEstimated[implementationType] = float(tokens[pos+1])
                elif( implementationType.startswith("loop") ):
                    loopVersions[implementationType] = flops
                elif( implementationType.startswith("ttgemmt") ):
                    ttgtVersions[implementationType] = flops
                elif( implementationType.startswith("gemm") ):
                    gemmVersions[implementationType] = flops
                else:
                    if( not implementationType.startswith("packing") ):
                        tccg_util.tccError("unknown implementation."+ implementationType )

            if( line.find("tcc_end") != -1):
                break
            if( len(line) > 2 ):
                print Line.replace('\n','')
            if( line.find("error") != -1 or line.find("fault") != -1):
                print FAIL + Line + ENDC
                failCount += 1
                break
            time.sleep(0.1)

        proc.wait()
        print "Compilation took %.2f seconds"%(compTime)
        print "Code-Generation took %.2f seconds"%(codeGenTime)
        print "Timing all candidates took %.2f seconds"%(time.time() - startTime)
        if failCount > 0 or proc.returncode != 0:
            print FAIL+"[TCC Error] runtime error. (error code: %d)"%proc.poll(), ENDC
            raise

        ###########################################
        # update Database
        ###########################################
        measurement_id = sql_util.insertIntoMeasurements(cursor,
                          str(self.A),
                          str(self.B),
                          str(self.C),
                          tccg_util.getSizeStr(self.A, self.B, self.C),
                          tccg_util.getCPUarchitecture(), #self.args.architecture,
                          self.args.numThreads,
                          tccg_util.getCompilerVersion(self.args.compiler),
                          self.args.floatTypeA,
                          self.args.floatTypeB,
                          self.args.floatTypeC,
                          referenceFlops,
                          0,
                          self.beta)

        maxFlopsLoop = 0
        bestLoopVersion = ""
        for loopVersion in loopVersions:
            sql_util.insertIntoLoop(cursor, measurement_id, loopVersions[loopVersion])
            if( maxFlopsLoop < loopVersions[loopVersion]):
                maxFlopsLoop = loopVersions[loopVersion]
                bestLoopVersion = loopVersion
        print "Best loop-based implementation (%s) attained: %.2f GFLOPS/s"%(bestLoopVersion, maxFlopsLoop)
            

        maxFlopsGEMM = 0
        if( len(gemmVersions) > 0 ):
            bestGEMMVersion = ""
            for GEMMVersion in gemmVersions:
                sql_util.insertIntoGEMM(cursor, measurement_id, gemmVersions[GEMMVersion])
                if( maxFlopsGEMM < gemmVersions[GEMMVersion]):
                    maxFlopsGEMM = gemmVersions[GEMMVersion]
                    bestGEMMVersion = GEMMVersion
            print "Best GEMM-based implementation (%s) attained: %.2f GFLOPS/s"%(bestGEMMVersion, maxFlopsGEMM)



        maxFlopsttgt = 0
        bestttgtVersion = ""
        for ttgtVersion in ttgtVersions:
            sql_util.insertIntoTTGT(cursor, measurement_id, ttgtVersions[ttgtVersion])
            if( maxFlopsttgt < ttgtVersions[ttgtVersion]):
                maxFlopsttgt = ttgtVersions[ttgtVersion]
                bestttgtVersion = ttgtVersion
        print "Best TTGT-based implementation (%s) attained: %.2f GFLOPS/s"%(bestttgtVersion, maxFlopsttgt)

        maxFlopsGETT = 0
        bestGETTVersion = ""
        bestKey = ()
        for gettVersion in gettVersions:
            (loopVar, mc, nc, kc, mc1, nc1, mr, nr, indicesStr) = self.gett.decodeName(gettVersion)
            sql_util.insertIntoGETT(cursor, 
                  loopVar,
                  gettVersions[gettVersion], 
                  gettVersionsEstimated[gettVersion],
                  mc ,
                  nc,
                  kc ,
                  mc1,
                  nc1,
                  mr,
                  nr,
                  indicesStr,
                  measurement_id)
            if( maxFlopsGETT < gettVersions[gettVersion]):
                maxFlopsGETT = gettVersions[gettVersion]
                bestGETTVersion = gettVersion
                bestKey = self.gett.decodeName(bestGETTVersion)

        self.emitFinalCode(maxFlopsGETT, maxFlopsLoop, maxFlopsttgt, bestKey)
        print "Best gett-based implementation (%s) attained: %.2f GFLOPS/s"%(bestGETTVersion, maxFlopsGETT)
        print "Best Loop/TTGT/GETT/reference/GEMM: %.2f / %.2f / %.2f / %.2f / %.2f GFLOPS"%(maxFlopsLoop, maxFlopsttgt, maxFlopsGETT,referenceFlops, maxFlopsGEMM)

        # commit changes to database
        connection.commit()
        connection.close()

    def fuseIndices(self):
        A = self.A
        B = self.B
        C = self.C

        done = []

        n = len(C.indices)
        for i in range(n):
            if( i >= n or C.indices[i] in done):
                break
            numTensorsWithSameIndex = 0

            searchThroughTensors = []
            if( A.hasIndex(C.indices[i]) ):
                numTensorsWithSameIndex += 1
                searchThroughTensors.append(A)
            if( B.hasIndex(C.indices[i]) ):
                numTensorsWithSameIndex += 1
                searchThroughTensors.append(B)

            if(numTensorsWithSameIndex == 2): #loop indices
                pos1 = searchThroughTensors[0].getPos(C.indices[i])
                pos2 = searchThroughTensors[1].getPos(C.indices[i])

                l = 1
                incSize = 1
                while( i+l < len(C.indices) and
                    pos1+l < len(searchThroughTensors[0].indices) and 
                    pos2+l < len(searchThroughTensors[1].indices) and 
                    searchThroughTensors[0].indices[pos1+l] == C.indices[i+l] and
                    searchThroughTensors[1].indices[pos2+l] == C.indices[i+l] and
                    not (C.indices[i+l] in done)):
                    incSize *= C.indices[i+l].size
                    l += 1

                if l > 1:
                    done.append(C.indices[i])
                    C.indices[i].size *= incSize
                    searchThroughTensors[0].indices[pos1].size *= incSize
                    searchThroughTensors[1].indices[pos2].size *= incSize

                while l > 1:
                    n-=1
                    C.indices.pop(i+1)
                    C.ld.pop(i+1)
                    searchThroughTensors[0].indices.pop(pos1+1)
                    searchThroughTensors[0].ld.pop(pos1+1)
                    searchThroughTensors[1].indices.pop(pos2+1)
                    searchThroughTensors[1].ld.pop(pos2+1)
                    l -= 1

        n = len(C.indices)
        for i in range(n):
            if( i >= n or C.indices[i] in done):
                continue
            numTensorsWithSameIndex = 0

            searchThroughTensors = []
            if( A.hasIndex(C.indices[i]) ):
                numTensorsWithSameIndex += 1
                searchThroughTensors.append(A)
            if( B.hasIndex(C.indices[i]) ):
                numTensorsWithSameIndex += 1
                searchThroughTensors.append(B)

            if(numTensorsWithSameIndex == 1):
                pos1 = searchThroughTensors[0].getPos(C.indices[i])

                l = 1
                incSize = 1
                while( i+l < len(C.indices) and
                    pos1+l < len(searchThroughTensors[0].indices) and 
                    searchThroughTensors[0].indices[pos1+l] == C.indices[i+l] and
                    not (C.indices[i+l] in done)):
                    incSize *= C.indices[i+l].size
                    l += 1

                if l > 1:
                    done.append(C.indices[i])
                    C.indices[i].size *= incSize
                    searchThroughTensors[0].indices[pos1].size *= incSize

                while l > 1:
                    n-=1
                    C.indices.pop(i+1)
                    C.ld.pop(i+1)
                    searchThroughTensors[0].indices.pop(pos1+1)
                    searchThroughTensors[0].ld.pop(pos1+1)
                    l -= 1

        #fuse indices within A and B
        n = len(A.indices)
        for i in range(n):
            if( i >= n or A.indices[i] in done):
                continue
            pos1 = B.getPos(A.indices[i])

            if(pos1 != -1):

                l = 1
                incSize = 1
                while( i+l < len(A.indices) and
                    pos1+l < len(B.indices) and 
                    B.indices[pos1+l] == A.indices[i+l] and
                    not (A.indices[i+l] in done)):
                    incSize *= A.indices[i+l].size
                    l += 1

                if l > 1:
                    done.append(A.indices[i])
                    A.indices[i].size *= incSize
                    B.indices[pos1].size *= incSize

                while l > 1:
                    n-=1
                    A.indices.pop(i+1)
                    A.ld.pop(i+1)
                    B.indices.pop(pos1+1)
                    B.ld.pop(pos1+1)
                    l -= 1

    def printSyntax(self):
        print "Tensors have to start with a capital letter."
        print "Underscores are not allowed for any variables."
        print "indices have to start with a lower case character."

    def parseFile(self, filename):
        if( not os.path.isfile(filename) ):
            tccg_util.tccError("File: %s does not exist."%filename)
            exit(-1)
#TODO BUGFIX FOR C = A*B + C
        #TODO this function needs more attention
        f = open(filename,"r")
        tensors = []
        alpha = 0.0
        beta = 0.0
        sizes = {}
        for l in f:
            content = l.split("#")[0] #remove comments
            content = content.replace(" ","") #remove whitespaces
            if(len(content) > 0 ):
                ttensors = re.findall("[A-Z][0-9]*\[[a-z][a-z,0-9]*[,[a-z][a-z,0-9]*]*\]", content)
                if( len(ttensors) == 3 or len(ttensors) == 4):
                    if( content.find("+=") != -1 or content.find("-=") != -1):
                        print FAIL + "Syntax error: += or -= is not allowed." + ENDC
                        print FAIL + "Please use the following syntax instead: C[...] = ... + beta * C[...]" + ENDC
                        exit(-1)
                    if( content.find("+-") != -1 or content.find("-+") != -1):
                        print FAIL + "Syntax error: +- or -+ is not allowed." + ENDC
                        exit(-1)

                    for t in ttensors:
                        found = 0
                        for t1 in tensors:
                            if t == t1:
                                found = 1
                        if( found == 0):
                            tensors.append(t)

                    #find alpha and beta
                    content1 = content.split("=")[1]
                    content2 = content.split("=")[1]
                    for t in tensors:
                        content2 = content2.replace(t,"")
                    tmp = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+",content2)

                    posAB  = min(content1.find(tensors[1]), content1.find(tensors[2]))
                    posC  = content1.find(tensors[0])

                    alpha = 1.0
                    beta = 0.0
                    if( posC != -1 ):
                        beta = 1.0
                    if(len(tmp) == 0):
                        continue
                    elif(len(tmp) == 1):
                        posFloat = content1.find(tmp[0])
                        if(posC == -1):
                            alpha = float(tmp[0])
                            beta = 0
                        elif(posAB < posC):
                            if(posFloat < posAB):
                                alpha = float(tmp[0])
                            else:
                                beta = float(tmp[0])
                    elif(len(tmp) == 2):
                        posFloat1 = content1.find(tmp[0])
                        posFloat2 = content1.find(tmp[1])

                        if(posAB < posC):
                            if(posFloat1 < posFloat2):
                                alpha = tmp[0]
                                beta = tmp[1]
                            else:
                                alpha = tmp[1]
                                beta = tmp[0]
                        else:
                            if(posFloat1 < posFloat2):
                                alpha = tmp[1]
                                beta = tmp[0]
                            else:
                                alpha = tmp[0]
                                beta = tmp[1]
                        alpha = float(alpha)
                        beta= float(beta)
                    else:
                        print FAIL + "Syntax error too many scalars specified." + ENDC
                        exit(-1)


                tsizes = re.findall("[a-z][a-z0-9]*=[0-9]+", content)
                for s in tsizes:
                    label = s.split("=")[0]
                    size = int(s.split("=")[1])
                    sizes[label] = size

        if(len(tensors) != 3):
            print FAIL + "ERROR: You must specify exactly three tensors." + ENDC
            print "We found these tensors:"
            for tensor in tensors:
                print tensor
            self.printSyntax()
            exit(-1)

        retTensors = []
        # create Tensor objects from strings
        for t in tensors:
            label = re.findall("[A-Z][0-9]*",t)[0]
            tmp = re.findall("\[[a-z][a-z,0-9]*[,[a-z][a-z,0-9]*]*\]",t)
            tmp = tmp[0][1:-1] #delte '[' and ']'
            indices = tmp.split(',')
            final_indices = []
            for idx in indices:
                if( sizes.has_key(idx) ):
                    final_indices.append( index(idx,sizes[idx]) )
                else:
                    print FAIL + "ERROR: You must specify the size of each index." + ENDC
                    exit(-1)
            retTensors.append(Tensor(label, final_indices))

        return (retTensors[0],retTensors[1],retTensors[2],alpha, beta)

    def getReferenceVersion(self):
       level = 1
       indent = "   "
       offsetA = self.A.getOffset(self.A.indices)
       offsetB = self.B.getOffset(self.B.indices)
       offsetC = self.C.getOffset(self.C.indices)
       code = "//pure loop implementation\n"
       code += "void referenceVersion(const %s * __restrict__ A, const %s *__restrict__ B, %s *__restrict__ C, %s alpha, %s beta){\n"%(self.floatType,self.floatType,self.floatType,self.floatType,self.floatType)
       if( self.args.numThreads > 1 ):
           code += "%s#pragma omp parallel for collapse(%d)\n"%(level * indent, len(self.C.indices))
       for idx in self.C.indices[::-1]:
           code += "%sfor(int %s = 0; %s < %d; %s++)\n"%(level*indent, idx.label, idx.label, idx.size, idx.label)
           level +=1

       level -=1
       code += "%s{\n"%(level*indent)
       level +=1
       code += "%s%s tmp = 0;\n"%(level*indent, self.floatType)

       kIndices = getContractedIndices(self.A, self.B, self.C)

       for idx in kIndices:
           code += "%sfor(int %s = 0; %s < %d; %s++)\n"%(level*indent, idx.label, idx.label, idx.size, idx.label)
           level +=1

       code += "%stmp += A[%s] * B[%s];\n"%(level*indent,offsetA, offsetB)
       level -=len(kIndices)

       if(self.beta == 0):
           code += "%sC[%s] = alpha * tmp;\n"%(level*indent,offsetC)
       else:
           code += "%sC[%s] = alpha * tmp + beta * C[%s];\n"%(level*indent,offsetC, offsetC)
       level -=1
       code += "%s}\n"%(level*indent)
       level -=3
       code += "%s}\n"%(level*indent)

       return code

    def getTrashCache(self, headerOnly):
        if( headerOnly ):
            return  "void trashCache(%s *A, %s *B, int n);"%(self.floatType, self.floatType)
        cppCode = "void trashCache(%s *A, %s *B, int n)\n"%(self.floatType, self.floatType)
        cppCode += "{\n"
        cppCode += "   for(int j = 0; j < 3; j++){\n"
        cppCode += "#pragma omp parallel\n"
        cppCode += "      for(int i = 0; i < n; i++)\n"
        cppCode += "         A[i] += 0.999 * B[i];\n"
        cppCode += "   }\n"
        cppCode += "}\n"

        return cppCode

    def genMemoryBroker(self):
        code = "#pragma once\n"
        code += "#include <stdlib.h>\n"
        code += "#include <stdint.h>\n"
        code += "class MemoryBroker {\n"
        code += "   public:\n"
        code += "      MemoryBroker();\n"
        code += "\n"
        code += "      void alloc( size_t size );\n"
        code += "      char* requestMemory( size_t size );\n"
        code += "      void reset();\n"
        code += "      void release();\n"
        code += "      bool isInit() const;\n"
        code += "      uint64_t size() const;\n"
        code += "\n"
        code += "   private:\n"
        code += "\n"
        code += "      char *ptr;\n"
        code += "      uint64_t totalSize;\n"
        code += "      uint64_t currentOffset;\n"
        code += "};\n"
        f = open("memoryBroker.h","w")
        f.write( code)
        f.close()

        code = "#include \"memoryBroker.h\"\n"
        code += "      MemoryBroker::MemoryBroker() : ptr(nullptr), totalSize(0), currentOffset(0) {}\n"
        code += "\n"
        code += "      void MemoryBroker::alloc( size_t size )\n"
        code += "      {\n"
        code += "         posix_memalign((void**)&(this->ptr), 4096, size);\n"
        code += "         this->totalSize = size;\n"
        code += "         this->currentOffset = 0;\n"
        code += "      }\n"
        code += "      char* MemoryBroker::requestMemory( size_t size )\n"
        code += "      {\n"
        code += "         char* ret = &ptr[currentOffset];\n"
        code += "         currentOffset += size;\n"
        code += "         return ret;\n"
        code += "      }\n"
        code += "      void MemoryBroker::reset() { this->currentOffset = 0; }\n"
        code += "\n"
        code += "      void MemoryBroker::release() {\n"
        code += "         free(this->ptr);\n"
        code += "         this->totalSize = 0;\n"
        code += "         this->currentOffset = 0;\n"
        code += "      }\n"
        code += "\n"
        code += "      bool MemoryBroker::isInit() const { return totalSize != 0; }\n"
        code += "      uint64_t MemoryBroker::size() const { return totalSize; }\n"
        f = open("memoryBroker.cpp","w")
        f.write( code)
        f.close()

    def printMain(self, workspace):
       code = ""
       if( self.gett.numImpl > 0 ):
           code += "#include \"gett.hpp\"\n"
       if( self.gemmLoop.numCandidates() > 0 ):
           code += "#include \"loopOverGemm.hpp\"\n"
       if( self.ttgemmt.numCandidates() > 0 ):
           code += "#include \"ttgemmt.hpp\"\n"
       if( self.gemm.numCandidates() > 0 ):
           code += "#include \"gemm.hpp\"\n"
       if( self.args.architecture  == "cuda"):
           code += "#include <cuda_runtime.h>\n"
           code += "#include <cublas_v2.h>\n"
       code += "#include <time.h>\n"
       code += "#include <omp.h>\n"
       code += "#include <stdlib.h>\n"
       code += "#include <unistd.h>\n"
       code += "#include <stdio.h>\n"
       code += "#include <float.h>\n"
       code += "\n"
       code += "#include \"memoryBroker.h\"\n"
       code += "MemoryBroker memBroker;\n"
       code += self.getTrashCache(1)
       
       f = open("trash.cpp","w")
       f.write( self.getTrashCache(0) )
       f.close()

       code += "\n"
       code += self.getReferenceVersion()
       code += "\n"

       code +="void restore(const %s *in, %s*out, int total_size)"%(self.floatType,self.floatType)
       code +="{\n"
       code +="   for(size_t i=0;i < total_size ; ++i){\n"
       code +="      out[i] = in[i];\n"
       code +="   }\n"
       code +="}\n"
       code += "\n"

       code +="int equal(const %s *A, const %s*B, int total_size)"%(self.floatType,self.floatType)
       code +="{\n"

       code +="  int error = 0;\n" 
       if( self.floatType.find("complex") != -1 ):
           _floatType = "float"
           if( self.floatType.find("double") != -1 ):
               _floatType = "double"
           code +="   const %s *Atmp = (%s*)A;\n"%(_floatType,_floatType)
           code +="   const %s *Btmp= (%s*)B;\n"%(_floatType,_floatType)
           code +="   #pragma omp parallel for reduction(+:error) \n"
           code +="   for(size_t i=0;i < 2*total_size ; ++i){\n"
       else:
           _floatType = self.floatType
           code +="   const %s *Atmp= A;\n"%self.floatType
           code +="   const %s *Btmp= B;\n"%self.floatType
           code +="   //#pragma omp parallel for reduction(+:error) \n"
           code +="   for(size_t i=0;i < total_size ; ++i){\n"

       code +="      %s Aabs = (Atmp[i] < 0) ? -Atmp[i] : Atmp[i];\n"%_floatType
       code +="      %s Babs = (Btmp[i] < 0) ? -Btmp[i] : Btmp[i];\n"%_floatType
       code +="      %s max = (Aabs < Babs) ? Babs : Aabs;\n"%_floatType
       code +="      %s diff = (Aabs - Babs);\n"%_floatType
       code +="      diff = (diff < 0) ? -diff : diff;\n"
       code +="      if(diff > 0){\n"
       code +="        %s relError = ((diff / max) > diff) ? diff : (diff / max); //max of relative and absolute error to avoid problems close to zero\n"%_floatType
       if( self.floatType.find("float") != -1 ):
           code +="        if(relError > 1e-4){\n"
       else:
           code +="        if(relError > 1e-9){\n"
       code +="            //printf(\"i: %l relError: %.8e\\n\",i,relError);\n"
       code +="            error += 1;\n"
       code +="            return 0;\n"
       code +="         }\n"
       code +="      }\n"
       code +="   }\n"
       #code +="   return error;\n"
       code +="   return (error > 0) ? 0 : 1;\n"
       code +="}\n"

       code += "int main(int argc, char** argv)\n"
       code += "{\n"
       code += "  printf(\"%s\");\n"%self.args.getSettingsStr(1)
       code +="   %s *A, *B, *C, *C_ref, *C_copy, *trash1, *trash2, *work_;\n"%(self.floatType)
       if( self.args.architecture == "cuda"):
           code +="   %s *A_d, *B_d, *C_d, *work_d;\n"%(self.floatType)
       code +="   int sizeA[%d];\n"%len(self.A.indices)
       code +="   int sizeB[%d];\n"%len(self.B.indices)
       code +="   int sizeC[%d];\n"%len(self.C.indices)
       sizeA = ""
       for i in range(len(self.A.indices)):
            code += "   sizeA[%d] = %d;\n"%(i,self.A.indices[i].size)
            sizeA += "sizeA[%d]"%i
            if( i != len(self.A.indices) -1 ):
                sizeA += "*"
       sizeB = ""
       for i in range(len(self.B.indices)):
            code += "   sizeB[%d] = %d;\n"%(i,self.B.indices[i].size)
            sizeB += "sizeB[%d]"%i
            if( i != len(self.B.indices) -1 ):
                sizeB += "*"
       sizeC = ""
       for i in range(len(self.C.indices)):
            code += "   sizeC[%d] = %d;\n"%(i,self.C.indices[i].size)
            sizeC += "sizeC[%d]"%i
            if( i != len(self.C.indices) -1 ):
                sizeC += "*"
       code +="   int largerThanL3 = 1024*1024*128/sizeof(%s); //128MB \n"%(self.floatType)
       code +="   int ret = posix_memalign((void**) &trash1, %d, sizeof(%s) * largerThanL3);\n"%(self.alignmentRequirement, self.floatType)
       code +="   ret += posix_memalign((void**) &work_, %d, %d);\n"%(self.alignmentRequirement, workspace)
       code +="   ret += posix_memalign((void**) &trash2, %d, sizeof(%s) * largerThanL3);\n"%(self.alignmentRequirement, self.floatType)
       code +="   ret += posix_memalign((void**) &A, %d, sizeof(%s) * %s);\n"%(self.alignmentRequirement, self.floatType, sizeA)
       code +="   ret += posix_memalign((void**) &B, %d, sizeof(%s) * %s);\n"%(self.alignmentRequirement, self.floatType, sizeB)
       code +="   ret += posix_memalign((void**) &C, %d, sizeof(%s) * %s);\n"%(self.alignmentRequirement, self.floatType, sizeC)
       code +="   ret += posix_memalign((void**) &C_ref, %d, sizeof(%s) * %s);\n"%(self.alignmentRequirement, self.floatType, sizeC)
       code +="   ret += posix_memalign((void**) &C_copy, %d, sizeof(%s) * %s);\n"%(self.alignmentRequirement, self.floatType, sizeC)
       code +="   if( ret != 0){ printf(\"[TCC] ERROR: posix_memalign failed\\n\"); exit(-1); }\n"
       if( self.args.architecture == "cuda"):
           code +="   cudaMalloc((void**) &A_d, sizeof(%s) * %s);\n"%(self.floatType, sizeA)
           code +="   cudaMalloc((void**) &B_d, sizeof(%s) * %s);\n"%(self.floatType, sizeB)
           code +="   cudaMalloc((void**) &C_d, sizeof(%s) * %s);\n"%(self.floatType, sizeC)
           code +="   cudaMalloc((void**) &work_d, %d);\n"%(workspace)
           code +="   cublasHandle_t cublas_handle;\n"
           code +="   cublasCreate(&cublas_handle);\n"

       code += "   //initialize A\n"
       code +="   #pragma omp parallel for\n"
       code += "   for(size_t i=0;i < %s; ++i)\n"%sizeA
       code += "      A[i] = ((float)(10 + ((i+1)*17)%100))/110.0;\n"
       code += "   //initialize B\n"
       code +="   #pragma omp parallel for\n"
       code += "   for(size_t i=0;i < %s; ++i)\n"%sizeB
       code += "      B[i] = ((float)(10 + ((i+2)*3)%100))/110.0;\n"
       code += "   //initialize A\n"
       code +="   #pragma omp parallel for\n"
       code += "   for(size_t i=0;i < %s; ++i){\n"%sizeC
       code += "      C[i] = ((float)(10 + ((i+3)*13)%100))/110.0;\n"
       code += "      C_ref[i] = C[i];\n"
       code += "      C_copy[i] = C[i];\n"
       code += "   };\n"
       code += "   const %s alpha = %f;\n"%(self.floatType, self.alpha)
       code += "   const %s beta = %f;\n"%(self.floatType, self.beta)
       code += "\n"
       code +="   #pragma omp parallel for\n"
       code +="   for(int i=0;i < largerThanL3; ++i){\n"
       code +="      trash1[i] = (i*1.139182);\n"
       code +="      trash2[i] = (i*1.0312912);\n"
       code +="   }\n"
       if( self.args.architecture == "cuda"):
           code +="   cudaMemcpy(A_d, A, sizeof(%s) * %s, cudaMemcpyHostToDevice);\n"%(self.floatType,sizeA)
           code +="   cudaMemcpy(B_d, B, sizeof(%s) * %s, cudaMemcpyHostToDevice);\n"%(self.floatType,sizeB)
           code +="   cudaMemcpy(C_d, C, sizeof(%s) * %s, cudaMemcpyHostToDevice);\n"%(self.floatType,sizeC)

       indent = "   "
       level = 1
       code += "   int mRepeat = 2;\n"
       code += "   int nRepeat = 4;\n"

       code += "   double ttgemmt_time[%d];\n"%self.ttgemmt.numCandidates()
       code += "   double gemm_time[%d];\n"%self.gemm.numCandidates()
       code += "   double loopGemm_time[%d];\n"%self.gemmLoop.numCandidates()
       code += "   double gett_time[%d];\n"%self.gett.numImpl

       code += "   for(int l = 0; l < %d; l++)\n"%self.gett.numImpl
       code += "      gett_time[l] = FLT_MAX;\n"
       code += "   for(int l = 0; l < %d; l++)\n"%self.ttgemmt.numCandidates()
       code += "      ttgemmt_time[l] = FLT_MAX;\n"
       code += "   for(int l = 0; l < %d; l++)\n"%self.gemmLoop.numCandidates()
       code += "      loopGemm_time[l] = FLT_MAX;\n"
       code += "   for(int l = 0; l < %d; l++)\n"%self.gemm.numCandidates()
       code += "      gemm_time[l] = FLT_MAX;\n"

       code += "   const double flops = %e;\n"%self.getFlopCount()
       code += "   for(int r = 0; r < mRepeat; r++){\n"
       ############################
       # REFERENCE version
       ############################
       if( self.args.testing == 1):
           code += "   double ref_time = FLT_MAX;\n"
           code += "   for(int i = 0; i < nRepeat; i++){\n"
           code += "      restore(C_copy, C_ref, %s);\n"%sizeC
           code += "      trashCache(trash1, trash2, largerThanL3);\n"
           code += "      double start = omp_get_wtime();\n"
           code += "      referenceVersion(A, B, C_ref, alpha, beta);\n"
           code += "      double tmp = omp_get_wtime() - start;\n"
           code += "      ref_time = (tmp < ref_time) ? tmp : ref_time;\n"
           code += "  }\n"
           code += "  if( r == (mRepeat-1) )\n"
           code += "   printf(\"reference: %.2f GFLOPS\\n\", flops / 1e9 / ref_time );\n"
           code += "\n"

       ############################
       # GETT version
       ############################
       code += "   //launch gett kernels\n"
       count = 0
       for key in self.gett.implementations:
           for (variant, estimatedGflops) in self.gett.implementations[key]:
               code += "   {\n"
               code += "      int success = 1;\n"
               code += "      for(int i = 0; i < nRepeat; i++){\n"
               code += "         restore(C_copy, C, %s);\n"%sizeC
               code += "         trashCache(trash1, trash2, largerThanL3);\n"
               code += "         double start = omp_get_wtime();\n"
               code += "         %s(A, B, C, alpha, beta);\n"%variant
               code += "         double tmp = omp_get_wtime() - start;\n"
               code += "         gett_time[%d] = (tmp < gett_time[%d]) ? tmp : gett_time[%d];\n"%(count,count,count)
               code += "   \n"
               if( self.args.testing == 1 and self.gett.fastMeasurements != 1):
                   code += "         if( i==0 )\n"
                   code += "            if( equal(C_ref, C, %s) != 1 ){\n"%sizeC
                   code += "               printf(\"ERROR: version '%s' failed\\n\");\n"%variant
                   code += "               success = 0;\n"
                   code += "               break;\n"
                   code += "            }\n"
               code += "      }\n"
               code += "   if( success && r == (mRepeat-1) )\n"
               if( self.gett.fastMeasurements ):
                   code += "   printf(\"%s: %%.2f GFLOPS // estimated: %.2f GFLOPS\\n\", %e / 1e9 / gett_time[%d]);\n"%(variant, estimatedGflops, self.gett.getFastFlopsFromGettName(variant), count)
               else:
                   code += "   printf(\"%s: %%.2f GFLOPS // estimated: %.2f GFLOPS\\n\", flops / 1e9 / gett_time[%d]);\n"%(variant, estimatedGflops, count)
               code += "   }\n"
               count += 1
       code += "\n"
       
       ############################
       # Loop-over-GEMM version
       ############################
       code += "   //launch loop-gemm kernels\n"
       count = 0
       for candidate in self.gemmLoop.candidates:
           code += "  {\n"
           code += "   int success = 1;\n"
           code += "   for(int i = 0; i < nRepeat; i++){\n"
           code += "      restore(C_copy, C, %s);\n"%sizeC
           code += "      trashCache(trash1, trash2, largerThanL3);\n"
           if( self.args.architecture == "cuda"):
               code +="      cudaMemcpy(C_d, C, sizeof(%s) * %s, cudaMemcpyHostToDevice);\n"%(self.floatType,sizeC)
           code += "      double start = omp_get_wtime();\n"
           if( self.args.architecture == "cuda"):
               code += "      int ret = %s(cublas_handle, A_d, B_d, C_d, alpha, beta);\n"%candidate
               code += "      cudaDeviceSynchronize();\n"
               code += "      if( ret != 0 || cudaSuccess != cudaGetLastError()) { \n"
               code += "            printf(\"ERROR: version 'loopGemm_%s' failed\\n\");\n"%candidate
               code += "            success = 0;\n"
               code += "            break;\n"
               code += "      }\n"
           else:
               code += "      %s(A, B, C, alpha, beta);\n"%candidate
           code += "      double tmp = omp_get_wtime() - start;\n"
           code += "      loopGemm_time[%d] = (tmp < loopGemm_time[%d]) ? tmp : loopGemm_time[%d];\n"%(count,count,count)
           if( self.args.testing == 1):
               code += "\n"
               code += "      if( i==0 ){\n"
               if( self.args.architecture == "cuda"):
                   code +="         cudaMemcpy(C, C_d, sizeof(%s) * %s, cudaMemcpyDeviceToHost);\n"%(self.floatType,sizeC)
               code += "         if( equal(C_ref, C, %s) != 1 ){\n"%sizeC
               code += "            printf(\"ERROR: version 'loopGemm_%s' failed\\n\");\n"%candidate
               code += "            success = 0;\n"
               code += "         }\n"
               code += "      }\n"
           code += "     }\n"
           code += "   if( success  && r == (mRepeat-1))\n"
           code += "     printf(\"loopGemm_%s: %%.2f GFLOPS\\n\", flops / 1e9 / loopGemm_time[%d]);\n"%(candidate,count)
           code += "  }\n"
           count+= 1

       ############################
       # GEMM version
       ############################
       code += "   //launch gemm kernels\n"
       count = 0
       for candidate in self.ttgemmt.candidates:
           code += "  {\n"
           code += "   int success = 1;\n"
           code += "   for(int i = 0; i < nRepeat; i++){\n"
           code += "      restore(C_copy, C, %s);\n"%sizeC
           code += "      trashCache(trash1, trash2, largerThanL3);\n"
           if( self.args.architecture == "cuda"):
               code +="      cudaMemcpy(C_d, C, sizeof(%s) * %s, cudaMemcpyHostToDevice);\n"%(self.floatType,sizeC)
           code += "      double start = omp_get_wtime();\n"
           if( self.args.architecture == "cuda"):
               code += "      int ret = gemm_%s(cublas_handle, A_d, B_d, C_d, alpha, beta, work_d);\n"%candidate
               code += "      cudaDeviceSynchronize();\n"
               code += "      if( ret != 0 || cudaSuccess != cudaGetLastError() ) { \n"
               code += "            printf(\"ERROR: version 'gemm_%s' failed\\n\");\n"%candidate
               code += "            success = 0;\n"
               code += "            break;\n"
               code += "      }\n"
           else:
               code += "      gemm_%s(A, B, C, alpha, beta, work_);\n"%candidate
           code += "      double tmp = omp_get_wtime() - start;\n"
           code += "      gemm_time[%d] = (tmp < gemm_time[%d]) ? tmp : gemm_time[%d];\n"%(count, count, count)
           code += "   }\n"
           code += "   if( success  && r == (mRepeat-1))\n"
           code += "     printf(\"gemm_%s: %%.2f GFLOPS\\n\", flops / 1e9 / gemm_time[%d]);\n"%(candidate,count)
           code += "  }\n"
           count += 1


       ############################
       # TTGEMMT version
       ############################
       code += "   //launch ttgemmt kernels\n"
       count = 0
       for candidate in self.ttgemmt.candidates:
           code += "  {\n"
           code += "   int success = 1;\n"
           code += "   for(int i = 0; i < nRepeat; i++){\n"
           code += "      restore(C_copy, C, %s);\n"%sizeC
           code += "      trashCache(trash1, trash2, largerThanL3);\n"
           if( self.args.architecture == "cuda"):
               code +="      cudaMemcpy(C_d, C, sizeof(%s) * %s, cudaMemcpyHostToDevice);\n"%(self.floatType,sizeC)
           code += "      double start = omp_get_wtime();\n"
           if( self.args.architecture == "cuda"):
               code += "      int ret = ttgemmt_%s(cublas_handle, A_d, B_d, C_d, alpha, beta, work_d);\n"%candidate
               code += "      cudaDeviceSynchronize();\n"
               code += "      if( ret != 0 || cudaSuccess != cudaGetLastError() ) { \n"
               code += "            printf(\"ERROR: version 'ttgemmt_%s' failed\\n\");\n"%candidate
               code += "            success = 0;\n"
               code += "            break;\n"
               code += "      }\n"
           else:
               code += "      ttgemmt_%s(A, B, C, alpha, beta, work_);\n"%candidate
           code += "      double tmp = omp_get_wtime() - start;\n"
           code += "      ttgemmt_time[%d] = (tmp < ttgemmt_time[%d]) ? tmp : ttgemmt_time[%d];\n"%(count,count,count)
           if( self.args.testing == 1):
               code += "\n"
               code += "      if( i==0 ){\n"
               if( self.args.architecture == "cuda"):
                   code +="         cudaMemcpy(C, C_d, sizeof(%s) * %s, cudaMemcpyDeviceToHost);\n"%(self.floatType,sizeC)
               code += "         if( equal(C_ref, C, %s) != 1 ){\n"%sizeC
               code += "            printf(\"ERROR: version 'ttgemmt_%s' failed\\n\");\n"%candidate
               code += "            success = 0;\n"
               code += "         }\n"
               code += "      }\n"
           code += "   }\n"
           code += "   if( success  && r == (mRepeat-1))\n"
           code += "      printf(\"ttgemmt_%s: %%.2f GFLOPS\\n\", flops / 1e9 / ttgemmt_time[%d]);\n"%(candidate,count)
           code += "  }\n"
           count += 1

       code += "  sleep(0.5);\n"
       code += " }\n"
       code += "\n"
       if( self.args.architecture == "cuda"):
           code += "   cudaFree(A_d);\n"
           code += "   cudaFree(B_d);\n"
           code += "   cudaFree(C_d);\n"
           code += "   cudaFree(work_d);\n"
       code += "   free(A);\n"
       code += "   free(B);\n"
       code += "   free(C);\n"
       code += "   free(C_ref);\n"
       code += "   free(C_copy);\n"
       code += "   free(trash1);\n"
       code += "   free(trash2);\n"
       code += "   free(work_);\n"
       code += "   printf(\"TCC_END\\n\");\n"
       code += "   return 0;\n"
       code += "}\n"

       fgett = open("main.cpp","w")
       fgett.write(code)
       fgett.close()

    def getFlopCount(self):
        if(self.beta != 0):
           return (self.fmul + self.fadd) * self.sizeM * self.sizeN * self.sizeK + (2. * self.fmul + self.fadd) * self.sizeM * self.sizeN
        else:
           return (self.fmul + self.fadd) * self.sizeM * self.sizeN * self.sizeK

def createTmpDirectory():
    directory = "tmp"
    i = 0
    while( os.path.exists(directory+"%d"%i) ):
        i += 1
    tmpDirectory = directory+"%d"%i
    os.makedirs(tmpDirectory)
    return tmpDirectory


def main():

    workingDir = os.getcwd()

    ###########################################
    # Parse Arguments
    ###########################################
    parser = argparse.ArgumentParser(description='Generate high-performance C++ code for a given tensor contraction.')
    parser.add_argument('filename', metavar='filename', type=str, help='filename of the file which contains the symbolic description of the tensor contraction')
    parser.add_argument('--verbose', action="store_true", help='Verbose output (useful for debugging).')
    parser.add_argument('--testing', action="store_true", help='enable testing of the generated routines.')
    parser.add_argument('--keep', action="store_true", help='keep generated C++ files under ${TCCG_ROOT}/tccg/tmp')
    parser.add_argument('--noBatchedGEMM', action="store_true", help='Disable batched GEMMs for LoG variant.')
    parser.add_argument('--generateOnly', action="store_true", help='only generate code')
    parser.add_argument('--gettOnly', action="store_true", help='Only use GETT implementation')
    parser.add_argument('--noGETT', action="store_true", help='Do not use GETT implementation')
    parser.add_argument('--noLoG', action="store_true", help='Do not use LoG implementation')
    parser.add_argument('--noGEMM', action="store_true", help='Do not time equally sized GEMM')
    parser.add_argument('--noTTGT', action="store_true", help='Do not use TTGT implementation')
    parser.add_argument('--useDynamicMemory', action="store_true", help='use dynamic memory for the small tiles. This might be required of the stack size is insufficient. However, this might degrade performance! (deactivated by default)')
    parser.add_argument('--timePacking', action="store_true", help='times packing routines.')
    parser.add_argument('--ignoreDatabase', action="store_true", help='ignore SQL database. This prevents a database lookup for an existing solution.')
    parser.add_argument('--fastMeasurements', action="store_true", help='Significantly reduces the measuring time for GETT. This feature deactivates testing. Moreover, the attained FLOPs are just a (very good) estimate for the real flops attained by GETT.')
    parser.add_argument('--numThreads', type=int, help='number of threads.')
    parser.add_argument('--maxImplementations', type=int, help='maximum number of GETT candidates to evaluate (default: 16). -1 denotes infinity.')
    parser.add_argument('--maxImplementationsTTGT', type=int, help='maximum number of TTGT candidates to evaluate (default: 1). -1 denotes infinity.')
    parser.add_argument('--maxImplementationsLoG', type=int, help='maximum number of LoG candidates to evaluate (default: 1). -1 denotes infinity.')
    parser.add_argument('--maxWorkspace', type=float, help='maximum auxiliary workspace in GB (default: no limit); this only affects the TTGT approach.')
    parser.add_argument('--arch', metavar='arch', type=str, help='architecture can be either avx2 (default), avx512, cuda.')
    parser.add_argument('--compiler', metavar='compiler', type=str, help='compiler can be either icpc (default), g++ or nvcc.')
    parser.add_argument('--floatType', metavar='floatType', type=str, help='floatType can bei either \'s\' or \'d\'.')
    parser.add_argument('--affinity', metavar='affinity', type=str, help="""thread affinity (WARNING: this
    value should to be specified by the user explicitly because it can effect
    performance severely and the optimal choice depends on the
    enumeration/numbering of the cores on your system)
    The thread affinity respectively sets the value for the 'KMP_AFFINITY' or the 'GOMP_CPU_AFFINITY' environment variable for icpc or g++ compiler.
       For instance, using --compiler=icpc _and_ --affinity=compact,1 will set 'KMP_AFFINITY=compact,1'.
       Similarly, using --compiler=g++ _and_ --affinity=0-4 will set 'GOMP_CPU_AFFINITY=0-4'.""")

    args = parser.parse_args()

    if( os.path.isabs(args.filename) ): #use relative path first
        tccgArgs = TccgArgs( args.filename )
    else: # treat filename as absolute path
        tccgArgs = TccgArgs( workingDir + "/" + args.filename )

    if( args.arch ):
        tccgArgs.architecture = args.arch
    if( args.compiler ):
        tccgArgs.compiler = args.compiler 
        if(args.compiler != "icpc" and args.compiler != "nvcc" and args.compiler
                != "g++"):
            print "ERROR: unknown compiler. Choose either 'icpc', 'g++' or 'nvcc'"
            exit(-1)
    if( tccgArgs.architecture  == "cuda"):
        tccgArgs.compiler = "nvcc"
    if( args.numThreads ):
        tccgArgs.numThreads = args.numThreads
    if( args.maxWorkspace != None):
        if( args.maxWorkspace == -1 ):
            tccgArgs.maxWorkspace = 1000000000
        else:
            tccgArgs.maxWorkspace = args.maxWorkspace
    if( args.maxImplementations != None):
        if( args.maxImplementations == -1 ):
            tccgArgs.maxImplementations = 1000000000
        else:
            tccgArgs.maxImplementations = args.maxImplementations
    if( args.floatType != None):
        if( args.floatType == "s" ): 
            tccgArgs.floatTypeA = "float"
            tccgArgs.floatTypeB = "float"
            tccgArgs.floatTypeC = "float"
        elif( args.floatType == "d" ): 
            tccgArgs.floatTypeA = "double"
            tccgArgs.floatTypeB = "double"
            tccgArgs.floatTypeC = "double"
        else:
            print "ERROR: floatType %s is not supported yet."%args.floatType
            exit(-1)
    if( args.affinity):
        tccgArgs.affinity = args.affinity

    print tccgArgs

    if( not args.affinity):
        if(tccgArgs.compiler == "g++" ):
            tccgArgs.affinity = "0-%d"%multiprocessing.cpu_count()
            print WARNING + "WARNING: you did not specify an thread affinity. We are using: GOMP_CPU_AFFINITY=%s by default"%tccgArgs.affinity +ENDC
            print WARNING + "WARNING: The default thread affinity might be suboptimal depending on the numbering of your CPU cores. We recommend using a ''compact'' thread affinity even for g++ (i.e., simulate KMP_AFFINITY=compact)."+ENDC
        else:
            tccgArgs.affinity = "compact,1"
            print WARNING + "WARNING: you did not specify an thread affinity. We are using: KMP_AFFINITY=%s by default"%tccgArgs.affinity +ENDC

    tccgArgs.workingDir = workingDir
    tccgArgs.verbose = args.verbose
    tccgArgs.gettOnly = args.gettOnly
    tccgArgs.noGETT = args.noGETT
    tccgArgs.noTTGT = args.noTTGT
    tccgArgs.noGEMM = args.noGEMM
    tccgArgs.noLoG = args.noLoG
    tccgArgs.testing = args.testing
    tccgArgs.useDynamicMemory = args.useDynamicMemory 
    tccgArgs.ignoreDatabase = args.ignoreDatabase
    tccgArgs.useTimings = args.timePacking
    tccgArgs.fastMeasurements = args.fastMeasurements
    tccgArgs.generateOnly = args.generateOnly 
    tccgArgs.batchedGEMM = not args.noBatchedGEMM

    _tccg_root = ""
    if( os.environ.has_key('TCCG_ROOT') ):
        _tccg_root = os.environ['TCCG_ROOT']
    else:
        print FAIL + "ERROR: TCCG_ROOT environment variable not set. Make sure that this variable points to the directory containing 'tccg.py'" + ENDC
        exit(-1)

    os.chdir(_tccg_root+"/tccg")

    tmpDirectory = createTmpDirectory()
    tccgArgs.tmpDirectory  = tmpDirectory 
    # change dir into a unique directory
    os.chdir(tmpDirectory)
 
    ###########################################
    # read config
    ###########################################
    if( not os.path.isfile("../../config.cfg") ):
        print "ERROR: config.cfg not found. This file should be available within the TCCG_ROOT directory."
        exit(-1)
    f = open("../../config.cfg","r")
    blasLib = ""
    cudaLib = ""
    cudaInclude = ""
    cudaArch = ""
    for l in f:
        pos = l.find("#") # remove comments
        line = l
        if( pos != -1 ):
            line = l[:pos]
        if( line.find("BLAS_LIB") != -1):
            blasLib = line.split("=")[1]
            if( blasLib.lower().find("mkl") == -1 ):
                tccgArgs.batchedGEMM = 0 #disable batched gemm
        if( line.find("CUDA_LIB") != -1):
            cudaLib = line.split("=")[1]
        if( line.find("CUDA_INCLUDE") != -1):
            cudaInclude = line.split("=")[1]
        if( line.find("CUDA_ARCH") != -1):
            cudaArch= line
    if( blasLib != "" ):
        tccgArgs.blasLib=blasLib
    if( cudaLib != "" ):
        tccgArgs.cudaLib=cudaLib
    if( cudaInclude != "" ):
        tccgArgs.cudaInclude = cudaInclude
    if( cudaArch != "" ):
        tccgArgs.cudaArch = cudaArch
  
    ###########################################
    # generate versions
    ###########################################
    print OKGREEN + "[generate] Generate implementations ...                ", ENDC
    try:
        tccg = Tccg( tccgArgs )
        tccg.codeGen()
    except:
        print "An error has occurred."
        print traceback.print_exc(file=sys.stdout)
        print "You could try to run with --useDynamicMemory option and see if the problem still exists."

    # remove/delete temporary directory
    os.chdir("..")
    if( (not args.keep) and os.path.exists(tmpDirectory) ):
        shutil.rmtree(tmpDirectory)

    os.chdir(workingDir)

