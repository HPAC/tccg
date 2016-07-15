# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
from Index import *
import itertools
import sys
import traceback
import ttc
import copy
import subprocess

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'
  
def getBatchedGemmFunctionName( floatType, arch ):
    if( floatType == "float" ):
        if( arch == "cuda" ):
            return "cublasSgemmBatched"
        else:
            return "sgemm_batch"
    elif( floatType == "double" ):
        if( arch == "cuda" ):
            return "cublasDgemmBatched"
        else:
            return "dgemm_batch"
    elif( floatType == "float complex" ):
        if( arch == "cuda" ):
            return "cublasCgemmBatched"
        else:
            return "cgemm_batch"
    elif( floatType == "double complex" ):
        if( arch == "cuda" ):
            return "cublasZgemmBatched"
        else:
            return "zgemm_batch"
    else:
        print "ERROR: floattype unknown."
        exit(-1)


def getGemmFunctionName( floatType, arch ):
    if( floatType == "float" ):
        if( arch == "cuda" ):
            return "cublasSgemm"
        else:
            return "sgemm_"
    elif( floatType == "double" ):
        if( arch == "cuda" ):
            return "cublasDgemm"
        else:
            return "dgemm_"
    elif( floatType == "float complex" ):
        if( arch == "cuda" ):
            return "cublasCgemm"
        else:
            return "cgemm_"
    elif( floatType == "double complex" ):
        if( arch == "cuda" ):
            return "cublasZgemm"
        else:
            return "zgemm_"
    else:
        print "ERROR: floattype unknown."
        exit(-1)

def getCPUarchitecture():
    try:
        f = open("/proc/cpuinfo", "r")
        for l in f:
            if( l.find("model name") != -1):
                arch = l.split(":")[1]
                arch = arch.replace(":","")
                pos = arch.find("@")
                f.close()
                return arch[0:pos]
    except:
        return "dummy"

def getSizeStr(A, B, C):
    size = {}
    for idx in A.indices:
        size[idx.label] = idx.size
    for idx in B.indices:
        size[idx.label] = idx.size
    for idx in C.indices:
        size[idx.label] = idx.size

    string = ""
    for idx in size:
        string += "%s:%d, "%(idx, size[idx])
    return string[0:-2]

def getCompilerVersion(compiler):
    comp = ""
    if( compiler == "icpc" ):
        comp = "icpc"
    if( compiler == "g++" or compiler == "gcc" ):
        comp = "g++"
    if( compiler == "ibm" ):
        comp = "bgxlc"
    if( compiler == "nvcc"):
	comp = "nvcc"

    version = "--version"
    if( compiler == "ibm" ):
        version = "-qversion"
    try:
        proc = subprocess.Popen([comp, version],stdout=subprocess.PIPE)
        proc.wait()
    except:
        print "ERROR: some error has occured. Does the selected compiler (%s) exist?"%comp
        exit(-1)

    output = proc.communicate()[0].split("\n")
    output = output[0].replace(":","")
    return output

def tccError(string):
    print FAIL + "[TCC] ERROR: %s."%string + ENDC

def hasItem( L, item ):
    for l in L:
        if( l == item):
            return 1
    return 0

def getFloatSize(floatType):
    if( floatType == "float" ):
        return 4
    elif( floatType == "double" or floatType == "complex" ):
        return 8
    elif( floatType == "double complex" ):
        return 16
    else:
        print "ERROR: float type unknown."
        exit(-1)

def allocateMemory(codeblocks, floatType):
    alignment = 32
    indent = "   "
    done = []
    code = ""
    workingSet = 0
    for block in codeblocks:
        if( block.OUT.label.find("_") != -1 and not hasItem(done, block.OUT) ):
            done.append(block.OUT)
            # ensure alignment
            tmpSize = block.OUT.getSize() * getFloatSize(floatType) #in bytes
            if( tmpSize % alignment != 0):
                tmpSize += (alignment - tmpSize % alignment)
            code += "%s%s *%s = (%s*) (((char*)work_)+%d);\n"%(indent, floatType, block.OUT.label, floatType, workingSet )
            workingSet += tmpSize

    return (code, workingSet)

def getFloatTypeSize(floatType ):
    if( floatType == "float" ):
        return 4
    if( floatType == "double" ):
        return 8
    if( floatType == "float complex" ):
        return 8
    if( floatType == "double complex" ):
        return 16


#TTC interprets the leadingdimension differently than TCC does.
def convertTTCldaToTCClda(lda, sizeOfLastIndex):
   ttcLDA = []
   for i in range(1,len(lda)):
      ttcLDA.append(lda[i]/lda[i-1])
   ttcLDA.append(sizeOfLastIndex)
   return ttcLDA

def getPerm(IN, OUT):
   perm = []
   for idx in OUT.indices:
       pos = IN.getPos(idx)
       perm.append(pos)
   return perm

def generateTranspose(IN, OUT, floatType, alpha, beta, numThreads,
        hotIN, hotOUT, nameOnly, arch, generateOnly):
   perm = getPerm(IN, OUT)

   size = []
   for idx in IN.indices:
      size.append(idx.size)

   lda = copy.deepcopy(convertTTCldaToTCClda(IN.ld, IN.indices[-1].size))
   ldb = copy.deepcopy(convertTTCldaToTCClda(OUT.ld, OUT.indices[-1].size))

#   print perm, size, lda, ldb

   try:
       ttcVersion = ttc.ttc_util.getVersion()
       if( ttcVersion[0] < 0 or ttcVersion[1] < 1 or ttcVersion[2] < 0 ):
           print "ERROR: your TTC version is not up to date. Please update TTC to version v0.1.0"
           exit(-1)
   except:
       print "ERROR: your TTC version is not up to date. Please update TTC to version v0.1.0"
       exit(-1)
   ttc_args = ttc.ttc_util.TTCargs(perm, size)
   ttc_args.alpha = alpha
   ttc_args.beta = beta
   ttc_args.affinity = "compact,1"
   ttc_args.numThreads = numThreads
   ttc_args.floatTypeA = floatType
   ttc_args.floatTypeB = floatType
   ttc_args.streamingStores = 0
   ttc_args.maxNumImplementations = 10
   if( generateOnly ):
      ttc_args.maxNumImplementations = 1
   ttc_args.ignoreDatabase = 0
   ttc_args.lda = lda
   ttc_args.ldb = ldb
   ttc_args.debug = 0
   if( arch == "avx" or arch == "avx2" ):
       ttc_args.architecture = "avx"
   else:
       ttc_args.architecture = arch
   ttc_args.align = 1
   ttc_args.blockings = []
   ttc_args.loopPermutations = []
   ttc_args.prefetchDistances  = []
   ttc_args.scalar = 0
   ttc_args.silent = 1
   ttc_args.hotA = hotIN
   ttc_args.hotB = hotOUT

   #ttc_args.getCommandLineString() #for debugging purposes

   try:
      if( nameOnly ):
          transposeName = ttc.ttc.getTransposeName(ttc_args)
          bandwidth = -1
      else:
          (transposeName, bandwidth) = ttc.ttc.generateTransposition( ttc_args )
          #print "TTC attains ",bandwidth, "GiB/s for permutation: ", perm
   except:
      print "TTC exited with an error for the given permutation:"
      print OUT, "<-", IN 
      print traceback.print_exc(file=sys.stdout)
      print ttc_args.getCommandLineString()
      raise

   if( generateOnly ):
       print ttc_args.getCommandLineString()
   return (transposeName, bandwidth, ttc_args.getPerm(), ttc_args.getSize(), ttc_args.lda, ttc_args.ldb)

def splitIndices(indices, pos, size):
    newIdx = index(indices[pos].label + "_1", indices[pos].size/size)
    indices[pos].label += "_0"
    indices[pos].size = size
    indices.insert(pos+1, newIdx) #add new index

#get all indices that are in A, B _and_ C
def getLoopIndices(A, B, C):
    indices = []

    for idxB in B.indices:
        okayC = 0
        for idx in C.indices:
            if(idxB == idx):
                okayC = 1 

        okayA = 0
        for idx in A.indices:
            if(idxB == idx):
                okayA = 1
        if( okayC == 1 and okayA == 1):
            indices.append(idxB)
    return indices

#get all indices that are in B _and_ C but not in A
def getNindices(A, B, C):
    indices = []

    if( A.hasIndex(C.indices[0]) ):
        AA = A
        BB = B
    else:
        AA = B
        BB = A

    for idxB in C.indices:
        okay = 0
        for idx in BB.indices:
            if(idxB == idx):
                okay = 1 
        for idx in AA.indices:
            if(idxB == idx):
                okay = 0
        if( okay == 1):
            indices.append(idxB)
    return indices


#get all indices that are in A _and_ C but not in B
def getMindices(A, B, C):
    indices = []

    if( A.hasIndex(C.indices[0]) ):
        AA = A
        BB = B
    else:
        AA = B
        BB = A

    for idxA in C.indices:
        okay = 0
        for idx in AA.indices:
            if(idxA == idx):
                okay = 1 
        for idx in BB.indices:
            if(idxA == idx):
                okay = 0
        if( okay == 1):
            indices.append(idxA)
    return indices

def listToString(l):
    out = ""
    if(len(l) > 0):
        out = "("
        for idx in l:
            out += str(idx) + ","
        out = out[:-1] + ")"
    return out

def splitIndex(idx, size):
    if( idx.size % size != 0):
        print "Index %s: nothing to split."%(str(idx))
        exit(-1)
    
    idx0 = index(idx.label + "_0", size)             # paper: S_{\widetilde{m}_0}
    idx1 = index(idx.label + "_1", idx.size / idx0.size ) # paper: S_{\widetilde{m}_1}
    return (idx0, idx1)

def getPrimeFactors(a):
    """
    IMPORTANT: Returns all prime factors of 'a' which are smaller than 5000! (special purpose)
    """
    primFactors = []

    primeNumbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903, 4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373, 6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111, 8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537, 8539, 8543, 8563, 8573, 8581, 8597, 8599, 8609, 8623, 8627, 8629, 8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807, 8819, 8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893, 8923, 8929, 8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203, 9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733, 9739, 9743, 9749, 9767, 9769, 9781, 9787, 9791, 9803, 9811, 9817, 9829, 9833, 9839, 9851, 9857, 9859, 9871, 9883, 9887, 9901, 9907, 9923, 9929, 9931, 9941, 9949, 9967, 9973]
    #if(a > primeNumbers[-1]):
    #  print "WARNING: number too large. The tcc::getPrimeFactors() implementation might yield the wrong result"

    for p in primeNumbers:
        if(p > a or p > 5000):
            break
        while (a != 1 and a%p == 0):
            primFactors.append(p)
            a = a/p
    return primFactors

def removeDuplicates(l):
    noDup = [l[0]]
    for i in l:
        if not i in noDup:
            noDup.append(i)
    return noDup

def findPosition(item, l):
    for i in range(len(l)):
        if(l[i] == item):
            return i
    return -1

def mergeIndicesWithSameLabel(l):
    newList = []
    for idx in l:
        pos = findPosition(idx, newList)
        if( pos >= 0 ):
            newList[pos].size *= idx.size
        else:
            newList.append(idx)
    return newList


def __getSolutions(primes, counts, indicesWithPrimeFactor, solutions):
    if( len (primes) == 0):
        return solutions

    p = primes[0]
    myPrimes = primes[1:]

    # pick count[p] many indices from indicesWithPrimfactor[p]
    subsets = itertools.combinations(indicesWithPrimeFactor[p], counts[p])

    #print "%d: %d"%(p, counts[p])
    #print "current sol: "
    #for s in solutions:
    #    print list2str(s)
    #print "----"

    uniqueSolutions = []
    for s in subsets:
        countOcc = {}
        for idx in s:
            if( countOcc.has_key(idx.label) ):
                countOcc[idx.label] += 1
            else:
                countOcc[idx.label] = 1

        sol = []
        for idx in s:
            idx0 = index(idx.label, p ** countOcc[idx.label])
            if( not idx0 in sol):
                sol.append(idx0)

        if not sol in uniqueSolutions:
            uniqueSolutions.append(sol)
    #print "unique sol: "
    #for s in uniqueSolutions:
    #    print list2str(s)
    #print "----"

    # recombine new solutions
    newSolutions = []
    for s1 in solutions:
        for s2 in uniqueSolutions:
            newSolutions.append( copy.deepcopy(s1 + s2) )

    newSolutions2 = []
    for s in newSolutions:
        s = mergeIndicesWithSameLabel(s)
        if( not s in newSolutions2 ):
            newSolutions2.append(s)

    #print "final sol: "
    #for s in newSolutions2:
    #    print list2str(s)
    #print "----"
    
    return __getSolutions(myPrimes, counts, indicesWithPrimeFactor, newSolutions2)


def countRequiredSplits(subset, superset):
    numSplits = 0
    for idx in subset:
        pos = findPosition(idx, superset)
        if( pos >= 0 and idx.size != superset[pos].size ):
            numSplits += 1
    return numSplits

    
    

# This function will split the given index set I into two separate index sets I^{1} and I^{2} := I \ I^{1}
# such that Size(I_1) = size and Size(I_2) = Size(I) / size.
# Note that I_1 may not just be a subset of I but it might be required to
# split some indices of I to achieve the constraint Size(I_1) = size. Hence,
# I = I_1 \union I_2 just rambles the rough idea of what's actually happening.
def splitIndexSet(indices, size, tensorA, tensorB):

    primsMC = getPrimeFactors(size)

    # count the occurrences of each prime factor of 'size'
    count = {} 
    indicesWithPrimfactor = {}
    for p in primsMC:
        if( count.has_key(p) ):
            count[p] += 1
        else:
            count[p] = 1
        indicesWithPrimfactor[p] = []

    for idx in indices:
        prims = getPrimeFactors(idx.size)

        for p in prims:
            if( indicesWithPrimfactor.has_key(p) ): #only add primefactor that contribute to the splitting of size 
                indicesWithPrimfactor[p].append(idx)

    # 1) get all candidates
    indL0 = set()
    candidates = __getSolutions(list(set(primsMC)), count, indicesWithPrimfactor, [[]])

    # 2) remove all invalid candidates
    validCandidates = []
    # remove all candidates that are not valid (i.e., those which would require a non-unite stride for the packing)
    for candidate in candidates: 
        if(( not hasItem(indices, tensorA.indices[0]) or hasItem(candidate, tensorA.indices[0])) and 
          ( not hasItem(indices, tensorB.indices[0]) or hasItem(candidate, tensorB.indices[0]) ) ):
            validCandidates.append(candidate)

    # 3) select good candidates
    minSplits = 10000000
    goodCandidates = []
    for s in validCandidates:
        minSplits = min( minSplits, countRequiredSplits(s, indices) )

    # only select candidates with the minimum number of splits => minimize the
    # dimension of the resulting tensor
    for s in validCandidates:
        if countRequiredSplits(s, indices) == minSplits:
            goodCandidates.append(s)

    if( len( goodCandidates ) <= 0):
        return []
        #raise NameError("It is not possible to find a subset of size mcL1 \
        #        (or ncL1) for the given set I_m (or I_n). Please choose a \
        #        different mcL1, use padding of the tensors or choose a \
        #        different set.")

    ####### At this point we have good candidates available #########
    ################ We need to make a choice now ###################

    solutions = []
    for candidate in goodCandidates:
        ind0 = []
        ind1 = []
        tensorAA = copy.deepcopy(tensorA)
        tensorBB = copy.deepcopy(tensorB)

        for idx in candidate:
            pos = findPosition(idx, indices)
            if( pos == -1 ):
                print "[TCC] Error: Index ",idx,"not found"
                traceback.print_stack()   
            toSplit = indices[pos]
            if( idx.size != toSplit.size ): #only split if necessary

                (idx0, idx1) = splitIndex(toSplit, idx.size)
                replaceWith = [idx0, idx1]
                
                if( tensorAA.hasIndex(toSplit) and tensorBB.hasIndex(toSplit) ):
                     #we also need to split the corresponding indices of the tensors
                     tensorAA.replaceIndex(toSplit, replaceWith)
                     tensorBB.replaceIndex(toSplit, replaceWith)
                else:
                    print "[TCC] Error in splitIndexSet(): index not found in tensor."
                    print tensorA, tensorB, listToString(indices), size, idx
                    traceback.print_stack()   
                    exit(-1)
                ind0.append(idx0)
                ind1.append(idx1)
            else:
                ind0.append(idx)

        # add all non-remaining indices to ind1
        for idx in indices:
            if( not hasItem(candidate, idx) ):
                ind1.append(idx)

        solutions.append((ind0, ind1, tensorAA, tensorBB))

    return solutions

#DEPRECATED
def splitIndexSet_OLD(indices, mc, tensorA, tensorB):
        ind0 = [] #find a set ind0 \subset indices s.t. size(ind0) == mc (ind0 does not have to be a subset since we might have to split indices)
        
        ind1 = []
        size = 1
        i = 0
        while( i < len(indices) and size * indices[i].size <= mc ):
            size *= indices[i].size
            ind0.append(copy.deepcopy(indices[i]))
            i += 1 

        # we need to split an index into two (paper: $\tilde{m}$, $\tilde{n}$ or # $\tilde{k}$)
        if( i < len(indices) and size < mc and size * indices[i].size > mc ):
            idx = indices[i]
            if( mc % size != 0):
                print "Index %s cannot be split by %d."%(str(idx), size)
                exit(-1)
            newSizeIndex = mc / size #resize and split the index such that size(ind0) == mc!
            (idx0, idx1) = splitIndex(idx, newSizeIndex)
            replaceWith = [idx0, idx1]

            if( tensorA.hasIndex(idx) and tensorB.hasIndex(idx) ):
                 #we also need to split the corresponding indices of the tensors
                 tensorA.replaceIndex(idx, replaceWith)
                 tensorB.replaceIndex(idx, replaceWith)
            else:
                print "Error: splitIndexSet(): index not found in tensor."
                exit(-1)
            ind0.append(idx0)
            ind1.append(idx1)
            i += 1

        # the remaining indices will only be affected by the outer-loops around the
        # macro kernel
        while( i < len(indices)):
            ind1.append(copy.deepcopy(indices[i]))
            i += 1 

        return (ind0, ind1)



def getContractedIndices(A, B, C):
    contracted = []

    for idxA in A.indices:
        okay = 0

        for idxB in B.indices:
            if(idxA == idxB):
                okay = 1 #an index is a contracted index if it appears in A _and_ B
        for idxC in C.indices:
            if(idxA == idxC):
                okay = 0 #but not in C
        if( okay == 1):
            contracted.append(idxA)
    return contracted
