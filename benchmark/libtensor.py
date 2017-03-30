def swap(s, i, j):
    lst = list(s);
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)

def getPermuteStr(astr,bstr,cstr):
    outputOrder = ""
    for c in astr:
        if( cstr.find(c) != -1 ):
            outputOrder += c
    for c in bstr:
        if( cstr.find(c) != -1 ):
            outputOrder += c
    if( outputOrder != cstr ):
        permuteStr = "permutation<%d>()"%(len(cstr))
        while( outputOrder != cstr ):
            # find first pos which differs
            posA = -1
            posB = -1
            for pos in range(0, len(cstr)):
                if( outputOrder[pos] != cstr[pos] ):
                    posA = pos
                    posB = outputOrder.find(cstr[pos])
                    break
            outputOrder = swap(outputOrder, posA, posB)
            permuteStr += ".permute(%d,%d)"%(posA,posB) # swap two indices

        return "(%s)"%permuteStr 
    else:
        return ""
def gen(size, astr,bstr, cstr, dataType, numThreads):
    astr = astr[::-1]
    bstr = bstr[::-1]
    cstr = cstr[::-1]

    numFreeA = 0
    numFreeB = 0
    for c in cstr:
        if( astr.find(c) != -1 ): #contracted indice
            numFreeA += 1
        else:
            numFreeB += 1
    numContracted = 0
    contractedIndices = []
    for c in astr:
        if( bstr.find(c) != -1 ): #contracted indice
            numContracted += 1
            contractedIndices.append(c)
    flops = 1.0
    for c in size:
      flops *= size[c]

    code = "#include <cstdlib>\n"
    code += "#include <iostream>\n"
    code += "#include <mkl.h>\n"
    code += "#include <omp.h>\n"
    code += "#include <libutil/timings/timer.h>\n"
    code += "#include <libutil/thread_pool/thread_pool.h>\n"
    code += "#include <libtensor/libtensor.h>\n"
    code += "#include <libtensor/core/batching_policy_base.h>\n"
    code += "#include <libtensor/block_tensor/btod_set.h>\n"
    code += "#include <libtensor/block_tensor/btod_contract2.h>\n"
    code += "\n"
    code += "static void trashCache(float* trash1, float* trash2, int nTotal){\n"
    code += "   for(int i = 0; i < nTotal; i ++) \n"
    code += "      trash1[i] += 0.99 * trash2[i];\n"
    code += "}\n"
    code += "using namespace libtensor;\n"
    code += "\n"
    code += "void warmup() {\n"
    code += "\n"
    code += "    double a[128*128], b[128*128], c[128*128];\n"
    code += "    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128,\n"
    code += "        1.0, a, 128, b, 128, 1.0, c, 128);\n"
    code += "}\n"
    code += "\n"
    code += "int run_bench() {\n"
    code += "\n"
    code += "    float *trash1, *trash2;\n"
    code += "    int nTotal = 1024*1024*100;\n"
    code += "    trash1 = (float*) malloc(sizeof(float)*nTotal);\n"
    code += "    trash2 = (float*) malloc(sizeof(float)*nTotal);\n"
    for c in size: 
      code += "    bispace<1> s%s(%d);\n"%(c,size[c])
    bispaceA = ""
    for c in astr: 
        bispaceA += "s%s|"%c
    bispaceB = ""
    for c in bstr: 
        bispaceB += "s%s|"%c
    bispaceC = ""
    for c in cstr: 
        bispaceC += "s%s|"%c
    floatType = ""
    if( dataType == 's' ):
        floatType = ", float"
    code += "    bispace<%d> sA(%s);\n"%(len(astr), bispaceA[:-1])
    code += "    dense_tensor<%d%s,double,allocator<double> > A(sA.get_bis().get_dims());\n"%(len(astr), floatType)
    code += "    bispace<%d> sB(%s);\n"%(len(bstr), bispaceB[:-1])
    code += "    dense_tensor<%d%s,double,allocator<double> > B(sB.get_bis().get_dims());\n"%(len(bstr), floatType)
    code += "    bispace<%d> sC(%s);\n"%(len(cstr), bispaceC[:-1])
    code += "    dense_tensor<%d%s,double,allocator<double> > C(sC.get_bis().get_dims());\n"%(len(cstr), floatType)
    code += "    tod_set<%d>(0.55).perform(true, A);\n"%len(astr)
    code += "    tod_set<%d>(2.0).perform(true, B);\n"%len(bstr)
    code += "\n"
    code += "    //free A, free B, cont\n"
    permuteStr = getPermuteStr(astr,bstr,cstr)
    code += "    contraction2<%d, %d, %d> contr%s;\n"%(numFreeA, numFreeB, numContracted,permuteStr)
    for c in contractedIndices:
        posA = astr.find(c)
        posB = bstr.find(c)
        code += "    contr.contract(%d, %d);\n"%(posA,posB)
    code += "\n"
    code += "    libutil::thread_pool tp(%d, %d);\n"%(numThreads, numThreads)
    code += "    tp.associate();\n"
    code += "\n"
    code += "    size_t buf_blk = 3;\n"
    code += "    batching_policy_base::set_batch_size(buf_blk);\n"
    code += "\n"
    code += "    double minTime = 1e100;\n"
    code += "    for (int i=0; i<3; i++){\n"
    code += "      trashCache(trash1, trash2, nTotal);\n"
    code += "      double t = omp_get_wtime();\n"
    code += "      tod_contract2<%d, %d, %d>(contr, A, B).perform(true, C);\n"%(numFreeA, numFreeB, numContracted)
    code += "      t = omp_get_wtime() - t;\n"
    code += "      minTime = (minTime < t) ? minTime : t;\n"
    code += "    }\n"
    code += "    double flops = 2.E-9 * %f; \n"%flops
    code += "    printf(\"%s-%s-%s %%.2lf seconds/lib, %%.2lf GF\\n\",minTime, flops/minTime);\n"%(cstr[::-1],astr[::-1],bstr[::-1])
    code += "\n"
    code += "    tp.dissociate();\n"
    code += "    return 0;\n"
    code += "}\n"
    code += "\n"
    code += "\n"
    code += "int main(int argc, char **argv) {\n"
    code += "\n"
    code += "    warmup();\n"
    code += "    return run_bench();\n"
    code += "}\n"
    code += "\n"
    return code
