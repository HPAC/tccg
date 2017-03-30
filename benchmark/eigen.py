def genEigen(size, astr,bstr, cstr, dataType):
   asize = ""
   ld = 1
   asizeTotal = 1
   contractedPairs = []
   for posA in range(len(astr)):
       posB = bstr.find(astr[posA])
       if(posB != -1):
           contractedPairs.append((posA, posB))
       
   outputCeigen = ""
   for c in astr:
       if cstr.find(c) != -1:
           outputCeigen = outputCeigen + c
   for c in bstr:
       if cstr.find(c) != -1:
           outputCeigen = outputCeigen + c
   permC = ""
   if outputCeigen != cstr: # untranspose required?
       for c in cstr:
           permC += str(outputCeigen.find(c)) + ","
       #for c in outputCeigen:
       #    permC += str(cstr.find(c)) + ","
       permC = ".shuffle(Eigen::array<int, %d>{%s})"%(len(cstr),permC[0:-1])

   for c in astr:
        asize += "%d,"%size[c]
        asizeTotal *= size[c]
        ld *= size[c]
   asize = asize[:-1]
   bsize = ""
   ld = 1
   bsizeTotal = 1
   for c in bstr:
        bsize += "%d,"%size[c]
        bsizeTotal *= size[c]
        ld *= size[c]
   bsize = bsize[:-1]
   csize = ""
   csizeTotal = 1
   ld = 1
   for c in cstr:
        csize += "%d,"%size[c]
        csizeTotal *= size[c]
        ld *= size[c]
   csize = csize[:-1]
   flops = 1.0
   for c in size:
      flops *= size[c]

   code = "#include <unsupported/Eigen/CXX11/Tensor>\n"
   code += "#include <stdio.h>\n"
   code += "#include <stdlib.h>\n"
   code += "#include <omp.h>\n"
   code += "\n"
   code += "void trashCache(float* trash1, float* trash2, int nTotal){\n"
   code += "   for(int i = 0; i < nTotal; i ++) \n"
   code += "      trash1[i] += 0.99 * trash2[i];\n"
   code += "}\n"
   code += "void example(int argc, char** argv)\n{\n"

   floatType = "float"
   if( dataType == 'd' ):
       floatType = "double"
   code += "  Eigen::Tensor<%s, %d> a(%s);\n"%(floatType,len(astr),asize)
   code += "  Eigen::Tensor<%s, %d> b(%s);\n"%(floatType,len(bstr),bsize)
   code += "  Eigen::Tensor<%s, %d> c(%s);\n"%(floatType,len(cstr),csize)
   
   contractStr = ""
   for (a,b) in contractedPairs:
       contractStr += "Eigen::IndexPair<int>(%d,%d), "%(a,b)
   code += "  Eigen::array<Eigen::IndexPair<int>, %d> product_dims = { %s };\n"%(len(contractedPairs),contractStr[0:-2])

   code += "\n"
   code += "  float *trash1, *trash2;\n"
   code += "  int nTotal = 1024*1024*100;\n"
   code += "  trash1 = (float*) malloc(sizeof(float)*nTotal);\n"
   code += "  trash2 = (float*) malloc(sizeof(float)*nTotal);\n"
   code += "  //* Creates distributed tensors initialized with zeros\n"
   code += "\n"
   code += "\n"
   code += "  double minTime = 1e100;\n"
   code += "  for (int i=0; i<3; i++){\n"
   code += "     trashCache(trash1, trash2, nTotal);\n"
   code += "     double t = omp_get_wtime();\n"
   #code += "     c%s = a.contract(b, product_dims);\n"%permC
   code += "     c = a.contract(b, product_dims)%s;\n"%permC
   code += "     t = omp_get_wtime() - t;\n"
   code += "     minTime = (minTime < t) ? minTime : t;\n"
   code += "  }\n"
   code += "  double flops = 2.E-9 * %f; \n"%flops
   code += "  printf(\"%s-%s-%s %%.2lf seconds/TC, %%.2lf GF\\n\",minTime, flops/minTime);\n"%(cstr,astr,bstr)
   code += " \n"
   code += "  free(trash1);\n"
   code += "  free(trash2);\n"
   code += "} \n"

   code += "\n"
   code += "int main(int argc, char ** argv){\n"
   code += "\n"
   code += "\n"
   code += "  example(argc, argv);\n"
   code += "\n"
   code += "  return 0;\n"
   code += "}\n"
   return code
 
