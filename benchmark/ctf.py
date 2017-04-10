def genCTF(size, astr,bstr, cstr, dataType):
   asize = ""
   for c in astr:
        asize += "%d,"%size[c]
   asize = asize[:-1]
   bsize = ""
   for c in bstr:
        bsize += "%d,"%size[c]
   bsize = bsize[:-1]
   csize = ""
   csizeTotal = 1
   for c in cstr:
        csize += "%d,"%size[c]
        csizeTotal *= size[c]
   csize = csize[:-1]
   flops = 1.0
   for c in size:
      flops *= size[c]

   code = "#include <ctf.hpp>\n"
   code += "#include <stdio.h>\n"
   code += "using namespace CTF;\n"
   code += "\n"
   code += "static void trashCache(float* trash1, float* trash2, int nTotal){\n"
   code += "   #pragma omp parallel for\n"
   code += "   for(int i = 0; i < nTotal; i ++) \n"
   code += "      trash1[i] += 0.99 * trash2[i];\n"
   code += "}\n"
   code += "void example(int argc, char** argv){\n"
   code += "  \n"
   code += "\n"
   code += "  World dw(MPI_COMM_WORLD, argc, argv);\n"
   code += "  int dimA = %d;\n"%len(astr)
   code += "  int shapeA[dimA];\n"
   code += "  int sizeA[] = {%s};\n"%asize
   code += "  for( int i = 0; i < dimA; i++ )\n"
   code += "    shapeA[i] = NS;\n"
   code += "  int dimB = %d;\n"%len(bstr)
   code += "  int shapeB[dimB];\n"
   code += "  int sizeB[] = {%s};\n"%bsize
   code += "  for( int i = 0; i < dimB; i++ )\n"
   code += "    shapeB[i] = NS;\n"
   code += "  int dimC = %d;\n"%len(cstr)
   code += "  int shapeC[dimC];\n"
   code += "  int sizeC[] = {%s};\n"%csize
   code += "  for( int i = 0; i < dimC; i++ )\n"
   code += "   shapeC[i] = NS;\n"
   code += "\n"
   code += "  float *trash1, *trash2;\n"
   code += "  int nTotal = 1024*1024*100;\n"
   code += "  trash1 = (float*) malloc(sizeof(float)*nTotal);\n"
   code += "  trash2 = (float*) malloc(sizeof(float)*nTotal);\n"
   code += "  //* Creates distributed tensors initialized with zeros\n"
   if( dataType == 'd' ):
       code += "  Tensor<double> A(dimA, sizeA, shapeA, dw);\n"
       code += "  Tensor<double> B(dimB, sizeB, shapeB, dw);\n"
       code += "  Tensor<double> C(dimC, sizeC, shapeC, dw);\n"
   else:
       code += "  Tensor<float> A(dimA, sizeA, shapeA, dw);\n"
       code += "  Tensor<float> B(dimB, sizeB, shapeB, dw);\n"
       code += "  Tensor<float> C(dimC, sizeC, shapeC, dw);\n"
   code += "\n"
   code += "\n"
   code += "  double minTime = 1e100;\n"
   code += "  for (int i=0; i<3; i++){\n"
   code += "     trashCache(trash1, trash2, nTotal);\n"
   code += "     double t = MPI_Wtime();\n"
   code += "     C[\"%s\"] = A[\"%s\"]*B[\"%s\"];\n"%(cstr,astr,bstr)
   code += "     t = MPI_Wtime() - t;\n"
   code += "     minTime = (minTime < t) ? minTime : t;\n"
   code += "  }\n"
   code += "  double flops = 2.E-9 * %f; \n"%flops
   code += "  printf(\"%s-%s-%s %%.2lf seconds/GEMM, %%.2lf GF\\n\",minTime, flops/minTime);\n"%(cstr,astr,bstr)
   code += " \n"
   #code += "  int64_t size;\n"
   #code += "  float* data = C.get_raw_data(&size);\n"
   #code += "  printf(\"%%d %d\\n\",size);\n"%csizeTotal
   #code += "  FILE *f = fopen(\"tmp.out\",\"w\");\n"
   #code += "  for (int i=0; i<size; i++){\n"
   #code += "     fprintf(f,\"%.2e \",data[i]);\n"
   #code += "  }\n"
   #code += "  fclose(f);\n"
   code += "  free(trash1);\n"
   code += "  free(trash2);\n"
   code += "} \n"
   code += "\n"
   code += "int main(int argc, char ** argv){\n"
   code += "\n"
   code += "  MPI_Init(&argc, &argv);\n"
   code += "\n"
   code += "  example(argc, argv);\n"
   code += "\n"
   code += "  MPI_Finalize();\n"
   code += "  return 0;\n"
   code += "}\n"
   return code
 
