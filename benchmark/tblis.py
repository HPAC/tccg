def genTBLIS(size, astr,bstr, cstr, dataType):
   asize = ""
   astride= ""
   ld = 1
   asizeTotal = 1
   for c in astr:
        asize += "%d,"%size[c]
        asizeTotal *= size[c]
        astride += "%d,"%ld
        ld *= size[c]
   asize = asize[:-1]
   astride = astride[:-1]
   bsize = ""
   bstride= ""
   ld = 1
   bsizeTotal = 1
   for c in bstr:
        bsize += "%d,"%size[c]
        bsizeTotal *= size[c]
        bstride += "%d,"%ld
        ld *= size[c]
   bsize = bsize[:-1]
   bstride = bstride[:-1]
   csize = ""
   cstride= ""
   csizeTotal = 1
   ld = 1
   for c in cstr:
        csize += "%d,"%size[c]
        csizeTotal *= size[c]
        cstride += "%d,"%ld
        ld *= size[c]
   csize = csize[:-1]
   cstride = cstride[:-1]
   flops = 1.0
   for c in size:
      flops *= size[c]

   code = "#include <tblis/tblis.h>\n"
   code += "#include <stdio.h>\n"
   code += "#include <stdlib.h>\n"
   code += "#include <omp.h>\n"
   code += "\n"
   code += "void trashCache(float* trash1, float* trash2, int nTotal){\n"
   code += "   #pragma omp parallel for\n"
   code += "   for(int i = 0; i < nTotal; i ++) \n"
   code += "      trash1[i] += 0.99 * trash2[i];\n"
   code += "}\n"
   code += "void example(int argc, char** argv)\n{\n"

   # -------- A --------
   code += "  len_type sizeA[] = {%s};\n"%asize
   code += "  stride_type stride_A[] = {%s};\n"%astride
   code += "  tblis_tensor A;\n"
   if( dataType == 'd' ):
       code += "  A.type = TYPE_DOUBLE;\n"
       #code += "  A.scalar.data.%s = 1.0;\n"%dataType
       code += "  *(double*)A.scalar = 1.0;\n"  # WORKING with commit 9bf5871a28e66e
       code += "  posix_memalign((void**) &A.data, 64, sizeof(double) * %d);\n"%asizeTotal
       code += "  double* Adata = A.data;\n"
   else:
       code += "  A.type = TYPE_FLOAT;\n"
       #code += "  A.scalar.data.%s = 1.0;\n"%dataType
       code += "  *(double*)A.scalar = 1.0;\n"
       code += "  posix_memalign((void**) &A.data, 64, sizeof(float) * %d);\n"%asizeTotal
       code += "  float* Adata = A.data;\n"
   code += "  A.ndim = %d;\n"%len(astr)
   code += "  A.len = sizeA;\n"
   code += "  A.stride = stride_A;\n"
   code += "  #pragma omp parallel for\n"
   code += "  for(len_type i = 0; i < %d; i++)\n"%asizeTotal
   code += "     Adata[i] = (((i+1)*3)%909) / 908.;\n"
   # -------- B --------
   code += "  len_type sizeB[] = {%s};\n"%bsize
   code += "  stride_type stride_B[] = {%s};\n"%bstride
   code += "  tblis_tensor B;\n"
   if( dataType == 'd' ):
       code += "  B.type = TYPE_DOUBLE;\n"
       #code += "  B.scalar.data.%s = 1.0;\n"%dataType
       code += "  *(double*)B.scalar = 1.0;\n"
       code += "  posix_memalign((void**) &B.data, 64, sizeof(double) * %d);\n"%bsizeTotal
       code += "  double* Bdata = B.data;\n"
   else:
       code += "  B.type = TYPE_FLOAT;\n"
       code += "  *(double*)B.scalar = 1.0;\n"
       #code += "  B.scalar.data.%s = 1.0;\n"%dataType
       code += "  posix_memalign((void**) &B.data, 64, sizeof(float) * %d);\n"%bsizeTotal
       code += "  float* Bdata = B.data;\n"
   code += "  B.ndim = %d;\n"%len(bstr)
   code += "  B.len = sizeB;\n"
   code += "  B.stride = stride_B;\n"
   code += "  #pragma omp parallel for\n"
   code += "  for(len_type i = 0; i < %d; i++)\n"%bsizeTotal
   code += "     Bdata[i] = (((i+1)*7)%909) / 908.0;\n"
   # -------- C --------
   code += "  len_type sizeC[] = {%s};\n"%csize
   code += "  stride_type stride_C[] = {%s};\n"%cstride
   code += "  tblis_tensor C;\n"
   if( dataType == 'd' ):
       code += "  C.type = TYPE_DOUBLE;\n"
       code += "  *(double*)C.scalar = 0.0;\n"
       #code += "  C.scalar.data.%s = 0.0;\n"%dataType
       code += "  posix_memalign((void**) &C.data, 64, sizeof(double) * %d);\n"%csizeTotal
       code += "  double* Cdata = C.data;\n"
   else:
       code += "  C.type = TYPE_FLOAT;\n"
       code += "  *(double*)C.scalar = 0.0;\n" 
       #code += "  C.scalar.data.%s = 0.0;\n"%dataType
       code += "  posix_memalign((void**) &C.data, 64, sizeof(float) * %d);\n"%csizeTotal
       code += "  float* Cdata = C.data;\n"
   code += "  C.ndim = %d;\n"%len(cstr)
   code += "  C.len = sizeC;\n"
   code += "  C.stride = stride_C;\n"
   code += "  #pragma omp parallel for\n"
   code += "  for(len_type i = 0; i < %d; i++)\n"%csizeTotal
   code += "     Cdata[i] = (((i+1)*11)%909) / 908.;\n"

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
   code += "     tblis_tensor_mult(NULL, NULL, &A, \"%s\", &B, \"%s\", &C, \"%s\");\n"%(astr,bstr,cstr)
   code += "     t = omp_get_wtime() - t;\n"
   code += "     minTime = (minTime < t) ? minTime : t;\n"
   code += "  }\n"
   code += "  double flops = 2.E-9 * %f; \n"%flops
   code += "  printf(\"%s-%s-%s %%.2lf seconds/GEMM, %%.2lf GF\\n\",minTime, flops/minTime);\n"%(cstr,astr,bstr)
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
 
