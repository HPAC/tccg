_floatType = "s" #either "s" or "d"
_case = 0 # either 0 or 1

transc = []
transc.append("""C[i1,j,i2] = 2.4 * A[i1,k,i2] * B[j,k] + 1.1 * C[i1,j,i2]
j=1152
i1=24
i2=48
""")
transc.append("""C[i1,j1,i2,j2] = 2.4 * A[i1,k,i2] * B[j1,k,j2] + 1.1 * C[i1,j1,i2,j2]
i1=24
i2=48
j1=24
j2=48
""")
bash = open("transC.sh", "w")

kvalues = range(8,65,8) + [96,112] + range(128,257,32) + [384,512,1024]

bash.write("rm -f tmp.dat\n")

for k in kvalues:
    filename = "transC%d"%k
    bash.write("python ../scripts/tccg %s.tccg --floatType=%s > %s.dat\n"%(filename,_floatType,filename))
    f = open(filename+".tccg","w")
    f.write(transc[_case]+ "k=%d\n"%k)
    f.close()
    bash.write("echo \"%d\" >> tmp.dat\n"%k)
    bash.write("cat %s.dat | grep \"Best Loop\" >> tmp.dat\n"%filename)

bash.write("cat tmp.dat | sed '$!N;s/\\n/ /' > transC.dat\n") #

bash.close()
