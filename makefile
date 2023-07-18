sgc: sgc.o utils-GPU.o
	nvcc sgc.o utils.o -o sgc.out -lcublas
	./sgc.out

bc-CPU: bc.o utils.o
	gcc bc.o utils.o -o bc-CPU.out
	./bc-CPU.out

bc-CPU-CSR: bc-CSR.o utils.o
	gcc bc-CSR.o utils.o -o bc-CPU-CSR.out
	./bc-CPU-CSR.out

bc-GPU: bc-GPU.o utils-GPU.o
	nvcc bc-GPU.o utils.o -o bc-GPU.out
	./bc-GPU.out

bc-GPU-CSR: bc-GPU-CSR.o utils-GPU.o
	nvcc bc-GPU-CSR.o utils.o -o bc-GPU-CSR.out
	./bc-GPU-CSR.out

cc-CPU: cc.o utils.o
	gcc cc.o utils.o -o cc-CPU.out
	./cc-CPU.out

cc-CPU-CSR: cc-CSR.o utils.o
	gcc cc-CSR.o utils.o -o cc-CPU-CSR.out
	./cc-CPU-CSR.out

cc-GPU: cc-GPU.o utils-GPU.o
	nvcc cc-GPU.o utils.o -o cc-GPU.out -lcublas
	./cc-GPU.out

cc-GPU-CSR: cc-GPU-CSR.o utils-GPU.o
	nvcc cc-GPU-CSR.o utils.o -o cc-GPU-CSR.out
	./cc-GPU-CSR.out

dc-CPU: dc.o utils.o
	gcc dc.o utils.o -o dc-CPU.out
	./dc-CPU.out

dc-CPU-CSR: dc-CSR.o utils.o
	gcc dc-CSR.o utils.o -o dc-CPU-CSR.out
	./dc-CPU-CSR.out

dc-GPU: dc-GPU.o utils-GPU.o
	nvcc dc-GPU.o utils.o -o dc-GPU.out
	./dc-GPU.out

dc-GPU-CSR: dc-GPU-CSR.o utils-GPU.o
	nvcc dc-GPU-CSR.o utils.o -o dc-GPU-CSR.out
	./dc-GPU-CSR.out

sgc.o:
	nvcc -c src/SubGraphCentrality/subgraph-centrality.cu -o sgc.o

bc.o:
	gcc -c src/BetweennessCentrality/betweenness-centrality.c -o bc.o

bc-CSR.o:
	gcc -c src/BetweennessCentrality/betweenness-centrality-CSR.c -o bc-CSR.o

bc-GPU.o:
	nvcc -c src/BetweennessCentrality/betweenness-centrality-GPU.cu -o bc-GPU.o

bc-GPU-CSR.o:
	nvcc -c src/BetweennessCentrality/betweenness-centrality-GPU-CSR.cu -o bc-GPU-CSR.o

cc.o:
	gcc -c src/ClosenessCentrality/closeness-centrality.c -o cc.o

cc-CSR.o:
	gcc -c src/ClosenessCentrality/closeness-centrality-CSR.c -o cc-CSR.o

cc-GPU.o:
	nvcc -c src/ClosenessCentrality/closeness-centrality-GPU.cu -o cc-GPU.o

cc-GPU-CSR.o:
	nvcc -c src/ClosenessCentrality/closeness-centrality-GPU-CSR.cu -o cc-GPU-CSR.o

dc.o:
	gcc -c src/DegreeCentrality/degree-centrality.c -o dc.o

dc-CSR.o:
	gcc -c src/DegreeCentrality/degree-centrality-CSR.c -o dc-CSR.o

dc-GPU.o:
	nvcc -c src/DegreeCentrality/degree-centrality-GPU.cu -o dc-GPU.o

dc-GPU-CSR.o:
	nvcc -c src/DegreeCentrality/degree-centrality-GPU-CSR.cu -o dc-GPU-CSR.o

utils.o:
	gcc -c include/utils.c

utils-GPU.o:
	nvcc -c include/utils.cu

clean:
	rm -rf *.o *.out