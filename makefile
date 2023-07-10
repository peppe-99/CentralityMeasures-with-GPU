betweenness-centrality-sequential: betweenness-centrality.o utils.o
	gcc betweenness-centrality.o utils.o -o betweenness-centrality-sequential.out
	./betweenness-centrality-sequential.out

betweenness-centrality-GPU: betweenness-centrality-GPU.o utils-GPU.o
	nvcc betweenness-centrality-GPU.o utils.o -o betweenness-centrality-GPU.out -lcublas
	./betweenness-centrality-GPU.out

closeness-centrality-sequential: closeness-centrality.o utils.o
	gcc closeness-centrality.o utils.o -o closeness-centrality-sequential.out
	./closeness-centrality-sequential.out

closeness-centrality-GPU: closeness-centrality-GPU.o utils-GPU.o
	nvcc closeness-centrality-GPU.o utils.o -o closeness-centrality-GPU.out -lcublas
	./closeness-centrality-GPU.out

dc-CPU: dc.o utils.o
	gcc dc.o utils.o -o dc-CPU.out
	./dc-CPU.out

dc-CPU-RCE: dc-RCE.o utils.o
	gcc dc-RCE.o utils.o -o dc-CPU-RCE.out
	./dc-CPU-RCE.out

dc-GPU: dc-GPU.o utils-GPU.o
	nvcc dc-GPU.o utils.o -o dc-GPU.out
	./dc-GPU.out

dc-GPU-RCE: dc-GPU-RCE.o utils-GPU.o
	nvcc dc-GPU-RCE.o utils.o -o dc-GPU-RCE.out
	./dc-GPU-RCE.out

betweenness-centrality.o:
	gcc -c ./CentralityMeasures/BetweennessCentrality/betweenness-centrality.c

betweenness-centrality-GPU.o:
	nvcc -c ./CentralityMeasures/BetweennessCentrality/betweenness-centrality-GPU.cu

closeness-centrality.o:
	gcc -c ./CentralityMeasures/ClosenessCentrality/closeness-centrality.c

closeness-centrality-GPU.o:
	nvcc -c ./CentralityMeasures/ClosenessCentrality/closeness-centrality-GPU.cu

dc.o:
	gcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality.c -o dc.o

dc-RCE.o:
	gcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality-RCE.c -o dc-RCE.o

dc-GPU.o:
	nvcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality-GPU.cu -o dc-GPU.o

dc-GPU-RCE.o:
	nvcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality-GPU-RCE.cu -o dc-GPU-RCE.o

utils.o:
	gcc -c ./lib/utils.c

utils-GPU.o:
	nvcc -c ./lib/utils.cu

clean:
	rm -rf *.o *.out