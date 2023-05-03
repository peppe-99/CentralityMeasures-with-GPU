betweenness-centrality-sequential: betweenness-centrality.o utils.o
	gcc betweenness-centrality.o utils.o -o betweenness-centrality-sequential.out
	./betweenness-centrality-sequential.out

closeness-centrality-sequential: closeness-centrality.o utils.o
	gcc closeness-centrality.o utils.o -o closeness-centrality-sequential.out
	./closeness-centrality-sequential.out

closeness-centrality-GPU: closeness-centrality-GPU.o utils-GPU.o
	nvcc closeness-centrality-GPU.o utils.o -o closeness-centrality-GPU.out -lcublas
	./closeness-centrality-GPU.out

degree-centrality-sequential: degree-centrality.o utils.o
	gcc degree-centrality.o utils.o -o degree-centrality-sequential.out
	./degree-centrality-sequential.out

degree-centrality-GPU: degree-centrality-GPU.o utils-GPU.o
	nvcc degree-centrality-GPU.o utils.o -o degree-centrality-GPU.out
	./degree-centrality-GPU.out

betweenness-centrality.o:
	gcc -c ./CentralityMeasures/BetweennessCentrality/betweenness-centrality.c

closeness-centrality.o:
	gcc -c ./CentralityMeasures/ClosenessCentrality/closeness-centrality.c

closeness-centrality-GPU.o:
	nvcc -c ./CentralityMeasures/ClosenessCentrality/closeness-centrality-GPU.cu

degree-centrality.o:
	gcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality.c

degree-centrality-GPU.o:
	nvcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality-GPU.cu

utils.o:
	gcc -c ./lib/utils.c

utils-GPU.o:
	nvcc -c ./lib/utils.cu

clean:
	rm -rf *.o *.out