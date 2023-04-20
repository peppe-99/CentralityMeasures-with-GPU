
closeness-centrality-sequential: closeness-centrality.o utils.o
	gcc closeness-centrality.o utils.o -o closeness-centrality-sequential.out
	./closeness-centrality-sequential.out

degree-centrality-sequential: degree-centrality.o utils.o
	gcc degree-centrality.o utils.o -o degree-centrality-sequential.out
	./degree-centrality-sequential.out

degree-centrality-GPU: degree-centrality-GPU.o utils.o
	nvcc degree-centrality-GPU.o utils.o -o degree-centrality-GPU

closeness-centrality.o:
	gcc -c ./CentralityMeasures/ClosenessCentrality/closeness-centrality.c

degree-centrality.o:
	gcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality.c

degree-centrality-GPU.o:
	nvcc -c ./CentralityMeasures/DegreeCentrality/degree-centrality-GPU.cu

utils.o:
	gcc -c ./lib/utils.c

clean:
	rm -rf *.o *.out
	clear