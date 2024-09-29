main: main.c tensor.c variable.c
	mkdir -p .build && cc -ggdb -o main main.c tensor.c variable.c utils.c && mv main .build
