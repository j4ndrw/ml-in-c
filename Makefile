main: main.c
	mkdir -p .build && cc -ggdb -o main main.c && mv main .build
