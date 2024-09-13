main: main.c
	mkdir -p .build && cc -o main main.c && mv main .build
