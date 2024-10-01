main: main.c
	mkdir -p .build
	chmod +x ./build.sh
	./build.sh
	mv main .build
