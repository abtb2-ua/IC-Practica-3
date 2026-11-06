OPTS = -Wall -Wextra -Werror -std=c++17

all: maze

maze: maze.cc
	g++ ${OPTS} maze.cc -o maze

# Hay que poner el tamaño delante del .maze para probar dicho laberinto
run: maze
	./maze -f 18.maze

clean:
	rm -f maze *~
