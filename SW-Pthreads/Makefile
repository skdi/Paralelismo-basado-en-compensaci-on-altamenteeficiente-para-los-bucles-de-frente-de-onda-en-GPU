#!/bin/bash

.PHONY: gpp
gpp:
	g++ SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -lpthread -lrt

.PHONY: gppO3
gppO3:
	g++ SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -lpthread -lrt -O3

.PHONY: icpc
icpc:
	icpc SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -lpthread -lrt

.PHONY: icpcO3
icpcO3:
	icpc SmithWaterman.cpp pthread_funcs.cpp functions.cpp globals.cpp -o SmithWaterman -W -Wall -Wextra -lpthread -lrt -O3
