import shutil, os
def func1(name):
	f = open(name)
	g = open("copia"+name,"w")
	linea = f.read()
	for i in range(6721):
		g.write(linea)
	g.close()
	f.close()

def func2(name):
	ruta = os.getcwd() + os.sep
	origen = ruta + name
	destino = ruta + "copia"+name
	for i in range(50):
		if os.path.exists(origen):
		    with open(origen, 'rb') as forigen:
		        with open(destino, 'wb') as fdestino:
		            shutil.copyfileobj(forigen, fdestino)
		            print("Archivo copiado")


func1("Adenovirus-C.txt")