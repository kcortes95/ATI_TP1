#http://www.vision.uji.es/courses/Doctorado/FVC/FVC-T5-DetBordes-Parte1-4p.pdf


def get_area(phi):
    if (phi >= 0 and phi < 22.5) or (phi >= 157.5 and phi < 180):
        return 0
    if phi >= 22.5 and phi < 67.5:
        return 45
    if phi >= 67.5 and phi < 112.5:
        return 90
    if phi >= 112.5 and phi < 157.5:
        return 135

#matrix_directions: matriz de direcciones. Es la que tiene todos los valores de áreas de get_area de la matriz.
def supr_no_max(matrix_directions, matrix_original):
    width, height = matrix_original.shape #buscar
    #Quiero una matriz auxiliar con la misma dimensión

    """
    Cual era la definicion de magnitud de borde -> magnitud del gradiente??? (ver supresión de no máximos)
    Rta: si. Calcular el gradiente (magnitud de borde)
    """
    matrix_to_ret = np.zeros(matrix_original.shape)
    for i in range(width):
        for j in range(height):
            #le paso el phi y la posicion en la que está parado
            dirs = get_direction(matrix_directions[i,j], i, j)
            #no se como se hace para acceder al primer valor, creo que es asi
            neigh_1 = matrix_original[dirs[0,0] , dirs[0,1]]
            neigh_2 = matrix_original[dirs[1,0] , dirs[1,1]]

            """
            Si la magnitud de cualquiera de los dos pixels adyacentes es mayor que la del pixel en cuestión, entonces borrarlo como borde (diapositiva)
            """
            if condition(matrix_original[i,j], neigh_1, neigh_2):
                #marcar como borde
                matrix_to_ret[i,j] = matrix_original[i,j]
            else:
                matrix_to_ret[i,j] = 0
                #borrarlo como borde
