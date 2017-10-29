import numpy as np
import actions
import histoperations as hist


def threshold(matrix, value):
    new_matrix = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            new_matrix[i, j] = 255 if matrix[i, j] >= value else 0

    return new_matrix


def global_threshold(matrix,self):

    new_matrix = np.zeros(matrix.shape)

    delta_umbral = 0.5
    new_umbral_value = 0

    value = 80

    # g1 para los valores que son menor al umbral
    count_g1=0
    count_g2=0
    sum_g1=0
    sum_g2=0

    print("aca")
    print("value: " + str(value))
    print("new_umbral_value: " + str(new_umbral_value))
    while abs(value - new_umbral_value) > delta_umbral:
        print("Entro")

        if new_umbral_value != 0:
            value = new_umbral_value

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                if matrix[i, j] <= value:
                    count_g1 += 1
                    sum_g1 += matrix[i, j]
                else:
                    count_g2 += 1
                    sum_g2 += matrix[i, j]



        # deshardcodear el tipo de la imagen!!!
        print("count_g1 " + str(count_g1) )
        print("count_g2 " + str(count_g2) )
        print("sum_g1 " + str(sum_g1) )
        print("sum_g2 " + str(sum_g2) )

        m1 = (1/count_g1) * sum_g1
        m2 = (1/count_g2) * sum_g2
        new_umbral_value = 0.5 * (m1 + m2)
        # Restauro valores para la proxima iteracion
        count_g1 = 0
        count_g2 = 0
        sum_g1 = 0
        sum_g2 = 0

        print("El nuevo valor de T es: " + str(new_umbral_value))

    return threshold(matrix,new_umbral_value)


def otsu(matrix):
    matrix = np.array(matrix,dtype=np.uint8)
    buck = hist.get_bucket(matrix.flatten()) / (matrix.shape[0] * matrix.shape[1])
    acum = hist.get_acum(buck)
    m = np.zeros(256)
    for i in range(1,256):
        m[i] = i*buck[i] + m[i-1]
    mg = m[255]

    var = np.zeros(256)
    max = 0
    max_val = pow(mg*acum[0] - m[0], 2) / (acum[0] * (1 - acum[0]))
    for i in range(1,256):
        var[i] = pow(mg*acum[i] - m[i], 2) / (acum[i] * (1 - acum[i]))
        if var[i] > max_val:
            max = i
            max_val = var[i]

    print("EL METODO DE OTSU DEVUELVE COMO MAX: " + str(max))
    return threshold(matrix, max)


def otsu_threshold(matrix):
    matrix = np.array(matrix, dtype=np.uint8)
    buck = hist.get_bucket(matrix.flatten()) / (matrix.shape[0] * matrix.shape[1])
    acum = hist.get_acum(buck)
    m = np.zeros(256)
    for i in range(1, 256):
        m[i] = i * buck[i] + m[i - 1]
    mg = m[255]

    var = np.zeros(256)
    max = 0
    max_val = pow(mg * acum[0] - m[0], 2) / (acum[0] * (1 - acum[0]))
    for i in range(1, 256-1):
        var[i] = pow(mg * acum[i] - m[i], 2) / (acum[i] * (1 - acum[i]))
        if var[i] > max_val:
            max = i
            max_val = var[i]

    print("EL METODO DE OTSU DEVUELVE COMO MAX: " + str(max))
    return max
