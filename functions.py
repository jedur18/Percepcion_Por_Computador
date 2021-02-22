import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from copy import deepcopy
from medpy.io import load, header
from medpy.graphcut import graph_from_voxels
from medpy.graphcut.energy_voxel import boundary_difference_exponential
import skimage.filters as skimage_filter
from matplotlib.path import Path

def comparation(img_seg,groundt):
    """
    Funcion que recive una imagen segmentada y su groundtruth, y muestra por
    pantalla las métircas de precision, recall y F1
    """
    try:
        img = foreground_white(img_seg)
        img = (img/np.amax(img).astype(int))
    except:
        img = img_seg
    groundt = cv2.resize(groundt, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    truePositive = 0
    falsePositive = 0
    falseNegative = 0
    for i in range(groundt.shape[0]):
        for j in range(groundt.shape[1]):
            if groundt[i][j]==1 and img[i][j]==1:
                truePositive+=1
            elif img[i][j]==1:
                falsePositive+=1
            elif groundt[i][j]==1:
                falseNegative+=1
                
    trueNegative=img.size-truePositive-falseNegative-falsePositive

    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    f1 = 2*(precision*recall)/(precision+recall)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)


def detect_plate_label(segments):
    n_contours = np.max(segments)
    contours_size = []
    plate_label = []
    segmented_image = np.zeros(segments.shape)
    for i in range(n_contours+1):
        pix_count = 0
        for row in range(segments.shape[0]):
            for col in range(segments.shape[1]):
                if segments[row,col] == i:
                    pix_count +=1
        contours_size.append(pix_count)
    for i in range(len(contours_size)):
        if 6500< contours_size[i] < 8500:
            plate_label.append(i)
            #print('Possible plate located associated to label {}'.format(i))
    return plate_label


def isolate_plate(label,segments,original_img):
    n_contours = np.max(segments)
   # segmented_plate = np.zeros(segments.shape)
    segmented_plate = original_img.copy()
    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            if segments[row,col] == label:
                segmented_plate[row,col] = original_img[row,col]
            else:
                segmented_plate[row,col] = 0
    return segmented_plate


def foreground_white(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_transf = image.copy()
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row,col] != 0:
                image_transf[row,col] = 255
    return image_transf


negro = np.array([0,0,0])
blanco = np.array([255,255,255])
color_fore = np.array([255,0,255])
color_back = np.array([0,255,255])

def imreadBinary(path):
    imagen = io.imread(path)
    outputBack =  (imagen == color_back).all(axis=-1)
    outputBack += (imagen == [1,255,255]).all(axis=-1)
    outputBack += (imagen == [0,254,255]).all(axis=-1)
    outputBack += (imagen == [1,254,255]).all(axis=-1)
    outputBack += (imagen == [0,255,254]).all(axis=-1)
    outputBack += (imagen == [1,255,254]).all(axis=-1)
    outputBack += (imagen == [0,254,254]).all(axis=-1)
    outputBack += (imagen == [1,254,254]).all(axis=-1)
    outputBack = outputBack>0
    outputFore =  (imagen == color_fore).all(axis=-1) 
    outputFore += (imagen == [255,1,255]).all(axis=-1)
    outputFore += (imagen == [254,0,255]).all(axis=-1) 
    outputFore += (imagen == [254,1,255]).all(axis=-1) 
    outputFore += (imagen == [255,0,254]).all(axis=-1) 
    outputFore += (imagen == [255,1,254]).all(axis=-1) 
    outputFore += (imagen == [254,0,254]).all(axis=-1)
    outputFore += (imagen == [254,1,254]).all(axis=-1)
    outputFore  = outputFore>0  
    return outputBack, outputFore

def marcasBackgroundForeground(imagen,foreground,background):    
    output = deepcopy(imagen)
    output[(foreground)] = color_fore
    output[(background)] = color_back
    #io.imshow_collection([output,foreground,background])
    return output


def resultadoMarcado(imagen,marca):
    output = deepcopy(imagen)
    output[(marca == blanco).all(axis = -1)] = (np.array(output[(marca == blanco).all(axis = -1)])+2*color_fore)/3
    output[(marca == negro).all(axis = -1)] = (np.array(output[(marca == negro).all(axis = -1)])+2*color_back)/3
    return output


# Función para imprimir las dos imagenes juntas
def imshow_compare(img1,img2):
    fig = plt.figure(figsize = (20,20))
    ax0 = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    # Imagen 1
    ax0.imshow(img1,cmap=plt.cm.gray)
    ax0.set_xticks([]), ax0.set_yticks([]) # Para evitar que aparezcan los números en los ejes
    ax0.set_title('Imagen 1')
    # Imagen 2
    ax1.imshow(img2,cmap=plt.cm.gray)
    ax1.set_xticks([]), ax1.set_yticks([])
    ax1.set_title('Imagen 2')

    # Plot
    plt.show()

    
def medpy_a_skimage(imagen):
    """
    Esta funcion pasa una imagen con un shape de (canales,ancho,alto) a una con shape (alto,ancho,canales)
    """
    canales = imagen.shape[0]
    ancho = imagen.shape[1]
    alto = imagen.shape[2]
    salida = np.zeros((alto,ancho,canales),dtype=imagen.dtype)
    for c in range(canales):
        salida[:,:,c] = imagen[c,:,:].transpose()
    return salida

def skimage_a_medpy(imagen):
    """ 
    Esta funcion pasa una imagen con un shape de (alto,ancho,canales) a una con shape (canales,ancho,alto)
    """
    canales = imagen.shape[2]
    ancho = imagen.shape[1]
    alto = imagen.shape[0]
    salida = np.zeros((canales,ancho,alto),dtype=imagen.dtype)
    for c in range(canales):
        salida[c,:,:] = imagen[:,:,c].transpose()
    return salida


def segmentacionGraphCut(imagen,foreground,background,sigma=15.0,spacing = (1.0,1.0,1.0)):
    print("Pasamos las imagenes a medpy")
    coche_medpy = skimage_a_medpy(imagen)
    cocheFore_medpy = foreground.transpose()
    cocheBack_medpy = background.transpose()
    print("Creamos el grafo")
    grafo = graph_from_voxels(np.array([cocheFore_medpy,cocheFore_medpy,cocheFore_medpy]), # Marcas de foreground
                              np.array([cocheBack_medpy,cocheBack_medpy,cocheBack_medpy]), # Marcas de backgorund
                              boundary_term = boundary_difference_exponential,  #Calculo pesos bordes
                              boundary_term_args = (coche_medpy, sigma, spacing));
    print("Calculamos el maxflow")
    grafo.maxflow()
    print("Calculmaos el resultado")
    resultado_medpy = np.zeros(coche_medpy.size, dtype=np.bool)
    for idx in range(len(resultado_medpy)):
        if grafo.termtype.SINK == grafo.what_segment(idx):
            resultado_medpy[idx] = False 
        else:
            resultado_medpy[idx] = True   
    resultado_medpy = resultado_medpy.reshape(coche_medpy.shape)
    resultado = medpy_a_skimage(resultado_medpy)
    mask = np.zeros([resultado.shape[0],resultado.shape[1]],dtype=bool)
    mask[np.count_nonzero(resultado == True, axis=2)>1.5] = True 
    resultado=medpy_a_skimage(np.array([mask.transpose(),mask.transpose(),mask.transpose()]))
    resultado = resultadoMarcado(imagen,resultado*255)
    io.imshow(resultado)
    return mask, resultado



def create_external_edge_force_gradients_from_img( img, sigma=30.):
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.
    img: ndarray
        The image.
    """
    # Gaussian smoothing.
    smoothed = skimage_filter.gaussian( (img-img.min()) / (img.max()-img.min()), sigma )
    # Gradient of the image in x and y directions.
    giy, gix = np.gradient( smoothed )
    
    # Gradient magnitude of the image.
    gmi = (gix*2 + giy*2)*(0.5)
    # Normalize. This is crucial (empirical observation).
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
    # Gradient of gradient magnitude of the image in x and y directions.
    ggmiy, ggmix = np.gradient( gmi )
    ext_force_x, ext_force_y = ggmiy, ggmix

    def fx(x, y):
        """
        Return external edge force in the x direction.
        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ext_force_x[ (y.round().astype(int), x.round().astype(int)) ]

    def fy(x, y):
        """
        Return external edge force in the y direction.
        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ext_force_y[ (y.round().astype(int), x.round().astype(int)) ]

    return fx, fy


def create_A(a, b, N):
    """
    a: float
    alpha parameter
    b: float
    beta parameter
    N: int
    N is the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
    """
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A
def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    """
    x: ndarray
        intial x coordinates of the snake
    y: ndarray
        initial y coordinates of the snake
    a: float
        alpha parameter
    b: float
        beta parameter
    fx: callable
        partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.
    fy: callable
        see fx.
    gamma: float
        step size of the iteration
    
    n_iters: int
        number of times to iterate the snake
    return_all: bool
        if True, a list of (x,y) coords are returned corresponding to each iteration.
        if False, the (x,y) coords of the last iteration are returned.
    """
    A = create_A(a,b,x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)

    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append( (x_.copy(),y_.copy()) )

    if return_all:
        return snakes
    else:
        return (x,y)
    
def plot_snakes(img, x, y, snakes):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(img.shape[0],0)
    ax.plot(np.r_[x,x[0]], np.r_[y,y[0]], c=(0,1,0), lw=2)

    for i, snake in enumerate(snakes):
        if i % 10 == 0:
            ax.plot(np.r_[snake[0], snake[0][0]], np.r_[snake[1], snake[1][0]], c=(0,0,1), lw=2)

    # Plot the last one a different color.
    ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)

    plt.show()

def get_mask(img, x, y, snakes):
    width, height = img.shape
    x_snake = np.r_[snakes[-1][0], snakes[-1][0][0]]
    y_snake = np.r_[snakes[-1][1], snakes[-1][1][0]]
    snake_final = np.hstack((x_snake.reshape((-1,1)), y_snake.reshape((-1,1))))
    polygon = []
    for x, y in snake_final:
        polygon.append((x, y))

    poly_path = Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

    mask = poly_path.contains_points(coors)
    mask = mask.reshape(height, width).T

    mask = mask.astype(int) 

    return mask