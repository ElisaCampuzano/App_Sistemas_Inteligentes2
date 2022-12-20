from flask import Flask, render_template, request
from flask import send_from_directory
import cx_Oracle
import pandas as pd
from datetime import datetime
import os
import io
from flask_cors import CORS
from routes import routes_folder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import math
import cv2
## Import the keras API
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf
import csv
import base64
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


app = Flask(__name__)

app.register_blueprint(routes_folder)

CORS(app)

try:
    connection = cx_Oracle.connect(
        user='INTELIGENTES',
        password='123',
        dsn='localhost:1521/XE',
        encoding='UTF-8'
    )
    print("db conectada")    
except Exception as ex:
    print('Excepción: ',ex)
# finally:
#     connection.close()
#     print('Conexión finalizada')

# para mostrar las imagenes
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
imagenestest = [any]


@app.route('/')
def index():
    # conexión con la base de datos
    sql = """SELECT * FROM DATASET"""
    cursor = connection.cursor()
    cursor.execute(sql)
    dataset = cursor.fetchall()
    print(dataset)
    connection.commit()
    # rows = cursor.fetchall()
    # for row in rows:
    #     print(row)    
    return render_template('index.html', dataset=dataset)


# Red neuronal
@app.route('/create')
def create():

    return render_template('create.html')


@app.route('/store', methods=['POST'])
def storage():
    _nombre = request.files['archivo'].filename
    _archivo = request.files['archivo']
    print("ARCHIVOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    print(_archivo)
    
    # guardar archivo con nombre con fecha y hora
    now = datetime.now()
    tiempo = now.strftime("%Y%M%H%S")
    if _archivo.filename != '':
        nuevoNombreArchivo = tiempo+_archivo.filename
        print("TIPOOOOOOOOOOO")
        print(_archivo.content_type)
        for i in _archivo._parsed_content_type:
            print("iiiiiiiiiiii")
            print(i)
        _archivo.save("uploads/"+nuevoNombreArchivo)

    csv_data = pd.read_csv("uploads/"+nuevoNombreArchivo, sep = ",")
    
    # insertar en base de datos
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO DATASET (NOMBRE, VALORES) 
        values (:nombre, :valores)""", [nuevoNombreArchivo, csv_data.to_json()])
    connection.commit()
    return render_template('index.html')

@app.route('/visualizar', methods=['POST'])
def visualizar():
    # obtiene una imagen
    # imagen = os.path.join(app.config['UPLOAD_FOLDER'], '0/0_0.jpg')
    
    # obtiene todas las imagenes de la ruta
    carpetas = os.listdir('static/dataset')
    IMG_LISTT = []
    for c in carpetas:
        # print(c)
        if c!='.DS_Store':
            IMG_LIST = os.listdir('static/dataset/'+c)
            IMG_LIST2 = ['dataset/'+c+'/' + i for i in IMG_LIST]
            IMG_LISTT += IMG_LIST2

    return render_template('create.html', imagenes=IMG_LISTT)

@app.route('/visualizartest', methods=['POST'])
def visualizartest():    
    # obtiene todas las imagenes de la ruta
    carpetas = os.listdir('static/test')
    IMG_LISTT = []
    for c in carpetas:
        if c!='.DS_Store':
            IMG_LIST = os.listdir('static/test/'+c)
            IMG_LIST2 = ['test/'+c+'/' + i for i in IMG_LIST]
            IMG_LISTT += IMG_LIST2
    imagenestest=IMG_LISTT

    return render_template('create.html', imagenestest=imagenestest)

@app.route('/metodo', methods=['POST'])
def redneuronal():
    # path = 'static/dataset/0/0_0.jpg'

    # imagen=cv2.imread(path,0)

    # if imagen is None:
    #     print("error al cargar la imagen")
    # else:
    #     plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    #     plt.axis("off")
    #     plt.show()

    img_size=28
    #Numero de neuronas de la cnn
    img_size_flat=img_size*img_size
    #Parametrizar la forma de imagenes
    num_chanels=1
    #RGB, HSV -> num_chanels=3
    img_shape=(img_size,img_size,num_chanels)
    num_clases=10
    limiteImagenesPrueba=60
    imagenes,etiquetas,probabilidades=cargarDatos("static/dataset/",num_clases,limiteImagenesPrueba)

    model=Sequential()
    #Capa entrada
    model.add(InputLayer(input_shape=(img_size_flat,)))
    #Reformar imagen
    model.add(Reshape(img_shape))

    #Capas convolucionales
    model.add(Conv2D(kernel_size=5,strides=1,filters=16,padding='same',activation='relu',name='capa_convolucion_1'))
    model.add(MaxPooling2D(pool_size=2,strides=2))

    model.add(Conv2D(kernel_size=5,strides=1,filters=36,padding='same',activation='relu',name='capa_convolucion_2'))
    model.add(MaxPooling2D(pool_size=2,strides=2))

    #Aplanar imagen
    model.add(Flatten())
    #Capa densa
    model.add(Dense(128,activation='relu'))

    #Capa salida
    model.add(Dense(num_clases,activation='softmax'))

    #Compilacion del modelo
    optimizador=Adam(lr=1e-3)
    model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
    )

    #Entrenamiento del modelo
    model.fit(x=imagenes,y=probabilidades,epochs=10,batch_size=100)

    limiteImagenesPrueba=20
    imagenesPrueba,etiquetasPrueba,probabilidadesPrueba=cargarDatos('static/test/',num_clases,limiteImagenesPrueba)
    resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)
    print("{0}: {1:.2%}".format(model.metrics_names[1], resultados[1]))
    printResultados = resultados
    loss = "{0}: {1:.2%}".format(model.metrics_names[0], resultados[0])
    accuracy = "{0}: {1:.2%}".format(model.metrics_names[1], resultados[1])
    #Carpeta y nombre del archivo como se almacenará el modelo
    nombreArchivo='models/modeloReconocimientoNumeros.keras'
    model.save(nombreArchivo)
    model.summary()

    return render_template('create.html', resultados=printResultados, loss=loss, accuracy=accuracy)

@app.route('/probar', methods=['POST'])
def probar():
    categorias=["0","1","2","3","4","5","6","7","8","9"]
    reconocimiento=prediccion()
    categoriaimg = request.form['img-catego']
    nombreimg = request.form['img-nombre']
    imagenPrueba=cv2.imread("static/test/"+categoriaimg+"/"+nombreimg,0)
    print(imagenPrueba)
    indiceCategoria=reconocimiento.predecir(imagenPrueba)
    imagenprueba = os.path.join('static/test/', categoriaimg+"/"+nombreimg)
    print("La imagen cargada es ",categorias[indiceCategoria])
    resultadoImagen = categorias[indiceCategoria]

    imagen=cv2.imread("static/test/"+categoriaimg+"/"+nombreimg,0)

    if imagen is None:
        print("error al cargar la imagen")
    else:
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return render_template('create.html', imagen=imagenprueba, resultadoImagen=resultadoImagen)

def cargarDatos(fase,numeroCategorias,limite):
    imagenesCargadas=[]
    etiquetas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite):
            ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen=cv2.imread(ruta,0)
            imagen=imagen.flatten()
            imagen=imagen/255
            imagenesCargadas.append(imagen)
            etiquetas.append(categoria)
            probabilidades=np.zeros(numeroCategorias)
            probabilidades[categoria]=1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento=np.array(imagenesCargadas)
    etiquetasEntrenamiento=np.array(etiquetas)
    valoresEsperados=np.array(valorEsperado)
    return imagenesEntrenamiento,etiquetasEntrenamiento,valoresEsperados
    # return render_template('create.html')

class prediccion():
    """
    Carga el modelo de la red neuronal de la ruta especificada
    """
    def __init__(self):
        self.rutaModelo="models/modeloReconocimientoNumeros.keras"
        self.model=load_model(self.rutaModelo)
        self.width=28
        self.heigth=28

    def predecir(self,imagen):
        """
            Toma la imagen de entrada y realiza el proceso de predicción
        """
        imagen=cv2.resize(imagen,(self.width,self.heigth))
        imagen=imagen.flatten()
        imagen=np.array(imagen)
        imagenNormalizada=imagen/255
        pruebas=[]
        pruebas.append(imagenNormalizada)
        imagenesAPredecir=np.array(pruebas)
        predicciones=self.model.predict(x=imagenesAPredecir)
        claseMayorValor=np.argmax(predicciones,axis=1)
        print(predicciones)
        print (claseMayorValor)
        return claseMayorValor[0]

    
# K-Means
@app.route('/createKMeans')
def createKMeans():

    return render_template('createKMeans.html')

@app.route('/KMeans', methods=['POST'])
def KMeans():
    num_puntos = int(request.form['puntos'])
    num_clusters = int(request.form['clusteres'])
    num_iteraciones = int(request.form['iteraciones'])
    print('puntos: '+str(num_puntos)+' clusters: '+str(num_clusters)+' iteraciones: '+str(num_iteraciones))

    # Generar los datos y seleccionar los centroides de manera aleatoria
    puntos = tf.constant(np.random.uniform(0, 10, (num_puntos, 2)))
    centroides = tf.Variable(tf.slice(tf.random.shuffle(puntos),[0,0], [num_clusters, -1]))

    # Aumentar una dimensión para poder restar
    puntos_expand = tf.expand_dims(puntos, 0)
    centroides_expand = tf.expand_dims(centroides, 1)

    # Calcular la distancia euclídea y obtener la asignación de cada punto con el número de cluster más cercano
    distancias = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(puntos_expand, centroides_expand)), 2))
    vector_dist_minimas = tf.argmin(distancias, 0)

    # Se calculan los nuevos centroides
    lista = tf.dynamic_partition(puntos, tf.cast(vector_dist_minimas, tf.int32), num_clusters)
    nuevos_centroides = [tf.reduce_mean(punto, 0) for punto in lista]

    # Se asignan los centroides calculados a la variable centroides
    centroides_actualizados = centroides.assign(nuevos_centroides)

    for i in range(num_iteraciones):
        [_, valores_centroides, valores_puntos, valores_distancias] = [centroides_actualizados, centroides, puntos, vector_dist_minimas]
    print ("Centroides finales:\n",valores_centroides.value())

    # Mostrar el resultado gráficamente
    plt.clf()
    img = io.BytesIO()
    plt.scatter(valores_puntos[:, 0], valores_puntos[:, 1], c=valores_distancias, s=40, alpha=1, cmap=plt.cm.rainbow)
    plt.plot(valores_centroides[:, 0], valores_centroides[:, 1], 'kx', markersize=15)
    # plt.show()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('createKMeans.html', centroidesFinales=str(valores_centroides.value()), imagen={ 'imagen': plot_url })

# Regresion lineal

def minCuadrados(datos,x):
    sumatoriaXY = sumXY(datos)
    sumatoriaX = sumX(datos)
    sumatoriaY = sumY(datos)
    sumatoriaX2 = sumX2(datos)

    # m
    m = round(((len(datos)*sumatoriaXY)-(sumatoriaX*sumatoriaY))/((len(datos)*sumatoriaX2)-sumatoriaX**2),2)
    print ('m = ',m)

    # b
    b = round(((sumatoriaY*sumatoriaX2)-(sumatoriaX*sumatoriaXY))/((len(datos)*sumatoriaX2)-sumatoriaX**2),2)
    print ('b = ',b)

    #  y
    y = m*x+b
    return m,b,y

# sumatoria de X
def sumX(datos):
    sumatoriaX = 0
    for i in range(len(datos)):
        sumatoriaX += (datos[i][0])
    return sumatoriaX

# sumatoria de Y
def sumY(datos):
    sumatoriaY = 0
    for i in range(len(datos)):
        sumatoriaY += (datos[i][1])
    return sumatoriaY

# sumatoria de XY
def sumXY(datos):
    sumatoriaXY = 0
    for i in range(len(datos)):
        sumatoriaXY += (datos[i][0]*datos[i][1])
    return sumatoriaXY

# sumatoria de X^2
def sumX2(datos):
    sumatoriaX2 = 0
    for i in range(len(datos)):
        sumatoriaX2 += (datos[i][0]**2)
    return sumatoriaX2


@app.route('/createregresion')
def createregresion():

    return render_template('createregresion.html')

@app.route('/regresion', methods=['POST'])
def regresion():
    datos1 = request.files['datos']
    print('datos regresion')
    print(datos1)
    datos1.save("uploads/datosRegresion.csv")
    
    datos = []
    with open("uploads/datosRegresion.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # cambiar contenidos a flotantes
        for row in reader: # cada fila es una lista
            datos.append(row)
    print(datos)
    lendatos = len(datos)

    x = int(request.form['valor'])
    print('x',str(x))
    m,b,y = minCuadrados(datos,x)
    print ('x = ',x)
    print ('y = ',y)

    # gráfico
    plt.clf()
    imgReg = io.BytesIO()
    for i in range(len(datos)):
        plt.scatter(datos[i][0],datos[i][1],label='linear', color='blue')
        plt.scatter(x,y,label='linear', color='red')
    plt.title('Regresión lineal')
    plt.savefig(imgReg, format='png')
    imgReg.seek(0)
    plot_url = base64.b64encode(imgReg.getvalue()).decode()

    return render_template('createregresion.html', m=m, b=b, x=x, y=round(y,2), imagen={ 'imagen': plot_url }, lendatos=lendatos, datos=datos)

# PCA
@app.route('/createPCA')
def createPCA():

    return render_template('createPCA.html')

@app.route('/PCA', methods=['POST'])
def PCA():
    datos = request.files['datos']
    print('datos PCA')
    print(datos)
    datos.save("uploads/datosPCA.csv")

    df = pd.read_csv('uploads/datosPCA.csv', names=['sepal_length','sepal_width','petal_length','petal_width','species'])
    print(df.tail())

    # Se divide la matriz del dataset en dos partes
    X = df.iloc[1:,0:4].values
    # la submatriz x contiene los valores de las primeras 4 columnas del dataframe y todas las filas

    y = df.iloc[1:,4].values
    # El vector y contiene los valores de la 4 columna (especie)para todas las filas

    #Aplicamos una transformación de los datos para poder aplicar las propiedades de la distribución normal
    X_std = StandardScaler().fit_transform(X)

    # Calculamos la matriz de covarianza
    print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

    #Calculamos los autovalores y autovectores de la matriz y los mostramos
    cov_mat = np.cov(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)

    #  Hacemos una lista de parejas (autovector, autovalor) 
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Ordenamos estas parejas den orden descendiente con la función sort
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visualizamos la lista de autovalores en orden desdenciente
    print('Autovalores en orden descendiente:')
    for i in eig_pairs:
        print(i[0])

    # A partir de los autovalores, calculamos la varianza explicada
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
    with plt.style.context('seaborn-pastel'):
        plt.clf()
        img = io.BytesIO()

        plt.figure(figsize=(6, 4))

        plt.bar(range(4), var_exp, alpha=0.5, align='center',
                label='Varianza individual explicada', color='g')
        plt.step(range(4), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
        plt.ylabel('Varianza Explicada')
        plt.xlabel('Componentes Principales')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.savefig(img)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

    #Generamos la matríz a partir de los pares autovalor-autovector
    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                        eig_pairs[1][1].reshape(4,1)))
    print('Matriz W:\n', matrix_w)
    Y = X_std.dot(matrix_w)

    with plt.style.context('seaborn-whitegrid'):
        plt.clf()
        img2 = io.BytesIO()
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('setosa', 'versicolor', 'virginica'),
                        ('magenta', 'cyan', 'limegreen')):
            plt.scatter(Y[y==lab, 0],
                        Y[y==lab, 1],
                        label=lab,
                        c=col)
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend(loc='lower center')
        plt.tight_layout()

        plt.savefig(img2)
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()

    return render_template('createPCA.html', imagen={ 'imagen': plot_url }, imagen2={ 'imagen': plot_url2 })

# SVM
@app.route('/createSVM')
def createSVM():

    return render_template('createSVM.html')

@app.route('/SVM', methods=['POST'])
def SVM():
    datos1 = request.files['datos']
    print('datos SVM')
    print(datos1)
    datos1.save("uploads/datosSVM.csv")

    datos = pd.read_csv('uploads/datosSVM.csv')
    datos.head(3)

    plt.clf()
    img1 = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(datos.X1, datos.X2, c=datos.y)
    ax.set_title("Datos ESL.mixture")
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()

    # División de los datos en train y test
    # ==============================================================================
    X = datos.drop(columns = 'y')
    y = datos['y']

    X_train, X_test, y_train, y_test = train_test_split(
                                            X,
                                            y.values.reshape(-1,1),
                                            train_size   = 0.8,
                                            random_state = 1234,
                                            shuffle      = True
                                        )

    # Creación del modelo SVM lineal
    # ==============================================================================
    modelo = SVC(C = 100, kernel = 'linear', random_state=123)
    modelo.fit(X_train, y_train)

    # Representación gráfica de los límites de clasificación
    # ==============================================================================
    # Grid de valores
    x = np.linspace(np.min(X_train.X1), np.max(X_train.X1), 50)
    y = np.linspace(np.min(X_train.X2), np.max(X_train.X2), 50)
    Y, X = np.meshgrid(y, x)
    grid = np.vstack([X.ravel(), Y.ravel()]).T

    # Predicción valores grid
    pred_grid = modelo.predict(grid)
    
    plt.clf()
    img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(grid[:,0], grid[:,1], c=pred_grid, alpha = 0.2)
    ax.scatter(X_train.X1, X_train.X2, c=y_train, alpha = 1)

    # Vectores soporte
    ax.scatter(
        modelo.support_vectors_[:, 0],
        modelo.support_vectors_[:, 1],
        s=200, linewidth=1,
        facecolors='none', edgecolors='black'
    )

    # Hiperplano de separación
    ax.contour(
        X,
        Y,
        modelo.decision_function(grid).reshape(X.shape),
        colors = 'k',
        levels = [-1, 0, 1],
        alpha  = 0.5,
        linestyles = ['--', '-', '--']
    )

    ax.set_title("Resultados clasificación SVM lineal")

    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Predicciones test
    # ==============================================================================
    predicciones = modelo.predict(X_test)
    predicciones

    # Accuracy de test del modelo 
    # ==============================================================================
    accuracy = accuracy_score(
                y_true    = y_test,
                y_pred    = predicciones,
                normalize = True
            )
    print("")
    print(f"El accuracy de test es: {100*accuracy}%")

    return render_template('createSVM.html', accuracy=100*accuracy, imagen1={ 'imagen': plot_url1 }, imagen={ 'imagen': plot_url })

# ReglasAsociacion
@app.route('/createReglasAsociacion')
def createReglasAsociacion():

    return render_template('createReglasAsociacion.html')

@app.route('/ReglasAsociacion', methods=['POST'])
def ReglasAsociacion():
    datos1 = request.files['datos']
    print('datos Reglas asociacion')
    print(datos1)
    datos1.save("uploads/datosReglasAsociacion.csv")

    with open('uploads/datosReglasAsociacion.csv') as f:
        reader = csv.reader(f)
        lst = list(reader)
        print(lst)
    dataset = np.array(lst)
    print(dataset)
    lendataset = len(dataset)

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    # print(te_ary)
    df = pd.DataFrame(te_ary)
    # print(df)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print("\nDataFrame")
    print(df)
    lendf = len(df)
    shape = df.shape
    numcols = shape[1]

    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    # frequent_itemsets = apriori(df, min_support=0.3)
    print("\nConjuntos de datos frecuentes")
    print(frequent_itemsets)
    lenfreq = len(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    print("\nReglas de asociación")
    print(rules)
    lenrules = len(rules)
    shaperules = rules.shape
    numcolsrules = shaperules[1]

    return render_template('createReglasAsociacion.html', lendf=lendf, numcols=numcols, df=df, lenfreq=lenfreq, frequent_itemsets=frequent_itemsets, lenrules=lenrules, numcolsrules=numcolsrules, rules=rules, lendataset=lendataset, dataset=dataset)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 