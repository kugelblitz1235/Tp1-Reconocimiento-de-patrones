import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import statsmodels.formula.api as smf

def corrHeatmap(df):
    correlation = df.corr()

    paleta = sns.diverging_palette(150, 275, s=80, l=40, n=20)

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(17, 12))
        ax = sns.heatmap(correlation,
                         annot=True,
                         annot_kws={'size': 12},
                         fmt='.2f',
                         vmax=1,
                         vmin=-1,
                         square=True,
                         linewidths=.01,
                         linecolor='lightgray',
                         cmap=paleta)
    return correlation



def getModelStatistics(predictions, df, target, modelo):
    y = df[target].values
    y = y.reshape(len(y),1)
    r2 = r2_score(y, predictions)
    model_res = smf.ols(modelo, df).fit()
    AIC = model_res.aic

    return r2, AIC

def getModel(features, target):
    modelo = str(target[0])
    modelo += " ~ %s" % (str(features[0]))
    for feature in features[1:]:
        modelo += " + %s" % (str(feature))

    return modelo

def mostCorrelated(df, prediction):
    correlation = df.corr()[prediction]
    correlation = abs(correlation).sort_values(by=prediction, ascending=False).reset_index()

    return list(correlation.iloc[1])[0]

def removeOutliers(df, features, target):
    cols = features + target
    df_remove = df[cols]

    df_zscore = df_remove[(np.abs(stats.zscore(df_remove)) < 3).all(axis=1)]

    x_zscore = df_zscore[features].values
    y_zscore = df_zscore[target].values

    x_zscore = x_zscore.reshape(len(x_zscore),len(features))
    y_zscore = y_zscore.reshape(len(y_zscore),len(target))

    x = pd.DataFrame(data=x_zscore, columns=features)
    y = pd.DataFrame(data=y_zscore, columns=target)

    return pd.concat([x,y], axis = 1, sort = False)

#Generales
def addFeatures(df):
    # Nuevas features
    jardin = []
    terraza = []
    seguridad = []
    luminoso = []

    # Procesamiento
    for i in range(len(df)):
        descripcion = df.loc[i,"descripcion"]
        if not isinstance(descripcion, str):
            jardin.append(0)
        else:
            palabras = ["jardin","jardín","garden","parque","jard&iacute;"]
            jardin.append(int(any([word in descripcion for word in palabras])))
        if not isinstance(descripcion, str):
            terraza.append(0)
        else:
            palabras = ["terraza"]
            terraza.append(int(any([word in descripcion for word in palabras])))
        if not isinstance(descripcion, str):
            seguridad.append(0)
        else:
            palabras = ["seguridad", "vigilancia", "alarma"]
            seguridad.append(int(any([word in descripcion for word in palabras])))
        if not isinstance(descripcion, str):
            luminoso.append(0)
        else:
            palabras = ["luminoso", "luminosidad", "iluminación", "iluminacion"]
            luminoso.append(int(any([word in descripcion for word in palabras])))

    df.insert(len(list(df))-1, "jardin", jardin, True)
    df.insert(len(list(df))-1, "terraza", terraza, True)
    df.insert(len(list(df))-1, "seguridad", seguridad, True)
    df.insert(len(list(df))-1, "luminoso", luminoso, True)

    return df


#Seguridad

def addSecurityFeatures(df):
    # Agregamos información de una base de datos externa
    df_indices_seg = df[["provincia","fecha"]]
    df_indices_seg = df_indices_seg[(~df_indices_seg.isnull()).all(axis=1)]

    # Percepción de la seguridad por Provincia (Estado)
    indice_2012 = [46.5,51.6,24.4,44.6,74.6,70.8,37.9,82.7,70.8,81.8,54.1,74.4,49.2,61.5,84.5,76.0,81.2,72.9,86.7,60.7,57.1,29.4,59.9,60.0,77.4,42.4,71.2,82.9,40.1,70.0,19.2,80.5]
    indice_2013 = [45.4,46.4,69.2,40.2,16.9,28.2,45.7,18.5,26.2,19.5,37.9,9.6,39.3,23.3,8.3,16.5,12.5,42.6,19.6,31.1,32.1,55.9,27.7,19.3,21.4,47.3,15.0,13.5,45.8,22.0,67.0,14.6]
    indice_2014 = [48.2,41.5,59.0,38.7,19.9,41.7,36.0,22.2,22.1,23.6,31.8,18.9,31.5,31.0,6.4,16.3,10.6,47.1,25.6,20.6,34.0,59.3,30.4,24.9,26.5,41.6,13.2,13.4,36.6,16.9,69.6,18.5]
    indice_2015_con_limites = [(62.2,67.1),(55.6,59.7),(59.1,63.1),(60.8,65.8),(62.1,66.4),(65.2,69.0),(65.7,69.6),(69.7,73.6),(48.6,53.7),(64.6,68.5),(47.5,52.6),(56.5,61.5),(54.5,59.5),(58.1,62.4),(31.5,36.0),(62.0,65.6),(43.6,48.0),(71.0,75.8),(57.9,61.7),(59.6,64.8),(54.2,57.9),(54.2,58.6),(54.6,59.0),(60.8,65.8),(66.2,69.9),(55.7,61.4),(35.9,40.3),(57.2,61.0),(50.5,55.3),(55.0,59.0),(64.4,68.4),(59.4,64.7)]
    indice_2015 = []
    for t in indice_2015_con_limites :
        indice_2015.append((t[0]+t[1])/2)

    indice_2016_con_limites = [(58.4,63.6),(54.2,58.3),(66.5,70.1),(60.1,64.3),(68.6,72.4),(52.7,56.4),(63.5,67.2),(69.2,73.6),(40.4,45.4),(70.2,73.5),(49.5,54.6),(56.6,61.3),(56.4,61.1),(54.7,59.2),(30.0,34.1),(64.0,68.2),(43.6,48.2),(75.7,79.4),(59.2,63.2),(62.0,66.9),(46.6,50.4),(50.9,55.2),(53.7,57.9),(60.4,66.2),(69.7,73.3),(56.6,62.1),(33.7,38.0),(56.5,60.3),(54.3,58.5),(53.1,57.3),(64.4,68.0),(58.3,63.5)]
    indice_2016 = []
    for t in indice_2016_con_limites :
        indice_2016.append((t[0]+t[1])/2)

    # Estados en el orden que aparecen los datos en cada índice
    estados = ["Aguascalientes","Baja California Norte","Baja California Sur","Campeche","Coahuila","Colima","Chiapas","Chihuahua","Distrito Federal","Durango","Guanajuato","Guerrero","Hidalgo","Jalisco","Edo. de México","Michoacán","Morelos","Nayarit","Nuevo León","Oaxaca","Puebla","Querétaro","Quintana Roo","San luis Potosí","Sinaloa","Sonora","Tabasco","Tamaulipas","Tlaxcala","Veracruz","Yucatán","Zacatecas"]

    lista = []
    indice_inseguridad = []

    for index, row in df_indices_seg.iterrows():
        lista.append((row["provincia"],row["fecha"]))

    for i in range(len(lista)):
        if("2012" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_inseguridad.append(indice_2012[j])
                    break

        elif("2013" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_inseguridad.append(indice_2013[j])
                    break

        elif("2014" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_inseguridad.append(indice_2014[j])
                    break

        elif("2015" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_inseguridad.append(indice_2015[j])
                    break

        elif("2016" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_inseguridad.append(indice_2016[j])
                    break

    return indice_inseguridad



#Infracciones

def addInfractionsFeature(df):
    # CALCULAMOS CON LA TASA MEDIA DE CRECIMIENTO POBLACIONAL ENTRE 2010-2015
    df_indices_seg = df[["provincia","fecha"]]
    df_indices_seg = df_indices_seg[(~df_indices_seg.isnull()).all(axis=1)]

    # Población por provincia por año
    poblacion_total_2012 = obtener_poblacion(2)
    poblacion_total_2013 = obtener_poblacion(3)
    poblacion_total_2014 = obtener_poblacion(4)
    poblacion_total_2015 = obtener_poblacion(5)
    poblacion_total_2016 = obtener_poblacion(6)

    # Reporte de infracciones por año
    infracciones_2012 = [ 4474, 5046, 6208, 6907, 2079, 6008, 7918, 7942,1046967,  878, 5137, 3635, 38833, 2982, 33949, 11528, 4416, 16840, 8141, 8081,  835, 3190, 4403, 15979,  455,  649, 5025, 1701, 2934, 11810, 19653,  130]
    infracciones_2012 = regularizacion_de_infracciones(poblacion_total_2012, infracciones_2012)

    infracciones_2013 = [ 19985, 4618, 9423, 19957, 1494, 6011, 11319, 4797,1200784,  626,1329, 6263, 43471, 2793, 31583, 9257, 4455, 3180, 15841, 11090, 8395, 4392, 3646, 19562, 3654, 1732, 4854, 22072, 1753, 12056, 20826,  140]
    infracciones_2013 = regularizacion_de_infracciones(poblacion_total_2013, infracciones_2013)

    infracciones_2014 = [ 19617, 4232,  946, 26561, 12621, 4727, 9187, 5278, 592135, 5525, 1090, 4013, 38962, 2391, 36331, 9994, 6335,  902, 10078, 8374, 4389, 1206, 4077, 26382,  701, 2871, 4011, 25356, 2310, 14498, 22606,  175]
    infracciones_2014 = regularizacion_de_infracciones(poblacion_total_2014, infracciones_2014)

    infracciones_2015 = [ 19729, 3106, 1188, 26980, 22375, 6388, 8816, 6208, 198094, 16017, 1174, 3895, 32931, 3569, 25357, 3920, 10969, 2148, 10618, 5931, 1822, 1708, 3404, 22169,  735, 1833, 6719, 29556, 3810, 16538, 264264,  216]
    infracciones_2015 = regularizacion_de_infracciones(poblacion_total_2015, infracciones_2015)

    infracciones_2016 = [ 69886, 2652,  796, 21101, 19304, 4547, 8799, 8525, 205320, 23752, 93899, 13552, 5548, 3459, 34539, 44615, 29294,  476, 9082, 6809, 1649, 36824, 12806, 20372,  707, 1365, 10846, 34944, 3945, 17176, 260592,  224]
    infracciones_2016 = regularizacion_de_infracciones(poblacion_total_2016, infracciones_2016)

    # Estados
    estados = ["Aguascalientes","Baja California Norte","Baja California Sur","Campeche","Coahuila","Colima","Chiapas","Chihuahua","Distrito Federal","Durango","Guanajuato","Guerrero","Hidalgo","Jalisco","Edo. de México","Michoacán","Morelos","Nayarit","Nuevo León","Oaxaca","Puebla","Querétaro","Quintana Roo","San luis Potosí","Sinaloa","Sonora","Tabasco","Tamaulipas","Tlaxcala","Veracruz","Yucatán","Zacatecas"]

    lista = []
    indice_infracciones = []

    for index, row in df_indices_seg.iterrows():
        lista.append((row["provincia"],row["fecha"]))

    for i in range(len(lista)):
        if("2012" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_infracciones.append(infracciones_2012[j])
                    break

        elif("2013" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_infracciones.append(infracciones_2013[j])
                    break

        elif("2014" in lista[i][1]) :
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_infracciones.append(infracciones_2014[j])
                    break

        elif("2015" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_infracciones.append(infracciones_2015[j])
                    break

        elif("2016" in lista[i][1]):
            for j in range(len(estados)):
                if estados[j].lower() in lista[i][0].lower():
                    indice_infracciones.append(infracciones_2016[j])
                    break

    return indice_infracciones

def obtener_poblacion(n):
    poblacion_total_2010 = [1184996,3155070,637026,822441,2748391,650555,4796580,3406465,8851080,1632934,5486372,3388768,2665018,7350682,1517862,4351037,1777227,1084979,4653458,3801962,5779829,1827937,1325578,2585518,2767761,2662480,2238603,3268554,1169936,7643194,1955577,1490668]
    tasa_de_crecimiento  = [2.2,1.3,2.6,2,1.6,2,1.8,1,0.3,1.6,1.4,0.9,1.5,1.5,1.4,1.2,1.6,1.9,2.1,0.9,1.4,2.4,2.7,1.1,1.5,1.6,1.5,1.2,1.8,1.3,1.5,1.3]
    poblacion = []
    for i in range(len(poblacion_total_2010)):
        poblacion.append(poblacion_total_2010[i] * tasa_de_crecimiento[i]*n)

    return poblacion

# Datos de infracciones registradas por intervenciones de la policía por Provincia (Estado)
def regularizacion_de_infracciones(poblacion_total, infracciones):
    rv = []
    for i in range(len(infracciones)) :
        #el dato que ingresamos es la cantidad de infracciones cada 1000 habitantes
        rv.append(infracciones[i] * 1000 / poblacion_total[i])
    return rv

def featureEng(df):
    df = addFeatures(df)
    indice_seguridad = addSecurityFeatures(df)
    infracciones     = addInfractionsFeature(df)

    df = df.dropna(subset=["provincia"])
    df.insert(len(list(df))-1, "infracciones", infracciones, True)
    df.insert(len(list(df))-1, "indice_seguridad", indice_seguridad, True)

    return df

def segmentaciones(df):
    # Tipo de uso de la propiedad
    urbano = ['Apartamento', 'Casa en condominio', 'Casa', 'Edificio', 'Casa uso de suelo', 'Departamento Compartido']
    comercial = ['Terreno comercial', 'Local Comercial', 'Oficina comercial', 'Local en centro comercial', 'Bodega comercial']

    # Tipo de propiedad
    casa = ['Casa en condominio', 'Casa', 'Duplex']
    depto = ['Apartamento']

    # Categorías de precios
    upper = df["precio"].max()
    lower = df["precio"].min()

    tipos = [urbano, comercial, casa, depto] # Agregar provincias
    segmentos = ["urbano", "comercial", "casa", "depto"]

    dataframes = {}
    for i in range(len(tipos)):
        dataframes[segmentos[i]] = df[df["tipodepropiedad"].isin(tipos[i])]

    categoria = (df["precio"]-lower)//((upper+1-lower)/3)
    df.insert(len(list(df))-1, "categoria", categoria, True)

    categorias = ["bajos", "medios", "altos"]

    for cat in range(3):
        dataframes[categorias[cat]] = df[df["categoria"] == cat]

    return dataframes

def masFrecuente(dicc):
    counter = 0
    elem    = ""
    listaR2 = []

    for var, r2scores in dicc.items():
        actualFrequency = len(r2scores)

        if actualFrequency > counter:
            counter = actualFrequency
            elem    = var
            listaR2 = r2scores

    r2score = np.mean(listaR2)

    return elem, r2score
