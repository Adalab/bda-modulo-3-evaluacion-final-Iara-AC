# importamos las librerías que necesitamos

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import re

# Visualización
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluar linealidad de las relaciones entre las variables
# y la distribución de las variables
# ------------------------------------------------------------------------------
import scipy.stats as stats
import scipy.stats as stats
from scipy.stats import shapiro, poisson, chisquare, expon, kstest


def columnas_minusculas(df):
    """
    Convierte los nombres de las columnas de un DataFrame a minúsculas.

    Args:
        df (pd.DataFrame): DataFrame cuyos nombres de columnas se transformarán.

    Returns:
        pd.DataFrame: DataFrame con los nombres de columnas en minúsculas.
    """
    df.columns = df.columns.str.lower()
    return df


def transformar_negativos_en_positivos(df, columna):
    """
    Transforma los valores negativos en positivos en una columna específica de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columna (str): Nombre de la columna a transformar.

    Returns:
        pd.DataFrame: DataFrame con la columna actualizada.
    """
    df[columna] = df[columna].apply(abs)
    return df

def transformar_ano_en_fecha(df, columna):
    """
    Convierte una columna de años (float) en un DataFrame a tipo de dato fecha
    y devuelve solo el año de esa columna.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columna (str): Nombre de la columna que contiene los años.

    Returns:
        pd.DataFrame: DataFrame con la columna transformada a fecha, con solo el año.
    """
    try:
        # Reemplazar NaN por un valor específico si es necesario (por ejemplo: 1900)
        df[columna] = df[columna].fillna(1900).astype(int)  # Reemplazar NaN y convertir a entero
        
        # Convertir la columna de años (ahora enteros) a fechas, utilizando el 1 de enero como fecha base
        df[columna] = pd.to_datetime(df[columna], format='%Y', errors='coerce')
        
        # Extraer solo el año
        df[columna] = df[columna].dt.year
    except Exception as e:
        print(f"Error al convertir la columna {columna} a fecha: {e}")
    
    return df


def transformar_mes(columna):
    """
    Convierte una columna de meses en formato float a enteros,
    manteniendo los valores NaN intactos.

    Args:
        columna (pd.Series): Columna que contiene los meses en formato float.

    Returns:
        pd.Series: Columna transformada con valores de meses como enteros o NaN.
    """
    # Redondear o convertir a enteros los valores no NaN
    return columna.apply(lambda x: int(x) if not pd.isna(x) else np.nan)


def float_a_int(dato):
    """
 Convierte un número float a entero, ignorando valores NaN.

    Args:
        dato (float): Número en formato float.

    Returns:
        int o NaN: Número convertido a entero, o NaN si el valor es NaN.
    """
    if pd.isna(dato):  # Si el dato es NaN, devuelve NaN
        return np.nan
    return int(dato)  # Convierte a entero si no es NaN


def normalidad(dataframe, columna):
    """
    Evalúa la normalidad de una columna de datos de un DataFrame utilizando la prueba de Shapiro-Wilk.

    Parámetros:
        dataframe (DataFrame): El DataFrame que contiene los datos.
        columna (str): El nombre de la columna en el DataFrame que se va a evaluar para la normalidad.

    Returns:
        None: Imprime un mensaje indicando si los datos siguen o no una distribución normal.
    """

    statistic, p_value = stats.shapiro(dataframe[columna])
    if p_value > 0.05:
        print(f"Para la columna {columna} los datos siguen una distribución normal.")
    else:
        print(f"Para la columna {columna} los datos no siguen una distribución normal.")
        


def test_man_whitney(dataframe, columnas_metricas, grupo_control, grupo_test, columna_grupos):

    """
    Realiza la prueba de Mann-Whitney U para comparar las medianas de las métricas entre dos grupos en un DataFrame dado.

    Parámetros:
    - dataframe (DataFrame): El DataFrame que contiene los datos.
    - columnas_metricas (list): Una lista de nombres de columnas que representan las métricas a comparar entre los grupos.
    - grupo_control (str): El nombre del grupo de control en la columna especificada por columna_grupos.
    - grupo_test (str): El nombre del grupo de test en la columna especificada por columna_grupos.
    - columna_grupos (str): El nombre de la columna que contiene la información de los grupos. Por defecto, "campaign_name".

    Returns 
    No devuelve nada directamente, pero imprime en la consola si las medianas son diferentes o iguales para cada métrica.
    Se utiliza la prueba de Mann-Whitney U para evaluar si hay diferencias significativas entre los grupos.
    """
    # filtramos el DataFrame para quedarnos solo con los datos de control
    control = dataframe[dataframe[columna_grupos] == grupo_control]
    
    # filtramos el DataFrame para quedarnos solo con los datos de control
    test = dataframe[dataframe[columna_grupos] == grupo_test]
    
    
    # iteramos por las columnas de las metricas para ver si para cada una de ellas hay diferencias entre los grupos
    for metrica in columnas_metricas:
        
        # filtrams el conjunto de datos para quedarnos solo con la columna de la metrica que nos interesa
        metrica_control = control[metrica]
        metrica_test = test[metrica]
        
        # aplicamos el estadístico
        u_statistic, p_value = stats.mannwhitneyu(metrica_control, metrica_test)
        
        if p_value < 0.05:
            print(f"Para la métrica {metrica}, las medianas son diferentes.")
        else:
            print(f"Para la métrica {metrica}, las medianas son iguales.")