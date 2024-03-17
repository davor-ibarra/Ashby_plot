# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:24:05 2024

@author: Davor Ibarra Pérez
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

st.set_page_config(page_title="Materiom_AshbyPlot", 
                   page_icon="📖", 
                   layout="wide")

if 'datos' not in st.session_state:
    st.session_state.datos = None
if 'hoja' not in st.session_state:
    st.session_state.hoja = None
if 'datos_hoja' not in st.session_state:
    st.session_state.datos_hoja = None
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 'Home'
    


def cargar_y_almacenar_datos():
    archivo = st.session_state.get("uploaded_file")
    if archivo is not None:
        datos = pd.read_excel(archivo, sheet_name=None)  # Carga todas las hojas
        st.session_state['datos'] = datos  # Guarda los nombres de las hojas

def convert_df_to_excel(df):
    """
    Convierte un DataFrame en un objeto Excel en memoria, que luego puede ser descargado.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)  # Regresa al inicio del stream
    return output.getvalue()

def data_processing():
    st.header('Data Cleaning')
    if 'datos' not in st.session_state or st.session_state['datos'] is None:
        st.warning('Please, upload data file.')
        return
    
    hojas_nombres = list(st.session_state['datos'].keys())
    hoja_seleccionada = st.selectbox('Selection excel sheet:', hojas_nombres, key='hoja_seleccionada')

    if 'hoja_actual' not in st.session_state or st.session_state['hoja_actual'] != hoja_seleccionada:
        # Carga una nueva hoja o recarga la hoja si se ha seleccionado una diferente
        st.session_state['hoja_actual'] = hoja_seleccionada
        st.session_state['datos_hoja'] = st.session_state['datos'][hoja_seleccionada].copy()

    datos_hoja = st.session_state['datos_hoja']

    # Cambio de nombres de columna
    col_seleccionada = st.selectbox('Select column for rename:', datos_hoja.columns)
    nuevo_nombre = st.text_input('New name for column:', value=col_seleccionada)
    if st.button('Rename column'):
        datos_hoja.rename(columns={col_seleccionada: nuevo_nombre}, inplace=True)
        st.session_state[hoja_seleccionada] = datos_hoja
        st.success(f'Column name "{col_seleccionada}" is modify to "{nuevo_nombre}"')
    
    # Sección para modificar valores específicos
    st.subheader('Modify values')
    col_a_modificar = st.selectbox('Select the column name for value modify:', datos_hoja.columns, key='col_mod')
    indice_a_modificar = st.number_input('Select index of row for value modify:', min_value=0, max_value=len(datos_hoja)-1, value=0, step=1, key='index_mod')

    valor_actual = datos_hoja.iloc[indice_a_modificar][col_a_modificar]
    st.write(f'Actual Value: {valor_actual}')

    # Determina el tipo de entrada basado en el tipo de datos de la columna
    if pd.api.types.is_numeric_dtype(datos_hoja[col_a_modificar]):
        nuevo_valor = st.number_input('New value:', value=float(valor_actual) if pd.notna(valor_actual) else 0.0, key='new_val')
    else:
        nuevo_valor = st.text_input('New value:', value=str(valor_actual) if pd.notna(valor_actual) else '', key='new_val_text')
    
    if st.button('Update Value'):
        datos_hoja.at[indice_a_modificar, col_a_modificar] = nuevo_valor
        st.session_state[hoja_seleccionada] = datos_hoja
        st.success(f'Update value to {nuevo_valor} int the row {indice_a_modificar}, and column "{col_a_modificar}"')

    st.write(datos_hoja)  # Muestra los datos para referencia del usuario
    
    # Botón de descarga
    if st.button('Save Change'):
        df_xlsx = convert_df_to_excel(datos_hoja)
        st.download_button(label='Download excel modify',
                           data=df_xlsx,
                           file_name='data_modify.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


def promediar_propiedades(datos, propiedades):
    """
    Promedia los valores mínimos y máximos de las propiedades seleccionadas.
    """
    for propiedad in propiedades:
        min_col, max_col = propiedad + "_min", propiedad + "_max"
        datos[propiedad + "_mean"] = datos[[min_col, max_col]].mean(axis=1)
    return datos

def generar_nombres_columnas(propiedades):
    columnas = []
    for propiedad in propiedades:
        columnas.append(propiedad + "_min")
        columnas.append(propiedad + "_max")
    return columnas


def interpolate_periodic_spline(points, smoothness=100, num_nodes=None):
    """ Interpola un spline periódico suave a través de puntos dados. """
    # Asegurarse de que el primer punto no se repita al final
    if np.all(points[0] == points[-1]):
        points = points[:-1]

    # Agregar el primer punto al final para cerrar el ciclo
    points = np.vstack([points, points[0]])

    # Calcular el arreglo de parámetros 't' para la interpolación
    num_points = len(points)
    t = np.linspace(0, num_points, num_points, endpoint=False)

    # Crear una función interpoladora cíclica para x e y
    cs_x = CubicSpline(t, points[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, points[:, 1], bc_type='periodic')

    # Generar puntos suavizados
    smooth_t = np.linspace(0, num_points, num=smoothness * num_points, endpoint=False)
    smooth_x = cs_x(smooth_t)
    smooth_y = cs_y(smooth_t)

    return smooth_x, smooth_y

def generar_plot_ashby(datos, materiales, propiedades, modo_visualizacion, tipo_conexion='Spline'):
    colores = dict(zip(materiales, plt.cm.tab10.colors[:len(materiales)]))  # Asignar colores
    
    if modo_visualizacion == 'Mean':
        
        datos = promediar_propiedades(datos, propiedades)
        
        x_min = datos[propiedades[0] + '_mean'].min()
        x_max = datos[propiedades[0] + '_mean'].max()
        y_min = datos[propiedades[1] + '_mean'].min()
        y_max = datos[propiedades[1] + '_mean'].max()
        
        # Mostrar los datos filtrados
        st.write(datos)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Genera puntos para cada material
        for material in materiales:
            datos_material = datos[datos['Abbreviations'] == material]
            datos_material.dropna(subset=[propiedades[0] + '_mean',
                                          propiedades[1] + '_mean'], inplace=True)
            
            if not datos_material.empty:
                x = datos_material[propiedades[0] + '_mean'].values
                y = datos_material[propiedades[1] + '_mean'].values if len(propiedades) > 1 else np.zeros_like(x)
                
                puntos = np.column_stack((x, y))
                
                if tipo_conexion == 'Lineal' and len(x) > 1:
                    
                    # Calcular la envolvente convexa (convex hull) de los puntos
                    if len(puntos) > 2:  # ConvexHull necesita al menos 3 puntos para calcular el hull
                        hull = ConvexHull(puntos)
                        for simplex in hull.simplices:
                            ax.fill(puntos[hull.vertices, 0], puntos[hull.vertices, 1], color=colores[material], alpha=0.05, edgecolor='white') #,hatch = '/'
                    else:
                        ax.scatter(x, y, label=material, color=colores[material])
                    # Dibujar puntos
                    ax.scatter(x, y, label=material, color=colores[material])
                
                elif tipo_conexion == 'Spline':
                    # Calcular la envolvente convexa (convex hull)
                    if len(puntos) > 2:  
                        hull = ConvexHull(puntos)
                        hull_points = puntos[hull.vertices]

                        # Interpola una curva suave que cierra el ciclo de puntos
                        smooth_x, smooth_y = interpolate_periodic_spline(hull_points, smoothness=200)
            
                        # Rellenar el área dentro de la curva suavizada
                        ax.fill(smooth_x, smooth_y, color=colores[material], alpha=0.1, edgecolor='white')
                    else:
                        ax.scatter(x, y, label=material, color=colores[material])
                    # Dibujar puntos
                    ax.scatter(x, y, label=material, color=colores[material])
            
        ax.set_xlim(0, x_max * 1.2)
        ax.set_ylim(0 * 0.8, y_max * 1.2)
        ax.set_xlabel(propiedades[0])
        ax.set_ylabel(propiedades[1] if len(propiedades) > 1 else 'Value')
        ax.set_title('Ashby diagram by materials')
        ax.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)
    
    if modo_visualizacion == 'Individual':
        # Mostrar los datos filtrados
        st.write(datos)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Encontrar límites para los ejes
        x_min_global, x_max_global = float('inf'), float('-inf')
        y_min_global, y_max_global = float('inf'), float('-inf')
    
        for _, row in datos.iterrows():
            x_min, x_max = row[propiedades[0] + '_min'], row[propiedades[0] + '_max']
            y_min, y_max = row[propiedades[1] + '_min'], row[propiedades[1] + '_max'] if len(propiedades) > 1 else (0, 0)
            center_x, center_y = (x_max + x_min) / 2, (y_max + y_min) / 2
            width, height = (x_max - x_min) * 1.1, (y_max - y_min) * 1.1
    
            # Actualizar los límites globales
            x_min_global, x_max_global = min(x_min_global, x_min), max(x_max_global, x_max)
            y_min_global, y_max_global = min(y_min_global, y_min), max(y_max_global, y_max)
    
            ellipse = patches.Ellipse((center_x, center_y), width, height, color=colores[row['Abbreviations']], alpha=0.4, label=row['Abbreviations'])
            ax.add_patch(ellipse)
    
        ax.set_xlim(x_min_global * 0.8, x_max_global * 1.2)
        ax.set_ylim(y_min_global * 0.8, y_max_global * 1.2)
        ax.set_xlabel(propiedades[0])
        ax.set_ylabel(propiedades[1] if len(propiedades) > 1 else 'Value')
        ax.set_title('Ashby diagram by sources')
        ax.legend(fontsize='small', ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)
    

def ashby_plot():
    st.header('Ashby Plot')

    if 'datos_hoja' not in st.session_state or st.session_state['datos_hoja'] is None:
        st.warning('Please, select sheet of data in data processing.')
        return

    datos_hoja = st.session_state['datos_hoja']
    
    st.write()
    
    c1, c2 = st.columns(2)
    
    with c1:
        modo_visualizacion = st.radio(
            "Visualization mode:",
            ('Mean', 'Individual'),
            help="Select 'Individual' for view by each source or row. Select 'Mean' for view the mean of min y max by each material."
        )
    with c2:
        tipo_conexion = 'Spline'
        if modo_visualizacion == "Mean":
            tipo_conexion = st.selectbox("Type of conection between points:",
                                         ['Spline', 'Lineal'],
                                         help="Select type of conection between points."
                                         )

    # Permite al usuario seleccionar los materiales y propiedades
    materiales_seleccionados = st.multiselect('Select materials:', options=datos_hoja['Abbreviations'].unique())
    propiedades_seleccionadas = st.multiselect('Select properties:', options=['UTS', 'YM', 'EB'])

    if not materiales_seleccionados or not propiedades_seleccionadas:
        st.warning('Please, select almost one material and two properties.')
        return

    # Generar los nombres de las columnas para filtrar
    columnas_filtradas = generar_nombres_columnas(propiedades_seleccionadas)
    columnas_filtradas.append('Abbreviations')

    # Filtrar los datos
    datos_filtrados = datos_hoja[columnas_filtradas]
    datos_filtrados = datos_filtrados[datos_filtrados['Abbreviations'].isin(materiales_seleccionados)]

    # Genera el plot de Ashby
    generar_plot_ashby(datos_filtrados, materiales_seleccionados, propiedades_seleccionadas, modo_visualizacion, tipo_conexion)
    
    # Inicializa el contador si no existe en el estado de sesión
    if 'contador_guardado' not in st.session_state:
        st.session_state.contador_guardado = 0
    
    # Botón para guardar el plot generado
    if st.button('Save Graph'):
        st.session_state.contador_guardado += 1  # Incrementa el contador
        buf = BytesIO()
        plt.savefig(buf, format="png")
        st.download_button(
            label="Download graph in PNG extension",
            data=buf,
            file_name=f"Materiom_AshbyPlot_{st.session_state.contador_guardado}.png",
            mime="image/png"
        )

def main():
    c1, c2, c3 = st.columns([1,2,1])
    
    with c2:
        st.title('Materiom Ashby Plots',)
    
    app_page = option_menu(menu_title=None, 
                           options=["Upload","Data Processing", "Ashby Plot"], 
                           icons=['cloud-upload','table','zoom-in'], 
                           menu_icon='cast', 
                           default_index=0, 
                           orientation='horizontal')
    
    if app_page == "Upload":
        st.header('Welcome to Materiom Ashby Plots app')
        archivo = st.file_uploader('Upload Excel file with properties of materials', type=['xlsx'], key='uploaded_file')
        if archivo:
            cargar_y_almacenar_datos()
            
    elif app_page == "Data Processing":
        data_processing()
    
    elif app_page == "Ashby Plot":
        ashby_plot()
            

if __name__ == '__main__':
    main()