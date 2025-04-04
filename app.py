# =============================================================================
# Importação de Bibliotecas 
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import geobr
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium
from folium.plugins import Draw, HeatMap
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import time
import json
import os
import logging
import yaml
from scipy.stats import gaussian_kde
import concurrent.futures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import Dict, List, Tuple, Optional, Union, Any
import rtree  # Para indexação espacial
from functools import wraps
from contextlib import nullcontext
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from datetime import datetime
import pickle
import re
from sklearn.model_selection import train_test_split

# =============================================================================
# Configuração de Logging
# =============================================================================

# Configuração do sistema de logging
def configurar_logging():
    """
    Configura o sistema de logging da aplicação com rotação de arquivos.
    """
    # Certifique-se de que o diretório de logs existe
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configuração do logger
    logger = logging.getLogger('taipa')
    logger.setLevel(logging.INFO)
    
    # Limpa handlers existentes para evitar duplicação
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Handler para arquivo COM ROTAÇÃO
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'logs/taipa.log',
        maxBytes=5 * 1024 * 1024,  # 5 MB por arquivo
        backupCount=3,             # Mantém até 3 arquivos de backup
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Adiciona os handlers ao logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Inicializa o logger
logger = configurar_logging()


# =============================================================================
# Carregamento de Configuração
# =============================================================================

def carregar_configuracao():
    """
    Carrega a configuração de um arquivo YAML.
    
    Tenta carregar a configuração de 'config.yaml'. Se o arquivo não existir,
    cria uma configuração padrão e a salva no arquivo.
    
    Returns:
        dict: Dicionário com as configurações da aplicação.
    """
    try:
        # Verifica se o arquivo de configuração existe
        if not os.path.exists('config.yaml'):
            # Configuração padrão
            config = {
                'api': {
                    'gbif': {
                        'base_url': 'https://api.gbif.org/v1/occurrence/search',
                        'limite_padrao': 100,
                        'limite_maximo': 10000
                    }
                },
                'pseudoausencias': {
                    'num_pontos_padrao': 100,
                    'minimo': 50,
                    'maximo': 500,
                    'distancia_buffer_padrao': 50,
                    'raio_exclusao_padrao': 1.0,
                    'max_tentativas_por_ponto': 1000
                },
                'vis_params': {
                    1:  {'min': 10,  'max': 30,   'palette': ['blue', 'cyan', 'green', 'yellow', 'red']},
                    2:  {'min': 0,   'max': 20,   'palette': ['white', 'blue']},
                    3:  {'min': 0,   'max': 100,  'palette': ['yellow', 'orange', 'red']},
                    4:  {'min': 0,   'max': 500,  'palette': ['white', 'blue']},
                    5:  {'min': 10,  'max': 50,   'palette': ['white', 'red']},
                    6:  {'min': 0,   'max': 30,   'palette': ['white', 'blue']},
                    7:  {'min': 0,   'max': 50,   'palette': ['white', 'green']},
                    8:  {'min': 5,   'max': 40,   'palette': ['yellow', 'red']},
                    9:  {'min': 0,   'max': 30,   'palette': ['white', 'yellow', 'blue']},
                    10: {'min': 10,  'max': 30,   'palette': ['white', 'red']},
                    11: {'min': 0,   'max': 30,   'palette': ['blue', 'green', 'yellow', 'red']},
                    12: {'min': 200, 'max': 3000, 'palette': ['red', 'yellow', 'blue']},
                    13: {'min': 0,   'max': 300,  'palette': ['red', 'yellow', 'blue']},
                    14: {'min': 0,   'max': 200,  'palette': ['red', 'yellow', 'blue']},
                    15: {'min': 0,   'max': 100,  'palette': ['white', 'orange', 'red']},
                    16: {'min': 0,   'max': 1000, 'palette': ['white', 'lightblue', 'blue']},
                    17: {'min': 0,   'max': 300,  'palette': ['red', 'yellow','blue']},
                    18: {'min': 0,   'max': 1000, 'palette': ['white', 'lightblue', 'blue']},
                    19: {'min': 0,   'max': 800,  'palette': ['white', 'lightblue', 'blue']}
                },
                'descricoes_bio': {
                    1: "Temperatura Média Anual",
                    2: "Amplitude Diurna Média (Média mensal de (temp. máxima - temp. mínima))",
                    3: "Isotermalidade (BIO2/BIO7) (×100)",
                    4: "Sazonalidade da Temperatura (desvio padrão ×100)",
                    5: "Temperatura Máxima do Mês Mais Quente",
                    6: "Temperatura Mínima do Mês Mais Frio",
                    7: "Amplitude Anual da Temperatura (BIO5 - BIO6)",
                    8: "Temperatura Média do Trimestre Mais Úmido",
                    9: "Temperatura Média do Trimestre Mais Seco",
                    10: "Temperatura Média do Trimestre Mais Quente",
                    11: "Temperatura Média do Trimestre Mais Frio",
                    12: "Precipitação Anual",
                    13: "Precipitação do Mês Mais Chuvoso",
                    14: "Precipitação do Mês Mais Seco",
                    15: "Sazonalidade da Precipitação (Coeficiente de Variação)",
                    16: "Precipitação do Trimestre Mais Chuvoso",
                    17: "Precipitação do Trimestre Mais Seco",
                    18: "Precipitação do Trimestre Mais Quente",
                    19: "Precipitação do Trimestre Mais Frio"
                },
                'vif': {
                    'limiar': 10
                }
            }
            
            # Salva a configuração padrão
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info("Arquivo de configuração criado com valores padrão.")
            
        # Carrega a configuração do arquivo
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info("Configuração carregada com sucesso.")
        return config
    
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {str(e)}")
        st.error(f"Erro ao carregar configuração: {str(e)}")
        # Retorna configuração padrão mínima em caso de erro
        return {'vis_params': vis_params_dict, 'descricoes_bio': bio_descriptions_pt}

# Carrega as configurações
CONFIG = carregar_configuracao()

# Extrai dicionários de configuração para uso direto
vis_params_dict = CONFIG['vis_params']
bio_descriptions_pt = CONFIG['descricoes_bio']


# =============================================================================
# Função decoradora para tratamento de exceções
# =============================================================================

def tratar_excecao(func):
    """
    Decorador que captura exceções em funções e registra no log.
    
    Args:
        func: A função a ser decorada.
        
    Returns:
        A função decorada que captura e trata exceções.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Erro em {func.__name__}: {str(e)}", exc_info=True)
            st.error(f"Ocorreu um erro: {str(e)}")
            return None
    return wrapper


# =============================================================================
# Funções Auxiliares (Utilitários)
# =============================================================================

# Modifique o decorador para compartilhar cache entre usuários e definir um tempo de vida
@st.cache_data(ttl="1d", show_spinner=False)
@tratar_excecao
def carregar_poligono_brasil():
    """
    Carrega o polígono do Brasil usando a biblioteca geobr.
    
    Returns:
        shapely.geometry.Polygon: Polígono representando o território brasileiro.
    """
    with st.spinner("Carregando o mapa do Brasil..."):
        try:
            logger.info("Carregando polígono do Brasil...")
            brazil_gdf = geobr.read_country(year=2019)
            if brazil_gdf.empty:
                raise ValueError("O geobr retornou um DataFrame vazio.")
            
            polygon = brazil_gdf.geometry.unary_union
            logger.info("Polígono do Brasil carregado com sucesso.")
            return polygon
        except Exception as e:
            logger.error(f"Erro ao carregar polígono do Brasil: {str(e)}", exc_info=True)
            st.error(f"Não foi possível carregar o mapa do Brasil: {str(e)}")
            # Retorna um polígono nulo em caso de erro
            return None

poligono_brasil = carregar_poligono_brasil()

@tratar_excecao
def carregar_var_bioclim(numero_var):
    """
    Carrega uma variável bioclimática do arquivo local.
    
    Args:
        numero_var (int): Número da variável bioclimática (1-19).
        
    Returns:
        rasterio.DatasetReader: Dataset raster com a variável bioclimática.
    """
    try:
        arquivo = f"bioclim_brasil/bio_{numero_var}_brasil.tif"
        if not os.path.exists(arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")
        return rasterio.open(arquivo)
    except Exception as e:
        logger.error(f"Erro ao carregar variável bioclimática {numero_var}: {str(e)}")
        raise ValueError(f"Não foi possível carregar a variável BIO{numero_var}")

@st.cache_data(ttl="1d", show_spinner=False)
def extrair_valores_em_cache(_raster, df, numero_var):
    """
    Extrai valores de um raster para pontos em um DataFrame.
    
    Args:
        _raster (rasterio.DatasetReader): Dataset raster.
        df (pd.DataFrame): DataFrame com colunas 'latitude' e 'longitude'.
        numero_var (int): Número da variável para fins de log.
        
    Returns:
        list: Lista com os valores extraídos.
    """
    try:
        # Cria uma lista de coordenadas
        coords = [(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        
        # Extrai os valores para cada ponto
        values = [x[0] for x in _raster.sample(coords)]
        
        # Remove valores None e NaN
        values = [v for v in values if v is not None and not np.isnan(v)]
        
        logger.info(f"Extraídos {len(values)} valores válidos para BIO{numero_var}")
        return values
    except Exception as e:
        logger.error(f"Erro ao extrair valores para BIO{numero_var}: {str(e)}")
        st.warning(f"Erro ao extrair valores para BIO{numero_var}: {str(e)}")
        return []

@tratar_excecao
def amostrar_valor_bio(raster, lat, lon, band_name):
    """
    Amostra um valor de um raster em um ponto específico.
    
    Args:
        raster (rasterio.DatasetReader): Dataset raster.
        lat (float): Latitude do ponto.
        lon (float): Longitude do ponto.
        band_name (str): Nome da banda no raster.
        
    Returns:
        float: Valor amostrado na posição especificada.
    """
    try:
        pt = Point(lon, lat)
        value = list(raster.sample([(lon, lat)]))[0][0]
        return value
    except Exception as e:
        logger.error(f"Erro ao amostrar valor: {str(e)}")
        return None

@tratar_excecao
def converter_shapely_para_geojson(shapely_geom):
    """
    Converte uma geometria Shapely para formato GeoJSON.
    
    Args:
        shapely_geom (shapely.geometry.BaseGeometry): Geometria Shapely.
        
    Returns:
        dict: Geometria no formato GeoJSON.
    """
    if shapely_geom is None:
        raise ValueError("Geometria Shapely não pode ser None")
    return shapely_geom.__geo_interface__

        
    geojson = shapely_geom.__geo_interface__
    
    if geojson['type'] == 'Polygon':
        return ee.Geometry.Polygon(geojson['coordinates'])
    elif geojson['type'] == 'MultiPolygon':
        return ee.Geometry.MultiPolygon(geojson['coordinates'])
    else:
        raise ValueError(f"Tipo de geometria não suportado: {geojson['type']}")

def adicionar_camada_ee(self, ee_image_object, vis_params, name):
    """
    Adiciona uma camada Earth Engine a um mapa Folium.
    
    Args:
        self: Instância do mapa Folium.
        ee_image_object (ee.Image): Imagem Earth Engine.
        vis_params (dict): Parâmetros de visualização.
        name (str): Nome da camada.
    """
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; Google Earth Engine',
            name=name,
            overlay=True,
            control=True
        ).add_to(self)
    except Exception as e:
        logger.error(f"Erro ao adicionar camada EE ao mapa: {str(e)}")
        # Não levantamos exceção aqui para não interromper o carregamento do mapa

# Adiciona o método ao mapa Folium
folium.Map.add_ee_layer = adicionar_camada_ee

def adicionar_camada_raster(self, raster_path, vis_params, name):
    """
    Adiciona uma camada raster local a um mapa Folium, recortada para o polígono do Brasil.
    
    Args:
        self: Instância do mapa Folium.
        raster_path (str): Caminho para o arquivo raster.
        vis_params (dict): Parâmetros de visualização.
        name (str): Nome da camada.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Lê os dados do raster
            data = src.read(1)
            transform = src.transform
            
            # Cria uma máscara de validade para o Brasil
            if poligono_brasil is not None:
                # Rasteriza o polígono do Brasil na mesma resolução do raster
                brasil_mask = rasterio.features.geometry_mask(
                    [poligono_brasil], 
                    out_shape=data.shape, 
                    transform=transform, 
                    invert=True
                )
                
                # Aplica a máscara (usando NaN para áreas fora do Brasil)
                masked_data = data.copy()
                masked_data[~brasil_mask] = np.nan
                data = masked_data
                
            # Normaliza os dados para o intervalo [0, 1]
            vmin = vis_params.get('min', np.nanmin(data))
            vmax = vis_params.get('max', np.nanmax(data))
            data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            
            # Cria uma imagem RGBA para controlar transparência
            # (as áreas com NaN serão completamente transparentes)
            cmap = plt.get_cmap(vis_params.get('colormap', 'viridis'))
            rgba_img = cmap(data_norm)
            # Define alpha channel (transparência)
            rgba_img[..., 3] = np.where(np.isnan(data_norm), 0, 0.7)  # 0.7 é a opacidade para pixels válidos
            
            # Cria uma imagem PNG temporária
            temp_filename = f'temp_{name.replace("/", "_").replace(":", "_")}.png'
            plt.figure(figsize=(10, 10))
            plt.imshow(rgba_img)
            plt.axis('off')
            plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            
            # Define os limites do raster para usar no mapa
            bounds = [[src.bounds.bottom, src.bounds.left], 
                      [src.bounds.top, src.bounds.right]]
            
            # Adiciona a imagem ao mapa
            folium.raster_layers.ImageOverlay(
                temp_filename,
                bounds=bounds,
                name=name,
                opacity=1.0  # Definimos como 1.0 porque já controlamos a transparência na imagem
            ).add_to(self)
            
            # Remove o arquivo temporário
            os.remove(temp_filename)
            
    except Exception as e:
        logger.error(f"Erro ao adicionar camada raster ao mapa: {str(e)}")
        st.warning(f"Erro ao adicionar {name} ao mapa: {str(e)}")
        # Não levantamos exceção aqui para não interromper o carregamento do mapa

# Adiciona o método ao mapa Folium
folium.Map.add_raster_layer = adicionar_camada_raster


# =============================================================================
# Funções de Geração de Pseudoausências e Busca de Ocorrências
# =============================================================================

@tratar_excecao
def gerar_pseudoausencias_otimizado(presence_df, n_points=100, buffer_distance=0.5, exclusion_radius=2.0):
    """
    Gera pontos de pseudoausência utilizando algoritmo espacialmente otimizado.
    
    Utiliza índice espacial rtree para verificação rápida de contenção e distribui
    pontos de forma mais eficiente dentro da área permitida.
    
    Args:
        presence_df (pd.DataFrame): DataFrame com pontos de presença (precisa ter colunas latitude/longitude).
        n_points (int): Número de pontos de pseudoausência a gerar.
        buffer_distance (float): Distância do buffer em graus.
        exclusion_radius (float): Raio em graus em torno dos pontos de presença onde não podem ser geradas pseudoausências.
        
    Returns:
        pd.DataFrame: DataFrame com pontos de pseudoausência.
    """
    try:
        logger.info(f"Gerando {n_points} pseudoausências com buffer de {buffer_distance} graus e raio de exclusão de {exclusion_radius} graus")
        
        # Converte pontos de presença para geometrias
        presence_points = [Point(lon, lat) for lon, lat in zip(presence_df["longitude"], presence_df["latitude"])]
        
        # Mostra progresso da geração de buffers
        with st.spinner("Gerando buffers ao redor dos pontos de presença..."):
            # Progressbar para visualização do processo
            progress_bar = st.progress(0)
            buffers = []
            
            # Divide em chunks para atualizar a barra de progresso
            chunk_size = max(1, len(presence_points) // 10)
            for i in range(0, len(presence_points), chunk_size):
                chunk = presence_points[i:i+chunk_size]
                buffers.extend([point.buffer(buffer_distance) for point in chunk])
                progress = min(1.0, (i + chunk_size) / len(presence_points))
                progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
        
        # Gera zonas de exclusão ao redor dos pontos de presença
        exclusion_zones = []
        if exclusion_radius > 0:
            with st.spinner("Criando zonas de exclusão..."):
                exclusion_zones = [point.buffer(exclusion_radius) for point in presence_points]
                combined_exclusion = unary_union(exclusion_zones)
                st.info(f"Criadas zonas de exclusão com raio de {exclusion_radius} graus (≈ {exclusion_radius * 111:.0f} km)")
        
        # Une todos os buffers e encontra a interseção com o Brasil
        with st.spinner("Calculando área de amostragem (buffers ∩ Brasil)..."):
            union_buffers = unary_union(buffers)
            
            if poligono_brasil is None:
                st.error("Polígono do Brasil não disponível. Usando apenas a união dos buffers.")
                allowed_region = union_buffers
            else:
                allowed_region = union_buffers.intersection(poligono_brasil)
        
        if allowed_region.is_empty:
            st.error("A área permitida (buffers ∩ Brasil) está vazia. Tente diminuir o buffer.")
            return pd.DataFrame()
        
        # Se existirem zonas de exclusão, remova-as da área permitida
        if exclusion_zones:
            combined_exclusion = unary_union(exclusion_zones)
            allowed_region = allowed_region.difference(combined_exclusion)
            if allowed_region.is_empty:
                st.error("A área permitida está vazia após aplicar zonas de exclusão. Tente reduzir o raio de exclusão.")
                return pd.DataFrame()
        
        # Exibe informações sobre a área permitida
        st.write("Limites da área permitida (buffers ∩ Brasil - zonas de exclusão):", allowed_region.bounds)
        st.write("Área permitida (valor em graus²):", allowed_region.area)
        
        # Obtém os limites da área permitida
        minx, miny, maxx, maxy = allowed_region.bounds
        
        # Cria um índice espacial para verificação rápida de contenção
        spatial_index = rtree.index.Index()
        
        # Otimização: pré-calcula células dentro da área permitida
        with st.spinner("Otimizando área de amostragem..."):
            # Divide a área em uma grade
            cell_size = min((maxx - minx), (maxy - miny)) / 20  # Tamanho adaptativo da célula
            valid_cells = []
            
            progress_bar = st.progress(0)
            rows = int((maxy - miny) / cell_size) + 1
            cols = int((maxx - minx) / cell_size) + 1
            
            # Verifica quais células estão dentro da área permitida
            for i in range(rows):
                y = miny + i * cell_size
                for j in range(cols):
                    x = minx + j * cell_size
                    cell_center = Point(x, y)
                    if allowed_region.contains(cell_center):
                        cell_id = len(valid_cells)
                        valid_cells.append((x, y, cell_size))
                        # Adiciona a célula ao índice espacial
                        spatial_index.insert(cell_id, (x, y, x + cell_size, y + cell_size))
                
                progress = min(1.0, (i + 1) / rows)
                progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
        
        if not valid_cells:
            st.error("Não foi possível encontrar células válidas na área permitida.")
            return pd.DataFrame()
        
        # Gera pseudoausências dentro das células válidas
        pseudo_points = []
        attempts = 0
        max_attempts = CONFIG['pseudoausencias']['max_tentativas_por_ponto'] * n_points
        
        with st.spinner(f"Gerando {n_points} pontos de pseudoausência..."):
            progress_bar = st.progress(0)
            
            while len(pseudo_points) < n_points and attempts < max_attempts:
                # Seleciona uma célula aleatória das células válidas
                if valid_cells:
                    cell_idx = np.random.randint(0, len(valid_cells))
                    x_base, y_base, cell_size = valid_cells[cell_idx]
                    
                    # Gera um ponto aleatório dentro da célula
                    random_lon = x_base + np.random.uniform(0, cell_size)
                    random_lat = y_base + np.random.uniform(0, cell_size)
                else:
                    # Fallback: gera dentro dos limites da área permitida
                    random_lon = np.random.uniform(minx, maxx)
                    random_lat = np.random.uniform(miny, maxy)
                
                p = Point(random_lon, random_lat)
                
                # Verifica se o ponto está dentro da área permitida
                if allowed_region.contains(p):
                    # Verifica distância mínima para pontos de presença (adicional à zona de exclusão)
                    is_valid = True
                    if exclusion_radius > 0:
                        for presence_point in presence_points:
                            if p.distance(presence_point) < exclusion_radius:
                                is_valid = False
                                break
                    
                    if is_valid:
                        pseudo_points.append({
                            "species": "pseudo-absence",
                            "latitude": random_lat,
                            "longitude": random_lon
                        })
                        
                        # Atualiza a barra de progresso
                        progress = min(1.0, len(pseudo_points) / n_points)
                        progress_bar.progress(progress)
                
                attempts += 1
                
                # Feedback a cada 1000 tentativas
                if attempts % 1000 == 0:
                    logger.info(f"Tentativa {attempts}/{max_attempts}, gerados {len(pseudo_points)}/{n_points}")
            
            progress_bar.progress(1.0)
        
        if len(pseudo_points) < n_points:
            st.warning(f"Atingido limite de tentativas. Gerados apenas {len(pseudo_points)}/{n_points} pontos.")
        else:
            st.success(f"Gerados {len(pseudo_points)} pontos de pseudoausência com sucesso.")
        
        # Cálculo das distâncias mínimas aos pontos de presença
        if pseudo_points and exclusion_radius > 0:
            distances = []
            for pp in pseudo_points:
                p = Point(pp["longitude"], pp["latitude"])
                min_dist = float('inf')
                for presence_point in presence_points:
                    d = p.distance(presence_point)
                    min_dist = min(min_dist, d)
                distances.append(min_dist)
            
            st.info(f"Distância mínima aos pontos de presença: {min(distances) * 111:.1f} km")
            st.info(f"Distância média aos pontos de presença: {(sum(distances) / len(distances)) * 111:.1f} km")
        
        logger.info(f"Gerados {len(pseudo_points)} pontos de pseudoausência em {attempts} tentativas")
        return pd.DataFrame(pseudo_points)
    
    except Exception as e:
        logger.error(f"Erro ao gerar pseudoausências: {str(e)}", exc_info=True)
        st.error(f"Erro ao gerar pseudoausências: {str(e)}")
        return pd.DataFrame()

@tratar_excecao
def buscar_ocorrencias_gbif(especie, limite=100):
    """
    Busca ocorrências de uma espécie na API do GBIF.
    
    Implementa tratamento de erros adequado e feedback mais detalhado ao usuário.
    
    Args:
        especie (str): Nome científico da espécie.
        limite (int): Número máximo de ocorrências a buscar.
        
    Returns:
        pd.DataFrame: DataFrame com as ocorrências encontradas.
    """
    try:
        # Validação básica da entrada
        if not especie or not isinstance(especie, str):
            raise ValueError("Nome da espécie deve ser uma string não vazia")
        
        especie = especie.strip()
        if not especie:
            raise ValueError("Nome da espécie não pode ser vazio")
        
        # Verifica limites configurados
        limite_maximo = CONFIG['api']['gbif']['limite_maximo']
        if limite > limite_maximo:
            st.warning(f"Limite ajustado para o máximo permitido: {limite_maximo}")
            limite = limite_maximo
        
        # Prepara a requisição
        url = CONFIG['api']['gbif']['base_url']
        params = {
            "scientificName": especie,
            "hasCoordinate": "true",
            "country": "BR",
            "limit": limite
        }
        
        logger.info(f"Buscando ocorrências para '{especie}' com limite {limite}")
        
        # Adicione um pequeno atraso antes da requisição para evitar sobrecarga da API
        time.sleep(0.5)  # Aguarda 0.5 segundos antes de fazer a requisição
        # Faz a requisição com timeout
        with st.spinner(f"Consultando a API do GBIF para {especie}..."):
            response = requests.get(url, params=params, timeout=30)
        
        # Verifica o status da resposta
        if response.status_code == 200:
            data = response.json()
            
            # Verifica se há resultados
            total_results = data.get("count", 0)
            if total_results == 0:
                st.warning(f"Nenhuma ocorrência encontrada para '{especie}' no Brasil.")
                return pd.DataFrame()
            
            # Processa os resultados
            occurrences = []
            for record in data.get("results", []):
                lat = record.get("decimalLatitude")
                lon = record.get("decimalLongitude")
                
                # Valida coordenadas
                if lat is not None and lon is not None:
                    # Verifica se está dentro do Brasil (opcional, já filtramos por país)
                    if poligono_brasil is None or Point(lon, lat).within(poligono_brasil):
                        occurrences.append({
                            "species": especie,
                            "latitude": lat,
                            "longitude": lon,
                            "id_gbif": record.get("key", ""),
                            "data_coleta": record.get("eventDate", "")
                        })
            
            # Cria o DataFrame
            df = pd.DataFrame(occurrences)
            
            # Feedback detalhado ao usuário
            if df.empty:
                st.warning(f"Encontradas {total_results} ocorrências, mas nenhuma com coordenadas válidas.")
            else:
                st.success(f"Encontradas {len(df)} ocorrências com coordenadas válidas (de {total_results} totais).")
            
            logger.info(f"Busca para '{especie}' retornou {len(df)} registros válidos de {total_results} totais")
            return df
        
        elif response.status_code == 429:
            # Limite de requisições excedido
            st.error("Limite de requisições à API do GBIF excedido. Tente novamente mais tarde.")
            logger.warning(f"Rate limit excedido na API do GBIF: {response.text}")
            return pd.DataFrame()
            
        else:
            # Outros erros
            error_msg = f"Erro {response.status_code} ao buscar dados na API do GBIF"
            try:
                error_details = response.json()
                error_msg += f": {error_details.get('message', '')}"
            except:
                error_msg += f": {response.text[:100]}"
            
            st.error(error_msg)
            logger.error(f"Erro na API do GBIF: {error_msg}")
            return pd.DataFrame()
    
    except requests.exceptions.Timeout:
        st.error("Tempo limite excedido ao conectar com a API do GBIF. Tente novamente mais tarde.")
        logger.error("Timeout na requisição à API do GBIF")
        return pd.DataFrame()
        
    except requests.exceptions.ConnectionError:
        st.error("Erro de conexão com a API do GBIF. Verifique sua conexão com a internet.")
        logger.error("Erro de conexão na requisição à API do GBIF")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Erro inesperado ao buscar dados: {str(e)}")
        logger.error(f"Erro ao buscar ocorrências: {str(e)}", exc_info=True)
        return pd.DataFrame()


# =============================================================================
# Funções de Visualização e Extração de Dados
# =============================================================================

@tratar_excecao
def criar_mapa(df, heatmap=False):
    """
    Cria um mapa Folium com pontos de ocorrência e opcionalmente um heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame com pontos (precisa ter colunas latitude/longitude).
        heatmap (bool): Se deve incluir uma camada de heatmap.
        
    Returns:
        folium.Map: Objeto de mapa Folium.
    """
    try:
        # Verifica se o DataFrame tem dados
        if df.empty:
            st.error("Não há dados para exibir no mapa.")
            # Retorna um mapa do Brasil vazio
            m = folium.Map(location=[-15.77, -47.92], zoom_start=4)
            return m
        
        # Calcula os limites e o centro do mapa com base nos dados
        min_lat = df["latitude"].min()
        max_lat = df["latitude"].max()
        min_lon = df["longitude"].min()
        max_lon = df["longitude"].max()
        
        # Adiciona uma margem aos limites (10%)
        lat_margin = (max_lat - min_lat) * 0.1
        lon_margin = (max_lon - min_lon) * 0.1
        min_lat -= lat_margin
        max_lat += lat_margin
        min_lon -= lon_margin
        max_lon += lon_margin
        
        # Define o centro do mapa
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    
        # Cria o mapa base
        m = folium.Map(location=center)
        
        # Ajusta os limites
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
        # Adiciona os pontos em um FeatureGroup para que possam ser gerenciados pelo LayerControl
        points_fg = folium.FeatureGroup(name="Ocorrências", show=True)
        
        # Adiciona cada ponto ao mapa
        for _, row in df.iterrows():
            popup_content = f"<b>Espécie:</b> {row['species']}<br><b>Lat:</b> {row['latitude']:.4f}<br><b>Lon:</b> {row['longitude']:.4f}"
            
            # Adiciona informações adicionais se disponíveis
            if 'id_gbif' in row and row['id_gbif']:
                popup_content += f"<br><b>ID GBIF:</b> {row['id_gbif']}"
            if 'data_coleta' in row and row['data_coleta']:
                popup_content += f"<br><b>Data:</b> {row['data_coleta']}"
                
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="black",
                fill=True,
                fill_color="black",
                fill_opacity=1,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(points_fg)
        
        points_fg.add_to(m)
    
        # Adiciona o plugin Draw para permitir que o usuário desenhe polígonos
        draw = Draw(
            export=False,
            draw_options={
                "polygon": True, "polyline": False, "rectangle": False,
                "circle": False, "marker": False, "circlemarker": False
            },
            edit_options={"edit": True}
        )
        draw.add_to(m)
    
        # Se a flag do heatmap estiver ativa, cria um FeatureGroup para o heatmap
        if heatmap:
            heatmap_fg = folium.FeatureGroup(name="Heatmap", show=True)
            pontos = df[["latitude", "longitude"]].values.tolist()
            # Parâmetros ajustados para manchas maiores
            HeatMap(pontos, radius=40, blur=15).add_to(heatmap_fg)
            heatmap_fg.add_to(m)
    
        # Adiciona o controle de layers para permitir a seleção das camadas
        folium.LayerControl().add_to(m)
        return m
        
    except Exception as e:
        logger.error(f"Erro ao criar mapa: {str(e)}", exc_info=True)
        st.error(f"Não foi possível criar o mapa: {str(e)}")
        # Retorna um mapa básico em caso de erro
        return folium.Map(location=[-15.77, -47.92], zoom_start=4)

@st.cache_data(show_spinner=True)
@tratar_excecao
def obter_estatisticas_bioclim(df_occ, df_pseudo):
    """
    Extrai os valores das 19 variáveis bioclimáticas e calcula estatísticas descritivas.
    
    Args:
        df_occ (pd.DataFrame): DataFrame com pontos de presença.
        df_pseudo (pd.DataFrame): DataFrame com pontos de pseudoausência.
        
    Returns:
        tuple: (stats_data, raw_data) contendo estatísticas e valores brutos.
    """
    try:
        stats_data = {}
        raw_data = {}
        stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        
        # Função auxiliar para extrair uma variável bioclimática
        def extrair_bio(i):
            try:
                bio_name = f"BIO{i}"
                bio_raster = carregar_var_bioclim(i)
                
                # Extrai os valores para os pontos de presença e pseudoausência
                presence_values = extrair_valores_em_cache(bio_raster, df_occ, i)
                pseudo_values = extrair_valores_em_cache(bio_raster, df_pseudo, i)
                combined_values = presence_values + pseudo_values
                
                # Calcula as estatísticas descritivas, ou preenche com NaN se não houver valores
                if combined_values:
                    combined_stats = pd.Series(combined_values).describe()
                else:
                    combined_stats = pd.Series({stat: np.nan for stat in stat_names})
                
                return i, combined_stats, combined_values
            except Exception as e:
                logger.error(f"Erro ao processar BIO{i}: {str(e)}")
                return i, pd.Series({stat: np.nan for stat in stat_names}), []
        
        # Usa executor para paralelismo
        with st.spinner("Extraindo valores das variáveis bioclimáticas..."):
            progresso = st.progress(0)
            
            # Configura o número de workers (ajuste conforme necessário)
            num_workers = min(4, os.cpu_count() or 4)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(extrair_bio, i): i for i in range(1, 20)}
                
                completados = 0
                total = len(futures)
                
                for future in concurrent.futures.as_completed(futures):
                    i, stats, values = future.result()
                    stats_data[f"BIO{i}"] = stats
                    raw_data[f"BIO{i}"] = values
                    
                    completados += 1
                    progresso.progress(completados / total)
        
        st.success(f"Extração de {total} variáveis bioclimáticas concluída!")
        return stats_data, raw_data
    
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas bioclimáticas: {str(e)}", exc_info=True)
        st.error(f"Erro ao processar variáveis bioclimáticas: {str(e)}")
        return {}, {}

@tratar_excecao
def plotar_densidade(values, label, color):
    """
    Plota a densidade de kernel para um conjunto de valores.
    
    Args:
        values (list): Lista de valores numéricos.
        label (str): Rótulo para a legenda.
        color (str): Cor da linha.
        
    Returns:
        tuple: (xs, density) contendo os valores x e y da curva.
    """
    if not values or len(values) < 5:  # Precisa de pelo menos alguns pontos para KDE
        logger.warning(f"Dados insuficientes para plotar densidade: {len(values) if values else 0} pontos")
        return None, None
    
    try:
        kde = gaussian_kde(values)
        xs = np.linspace(min(values), max(values), 200)
        density = kde(xs)
        plt.plot(xs, density, label=label, color=color)
        return xs, density
    except Exception as e:
        logger.error(f"Erro ao plotar densidade: {str(e)}")
        return None, None

@tratar_excecao
def amostrar_valor_bio(raster, lat, lon, band_name):
    """
    Amostra um valor de um raster em um ponto específico.
    
    Args:
        raster (rasterio.DatasetReader): Dataset raster.
        lat (float): Latitude do ponto.
        lon (float): Longitude do ponto.
        band_name (str): Nome da banda no raster.
        
    Returns:
        float: Valor amostrado na posição especificada.
    """
    try:
        pt = Point(lon, lat)
        value = list(raster.sample([(lon, lat)]))[0][0]
        return value
    except Exception as e:
        logger.error(f"Erro ao amostrar valor: {str(e)}")
        return None


# =============================================================================
# Páginas da Aplicação (Interface com o Usuário)
# =============================================================================

def pagina_inicial():
    """
    Página inicial da aplicação com descrição e instruções.
    """
    st.title("TAIPA - Tecnologia de Aprendizado Interativo em Predição Ambiental")
    
    st.write("""
    **Bem-vindo à TAIPA!**

    A TAIPA é uma plataforma interativa para o ensino de Modelagem de Distribuição de Espécies (SDM)
    utilizando o algoritmo Random Forest. Aqui, você pode explorar dados de ocorrência, gerar pseudoausências,
    visualizar variáveis ambientais e executar modelos simulados.
    """)
    
    st.info("Utilize o menu lateral para navegar pelas funcionalidades.")
    
    # Exibe detalhes técnicos em um expander
    with st.expander("Informações Técnicas"):
        st.write("""
        A TAIPA utiliza as seguintes tecnologias:
        - API do GBIF para busca de ocorrências
        - Streamlit para interface com o usuário
        - Folium para visualização de mapas interativos
        - GeoPandas/Shapely para operações espaciais
        - Rasterio para processamento de dados raster
        
        O fluxo de trabalho recomendado é:
        1. Buscar ocorrências de uma espécie via API
        2. Gerar pontos de pseudoausência
        3. Analisar variáveis ambientais
        4. Executar o modelo Random Forest
        5. Visualizar resultados e projeções
        """)
    
    # Links úteis
    st.subheader("Links úteis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[Documentação GBIF](https://www.gbif.org/developer/summary)")
        st.markdown("[Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)")
    with col2:
        st.markdown("[WorldClim](https://www.worldclim.org/data/bioclim.html)")
        st.markdown("[GeoPandas](https://geopandas.org/)")

def pagina_busca_api():
    """
    Página para busca de ocorrências via API do GBIF.
    """
    st.title("Busca de Ocorrências via API do GBIF")
    st.warning("Os dados da API do GBIF são gratuitos, mas requerem citação. Confira: https://www.gbif.org/citation-guidelines")
    
    # Expander com informações sobre o GBIF
    with st.expander("Sobre o GBIF"):
        st.write("""
        O **Global Biodiversity Information Facility (GBIF)** é uma infraestrutura internacional de dados abertos
        financiada por governos de todo o mundo, destinada a fornecer acesso a dados sobre todas as formas de vida na Terra.
        
        A API utilizada neste aplicativo consulta a base de dados do GBIF para obter registros de ocorrência de
        espécies no Brasil com coordenadas geográficas válidas.
        """)
    
    # Formulário de busca
    with st.form(key="formulario_busca"):
        especie = st.text_input("Digite o nome científico da espécie:")
        
        col1, col2 = st.columns(2)
        with col1:
            limite = st.number_input(
                "Número máximo de ocorrências:",
                min_value=10,
                max_value=CONFIG['api']['gbif']['limite_maximo'],
                value=CONFIG['api']['gbif']['limite_padrao'],
                step=10
            )
        
        with col2:
            st.write("Filtros:")
            apenas_com_coordenadas = st.checkbox("Apenas com coordenadas", value=True, disabled=True)
            apenas_brasil = st.checkbox("Apenas Brasil", value=True, disabled=True)
        
        submitted = st.form_submit_button("Buscar Ocorrências")
        
        if submitted:
            if not especie.strip():
                st.error("Por favor, digite o nome da espécie.")
            else:
                with st.spinner(f"Buscando ocorrências para {especie}..."):
                    df_api = buscar_ocorrencias_gbif(especie, limite)
                
                if df_api is not None:
                    st.session_state.df_api = df_api
                    st.session_state.especie = especie
                    # Limpa a flag do heatmap ao buscar novos dados
                    st.session_state.heatmap_generated = False

    # Se já existem dados de busca, mostra resultados
    if "df_api" in st.session_state:
        df_api = st.session_state.df_api
        
        if df_api.empty:
            st.warning("Nenhuma ocorrência encontrada com os critérios especificados.")
        else:
            st.write(f"**Total de ocorrências retornadas:** {len(df_api)}")
            
            # Tabs para diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["Dados", "Mapa", "Estatísticas"])
            
            with tab1:
                st.write("Visualização dos dados obtidos:")
                st.dataframe(df_api)
                
                # Opção para download
                csv = df_api.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV", 
                    csv, 
                    f"ocorrencias_{st.session_state.especie.replace(' ', '_')}.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with tab2:
                # Controles para o mapa
                st.subheader("Mapa de Ocorrências")
                
                # Checkbox para ativar o heatmap
                heatmap_active = st.checkbox(
                    "Mostrar Heatmap", 
                    value=st.session_state.get("heatmap_generated", False),
                    help="Exibe um mapa de calor das ocorrências"
                )
                
                # Atualiza o estado do heatmap
                st.session_state.heatmap_generated = heatmap_active
                
                # Constrói o mapa
                m = criar_mapa(df_api, heatmap=heatmap_active)
                
                # Renderiza o mapa e captura os dados interativos
                map_data = st_folium(m, width=700, height=500)
                
                # Verifica se o usuário desenhou algum polígono para remoção de pontos
                if map_data and map_data.get("all_drawings"):
                    polygon_features = [
                        feature for feature in map_data["all_drawings"]
                        if feature.get("geometry", {}).get("type") == "Polygon"
                    ]
                    
                    if polygon_features:
                        st.info(f"{len(polygon_features)} polígono(s) desenhado(s) detectado(s).")
                        
                        if st.button("Remover pontos dentro do(s) polígono(s)"):
                            total_removidos = 0
                            
                            for poly_feature in polygon_features:
                                poly_coords = poly_feature["geometry"]["coordinates"][0]
                                polygon_shapely = Polygon(poly_coords)
                                
                                indices_to_remove = []
                                for idx, row in df_api.iterrows():
                                    point = Point(row["longitude"], row["latitude"])
                                    if polygon_shapely.contains(point):
                                        indices_to_remove.append(idx)
                                
                                total_removidos += len(indices_to_remove)
                                
                                if indices_to_remove:
                                    df_api = df_api.drop(indices_to_remove).reset_index(drop=True)
                            
                            if total_removidos > 0:
                                st.session_state.df_api = df_api
                                st.success(f"{total_removidos} ponto(s) removido(s) dentro do(s) polígono(s).")
                                
                                # Atualiza o mapa
                                st.rerun()
                            else:
                                st.info("Nenhum ponto encontrado dentro do(s) polígono(s).")
            
            with tab3:
                st.subheader("Estatísticas Básicas")
                
                if len(df_api) > 0:
                    # Estatísticas de localização
                    st.write("**Estatísticas de Localização:**")
                    
                    # Divide em colunas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Latitude:")
                        st.write(f"- Mínima: {df_api['latitude'].min():.4f}°")
                        st.write(f"- Máxima: {df_api['latitude'].max():.4f}°")
                        st.write(f"- Média: {df_api['latitude'].mean():.4f}°")
                    
                    with col2:
                        st.write("Longitude:")
                        st.write(f"- Mínima: {df_api['longitude'].min():.4f}°")
                        st.write(f"- Máxima: {df_api['longitude'].max():.4f}°")
                        st.write(f"- Média: {df_api['longitude'].mean():.4f}°")
                    
                    # Distância máxima entre pontos (calculada de forma aproximada)
                    dist_lat = df_api['latitude'].max() - df_api['latitude'].min()
                    dist_lon = df_api['longitude'].max() - df_api['longitude'].min()
                    dist_km = ((dist_lat * 111) ** 2 + (dist_lon * 111) ** 2) ** 0.5
                    
                    st.write(f"**Distância máxima entre pontos:** {dist_km:.2f} km (aproximada)")
                    
                    if 'data_coleta' in df_api.columns and df_api['data_coleta'].notna().any():
                        # Tenta converter para datetime
                        try:
                            df_api['data_coleta'] = pd.to_datetime(df_api['data_coleta'])
                            anos = df_api['data_coleta'].dt.year
                            
                            st.write("**Distribuição temporal:**")
                            st.write(f"- Ano mais antigo: {anos.min()}")
                            st.write(f"- Ano mais recente: {anos.max()}")
                            
                            # Histograma de anos
                            fig, ax = plt.subplots()
                            anos.hist(bins=min(20, anos.nunique()), ax=ax)
                            ax.set_xlabel('Ano')
                            ax.set_ylabel('Número de ocorrências')
                            ax.set_title('Distribuição temporal das ocorrências')
                            st.pyplot(fig)
                        except:
                            st.write("Não foi possível analisar dados temporais.")
                else:
                    st.write("Sem dados suficientes para gerar estatísticas.")

def pagina_pseudoausencias():
    """
    Página para geração de pseudoausências.
    """
    st.title("Geração de Pseudoausências")
    st.write("Gera pontos de pseudoausência utilizando buffers dos pontos de presença (limitado ao Brasil).")
    
    # Expander com explicação sobre pseudoausências
    with st.expander("O que são pseudoausências?"):
        st.write("""
        **Pseudoausências** são pontos artificiais que representam locais onde a espécie provavelmente 
        não ocorre. Eles são necessários para algoritmos como o MaxEnt que requerem dados de presença e ausência.
        
        Nesta implementação, as pseudoausências são geradas:
        1. Criando um buffer ao redor de cada ponto de presença
        2. Unindo todos os buffers para criar uma área de amostragem
        3. Intersectando esta área com o polígono do Brasil
        4. Gerando pontos aleatórios dentro desta área
        
        Isso garante que as pseudoausências estejam em áreas ambientalmente similares às presenças,
        mas não muito próximas a pontos de ocorrência conhecidos.
        """)
    
    if "df_api" in st.session_state:
        df_presence = st.session_state.df_api
        
        if df_presence.empty:
            st.error("Os dados de presença estão vazios. Faça uma nova busca de ocorrências.")
            return
        
        st.write(f"**Dados de presença disponíveis:** {len(df_presence)} pontos")
        
        with st.form(key="formulario_pseudoausencia"):
            col1, col2 = st.columns(2)
            
            with col1:
                n_points = st.slider(
                    "Número de pseudoausências a gerar", 
                    min_value=CONFIG['pseudoausencias']['minimo'], 
                    max_value=CONFIG['pseudoausencias']['maximo'], 
                    value=CONFIG['pseudoausencias']['num_pontos_padrao'], 
                    step=10
                )
            
            with col2:
                # Slider para buffer em Km
                buffer_distance_km = st.slider(
                    "Tamanho do buffer (em Km)", 
                    min_value=200, 
                    max_value=1000, 
                    value=max(200, int(CONFIG['pseudoausencias']['distancia_buffer_padrao'] * 111)), 
                    step=50
                )
            
            # Converte de Km para graus (1 grau ≈ 111 Km)
            buffer_distance_degrees = buffer_distance_km / 111.0
            
            submitted = st.form_submit_button("Gerar Pseudoausências")
            
            if submitted:
                # Verifica se exclusion_radius está presente no CONFIG
                exclusion_radius = CONFIG['pseudoausencias'].get('raio_exclusao_padrao', 2.0)
                pseudo_df = gerar_pseudoausencias_otimizado(df_presence, n_points, buffer_distance_degrees, exclusion_radius)
                
                # Adiciona verificação para evitar erro com NoneType
                if pseudo_df is None:
                    st.error("Ocorreu um erro ao gerar pseudoausências. Nenhum dado foi retornado.")
                elif not pseudo_df.empty:
                    st.session_state.df_pseudo = pseudo_df
                    st.success(f"{len(pseudo_df)} pontos de pseudoausência gerados com sucesso.")
        
        if "df_pseudo" in st.session_state:
            pseudo_df = st.session_state.df_pseudo
            
            # Adiciona verificação para evitar erro com NoneType
            if pseudo_df is None:
                st.error("Dados de pseudoausência inválidos. Tente gerar novamente.")
            elif not pseudo_df.empty:
                st.subheader("Visualização das Pseudoausências")
                
                # Tabs para diferentes visualizações
                tab1, tab2 = st.tabs(["Dados", "Mapa"])
                
                with tab1:
                    st.write("Primeiros registros das pseudoausências:")
                    st.dataframe(pseudo_df.head())
                    
                    # Opção para download
                    csv = pseudo_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV", 
                        csv, 
                        f"pseudoausencias_{st.session_state.especie.replace(' ', '_')}.csv",
                        "text/csv",
                        key='download-pseudo-csv'
                    )
                
                with tab2:
                    # Preparar DataFrames para visualização no mapa
                    df_presence_viz = df_presence.copy()
                    df_presence_viz["marker_color"] = "blue"  # Presenças em azul
                    
                    pseudo_df_viz = pseudo_df.copy()
                    pseudo_df_viz["marker_color"] = "red"  # Pseudoausências em vermelho
                    
                    # Unir os DataFrames para visualização
                    df_combined = pd.concat([df_presence_viz, pseudo_df_viz])
                    
                    # Criar o mapa
                    min_lat = df_combined["latitude"].min()
                    max_lat = df_combined["latitude"].max()
                    min_lon = df_combined["longitude"].min()
                    max_lon = df_combined["longitude"].max()
                    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
                    
                    m = folium.Map(location=center)
                    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
                    
                    # Adicionar presenças (azul)
                    presence_group = folium.FeatureGroup(name="Presenças")
                    for _, row in df_presence.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=3,
                            color="blue",
                            fill=True,
                            fill_color="blue",
                            fill_opacity=1,
                            popup=f"Presença: {row['species']}"
                        ).add_to(presence_group)
                    presence_group.add_to(m)
                    
                    # Adicionar pseudoausências (vermelho)
                    pseudo_group = folium.FeatureGroup(name="Pseudoausências")
                    for _, row in pseudo_df.iterrows():
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=3,
                            color="red",
                            fill=True,
                            fill_color="red",
                            fill_opacity=1,
                            popup="Pseudoausência"
                        ).add_to(pseudo_group)
                    pseudo_group.add_to(m)
                    
                    # Adicionar controle de camadas
                    folium.LayerControl().add_to(m)
                    
                    # Exibir o mapa
                    st_folium(m, width=700, height=500)
    else:
        st.warning("Dados de presença não encontrados. Execute a busca via API primeiro.")
        
        # Link para a página de busca
        st.markdown("[Ir para a página de busca](#Busca-Ocorrência)")


def pagina_variaveis_ambientais():
    """
    Página para análise de variáveis ambientais.
    """
    st.title("Estatísticas Descritivas - Bioclima")
    
    # Expander com informações sobre variáveis bioclimáticas
    with st.expander("Sobre Variáveis Bioclimáticas"):
        st.write("""
        As **variáveis bioclimáticas** são derivadas de dados mensais de temperatura e precipitação. 
        Representam tendências anuais, sazonalidade e fatores ambientais extremos que são relevantes 
        para a distribuição de espécies.
        
        Estas 19 variáveis são frequentemente utilizadas em modelagem de nicho ecológico 
        e distribuição de espécies por capturarem aspectos importantes do clima que influenciam 
        a sobrevivência e reprodução dos organismos.
        """)

    # Tabela explicativa das variáveis BIO
    bio_desc_df = pd.DataFrame(bio_descriptions_pt.items(), columns=["BIO", "Significado"])
    st.subheader("Significado das Variáveis BIO")
    st.dataframe(bio_desc_df)
    
    # Verifica se os dados de presença e pseudoausência estão disponíveis
    if "df_api" not in st.session_state:
        st.warning("Dados de presença não encontrados. Execute a busca via API primeiro.")
        return
    if "df_pseudo" not in st.session_state:
        st.warning("Dados de pseudoausência não encontrados. Gere pseudoausências primeiro.")
        return

    df_occ = st.session_state.df_api
    df_pseudo = st.session_state.df_pseudo
    
    # Verifica se os DataFrames são None ou estão vazios
    if df_occ is None or df_occ.empty:
        st.error("Dados de presença inválidos ou vazios. Faça nova busca de ocorrências.")
        return
    if df_pseudo is None or df_pseudo.empty:
        st.error("Dados de pseudoausência inválidos ou vazios. Gere pseudoausências novamente.")
        return
    
    # Usa uma função cacheada para extrair os dados das variáveis bioclimáticas
    with st.spinner("Extraindo valores das variáveis bioclimáticas..."):
        stats_data, raw_data = obter_estatisticas_bioclim(df_occ, df_pseudo)
        
        if not stats_data or not raw_data:
            st.error("Não foi possível obter dados das variáveis bioclimáticas.")
            return

    # Exibe as estatísticas descritivas dos dados combinados
    df_stats = pd.DataFrame(stats_data).T
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df_stats)
    
    # Opção para download das estatísticas
    csv_stats = df_stats.to_csv().encode('utf-8')
    st.download_button(
        "Download Estatísticas CSV", 
        csv_stats, 
        "estatisticas_bioclim.csv",
        "text/csv",
        key='download-stats-csv'
    )

    # Cria DataFrame com os valores extraídos para correlação e VIF
    raw_data_df = pd.DataFrame({k: pd.Series(v) for k, v in raw_data.items()})
    
    # Análise de correlação
    with st.spinner("Calculando matriz de correlação..."):
        # Limpa os dados antes de calcular a correlação
        raw_data_df_clean = raw_data_df.copy()
        
        # Verifica se há colunas com todos os valores NaN
        all_nan_columns = raw_data_df_clean.columns[raw_data_df_clean.isna().all()].tolist()
        if all_nan_columns:
            st.warning(f"As seguintes variáveis contêm apenas valores NaN e foram removidas da análise: {', '.join(all_nan_columns)}")
            raw_data_df_clean = raw_data_df_clean.drop(columns=all_nan_columns)
        
        # Calcula a matriz de correlação apenas com dados válidos
        corr_matrix = raw_data_df_clean.corr()
        
        st.subheader("Matriz de Correlação Pairwise")
        
        # Exibe o heatmap da matriz de correlação
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Configura o matplotlib para ignorar avisos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True, fmt=".2f", ax=ax, annot_kws={"size": 8})
        
        ax.set_title("Matriz de Correlação entre Variáveis Bioclimáticas")
        st.pyplot(fig)
    
    # Cálculo do VIF incluindo o intercepto (constante)
    with st.spinner("Calculando VIF para todas as variáveis..."):
        try:
            raw_data_df_clean = raw_data_df.dropna()
            if raw_data_df_clean.empty:
                st.warning("Dados insuficientes para cálculo do VIF após remoção de NaNs.")
                return
            
            # Adiciona a constante
            X = add_constant(raw_data_df_clean)
            vif_all = pd.DataFrame()
            vif_all["Feature"] = X.columns
            vif_all["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            st.subheader("VIF - Valores Iniciais para Todas as Variáveis")
            st.dataframe(vif_all)
        except Exception as e:
            st.error(f"Erro ao calcular VIF inicial: {str(e)}")
            logger.error(f"Erro ao calcular VIF: {str(e)}", exc_info=True)
    
    # Eliminação stepwise: remoção das variáveis com maior VIF (excluindo a constante)
    with st.spinner("Realizando eliminação stepwise de variáveis com alto VIF..."):
        try:
            threshold = CONFIG['vif']['limiar']
            variables = list(raw_data_df_clean.columns)
            
            # Expander para mostrar o processo detalhado
            with st.expander("Ver processo de eliminação stepwise"):
                iteration = 1
                
                while True:
                    # Adiciona constante para o cálculo
                    X_temp = add_constant(raw_data_df_clean[variables])
                    vif_df = pd.DataFrame()
                    vif_df["Feature"] = X_temp.columns
                    vif_df["VIF"] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
                    
                    # Exclui a constante da análise para remoção
                    vif_df_no_const = vif_df[vif_df["Feature"] != "const"]
                    max_vif = vif_df_no_const["VIF"].max()
                    
                    # Exibe a tabela da iteração atual
                    st.write(f"**Iteração {iteration}**")
                    st.dataframe(vif_df_no_const)
                    
                    if max_vif < threshold or len(variables) <= 1:
                        st.write(f"**Processo concluído** - Todas as variáveis com VIF < {threshold} ou restou apenas 1 variável")
                        break
                        
                    max_var = vif_df_no_const.sort_values("VIF", ascending=False)["Feature"].iloc[0]
                    max_vif_value = vif_df_no_const.sort_values("VIF", ascending=False)["VIF"].iloc[0]
                    
                    st.write(f"Removendo variável com maior VIF: **{max_var}** (VIF = {max_vif_value:.2f})")
                    variables.remove(max_var)
                    
                    iteration += 1
            
            # Calcula VIF final
            X_final = add_constant(raw_data_df_clean[variables])
            final_vif = pd.DataFrame()
            final_vif["Feature"] = X_final.columns
            final_vif["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
            
            # Remove a constante para exibição
            final_vif = final_vif[final_vif["Feature"] != "const"]
            
            st.subheader("VIF Final Após Eliminação Stepwise")
            st.dataframe(final_vif)
            st.success(f"Variáveis selecionadas para modelagem (VIF < {threshold}): {', '.join(variables)}")
            
            # Armazenar variáveis selecionadas na sessão do Streamlit para uso na modelagem
            # Ignorando a constante, usamos apenas os números das variáveis BIO
            selected_vars_nums = [int(var.replace("BIO", "")) for var in variables]
            st.session_state["selected_variables"] = selected_vars_nums
            
        except Exception as e:
            st.error(f"Erro ao realizar eliminação stepwise: {str(e)}")
            logger.error(f"Erro em eliminação stepwise: {str(e)}", exc_info=True)

    # Permite que o usuário selecione as variáveis para visualização
    selected_vars = st.multiselect(
        "Selecione as variáveis para visualizar no mapa:",
        options=variables if 'variables' in locals() else raw_data_df.columns,
        default=variables[:3] if 'variables' in locals() else raw_data_df.columns[:3]
    )

    if selected_vars:
        with st.spinner("Preparando visualização das variáveis selecionadas..."):
            try:
                # Cria um mapa centrado no polígono do Brasil
                bounds = poligono_brasil.bounds
                m = folium.Map(tiles="OpenStreetMap")
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
                # Para cada variável selecionada, adiciona uma camada raster local
                for var in selected_vars:
                    # Extrai o número da variável (por exemplo, de "BIO3" extrai 3)
                    var_number = int(var.replace("BIO", ""))
                    
                    # Carrega e processa a variável
                    try:
                        raster_path = f"bioclim_brasil/bio_{var_number}_brasil.tif"
                        if os.path.exists(raster_path):
                            # Utiliza os parâmetros de visualização, se disponíveis
                            vis_params = vis_params_dict.get(var_number, {})
                            
                            # Adiciona a camada ao mapa
                            m.add_raster_layer(
                                raster_path,
                                vis_params,
                                f"{var} - {bio_descriptions_pt[var_number]}"
                            )
                    except Exception as e:
                        st.warning(f"Erro ao adicionar {var} ao mapa: {str(e)}")
                        continue
            
                # Adiciona os pontos de presença (preto)
                presence_group = folium.FeatureGroup(name="Presenças")
                for _, row in df_occ.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=3,
                        color="black",
                        fill=True,
                        fill_color="black",
                        fill_opacity=1,
                        popup=f"Presença: {row['species']}"
                    ).add_to(presence_group)
                presence_group.add_to(m)
            
                # Adiciona os pontos de pseudoausência (vermelho)
                pseudo_group = folium.FeatureGroup(name="Pseudoausências")
                for _, row in df_pseudo.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=3,
                        color="red",
                        fill=True,
                        fill_color="red",
                        fill_opacity=1,
                        popup="Pseudoausência"
                    ).add_to(pseudo_group)
                pseudo_group.add_to(m)
            
                # Adiciona um controle de camadas para que o usuário possa ativar/desativar as variáveis
                folium.LayerControl().add_to(m)
            
                st.subheader("Visualização das Variáveis Selecionadas no Brasil")
                st_folium(m, width=700, height=500)
            
            except Exception as e:
                st.error(f"Erro ao criar mapa: {str(e)}")
                logger.error(f"Erro ao criar mapa de variáveis: {str(e)}", exc_info=True)

def executar_modelo():
    """
    Página para execução do modelo de aprendizado de máquina usando Random Forest.
    Utiliza dados de presença e pseudo-ausência para treinar o modelo com variáveis bioclimáticas.
    """
    st.title("Modelagem de Nicho Ecológico")
    
    # ETAPA 1: Verificação de dados
    st.subheader("1️⃣ Dados Disponíveis")
    
    with st.expander("O que são esses dados?", expanded=True):
        st.write("""
        Para treinar o modelo de distribuição potencial da espécie, precisamos de:
        
        - **Pontos de presença**: locais onde a espécie foi observada (pontos em verde)
        - **Pontos de pseudo-ausência**: locais gerados aleatoriamente onde assumimos a ausência da espécie (pontos em vermelho)
        - **Variáveis bioclimáticas**: camadas ambientais que ajudam a prever a ocorrência da espécie
        """)
    
    # Verifica disponibilidade de dados
    if "df_api" not in st.session_state:
        st.warning("⚠️ Dados de presença não encontrados. Execute a busca via API primeiro.")
        return
    if "df_pseudo" not in st.session_state:
        st.warning("⚠️ Dados de pseudo-ausência não encontrados. Gere pseudo-ausências primeiro.")
        return
    
    # Verifica VIF
    if "selected_variables" not in st.session_state:
        # Verifica se já existe a etapa de VIF
        st.warning("⚠️ Variáveis bioclimáticas não pré-selecionadas. Visite a seção Variáveis Ambientais primeiro.")
        if st.button("Ir para Variáveis Ambientais"):
            st.session_state.page = "variaveis_ambientais"
            st.rerun()
        return
    
    df_occ = st.session_state.df_api
    df_pseudo = st.session_state.df_pseudo
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pontos de presença", len(df_occ))
    with col2:
        st.metric("Pontos de pseudo-ausência", len(df_pseudo))
    
    # ETAPA 2: Selecione as variáveis bioclimáticas
    st.subheader("2️⃣ Selecione as Variáveis Bioclimáticas")
    
    with st.expander("Por que selecionar variáveis?", expanded=True):
        st.write("""
        As variáveis bioclimáticas representam diferentes aspectos do clima que podem afetar a distribuição de espécies.
        
        - Usar **muitas variáveis correlacionadas** pode causar **overfitting** (modelo decorar os dados em vez de aprender padrões)
        - Previamente, usamos o **VIF (Fator de Inflação da Variância)** para excluir variáveis altamente correlacionadas
        - Agora, você pode selecionar quais das variáveis pré-selecionadas deseja usar no modelo
        """)
    
    # Recuperar variáveis pré-selecionadas pelo VIF
    pre_selected_variables = st.session_state.get("selected_variables", [])
    if not pre_selected_variables:
        st.error("⚠️ Nenhuma variável pré-selecionada encontrada. Execute a análise VIF primeiro.")
        return
    
    # Interface para seleção de variáveis
    bio_descriptions = CONFIG.get('descricoes_bio', {})
    var_options = [f"BIO{num} - {bio_descriptions.get(num, '')}" for num in pre_selected_variables]
    
    selected_bio_vars = st.multiselect(
        "Selecione as variáveis para usar no modelo:",
        options=var_options,
        default=var_options,
        help="Selecione pelo menos 2 variáveis para o modelo"
    )
    
    # Extrair números das variáveis selecionadas
    selected_vars_nums = []
    if selected_bio_vars:
        for var in selected_bio_vars:
            match = re.match(r"BIO(\d+)", var)
            if match:
                selected_vars_nums.append(int(match.group(1)))
    
    # Verificar se pelo menos 2 variáveis foram selecionadas
    if len(selected_vars_nums) < 2:
        st.warning("⚠️ Selecione pelo menos 2 variáveis para prosseguir.")
        proceed_to_model = False
    else:
        proceed_to_model = True
        st.success(f"✅ {len(selected_vars_nums)} variáveis selecionadas")
    
    # ETAPA 3: Configuração do Modelo
    st.subheader("3️⃣ Configuração do Modelo Random Forest")
    
    with st.expander("O que é Random Forest?", expanded=True):
        st.write("""
        **Random Forest** é um algoritmo de aprendizado de máquina baseado em um conjunto de árvores de decisão.
        
        #### Vantagens:
        - Alta precisão
        - Funciona bem com dados não-lineares
        - Menos propenso a overfitting que árvores individuais
        - Fornece importância das variáveis
        
        #### Parâmetros principais:
        - **n_estimators**: Número de árvores (mais árvores = mais robusto, mas mais lento)
        - **max_depth**: Profundidade máxima das árvores (limitar = reduzir overfitting)
        - **test_size**: Proporção dos dados usados para teste (para validar o modelo)
        - **max_features**: Número máximo de features por árvore (limitar = reduzir overfitting)
        """)
    
    # Formulário para configurar os hiperparâmetros
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider(
            "Número de árvores (n_estimators)", 
            min_value=50, 
            max_value=500, 
            value=100, 
            step=50,
            help="Mais árvores geralmente melhoram o desempenho, mas aumentam o tempo de processamento"
        )
        
        max_features = st.select_slider(
            "Número máximo de features por árvore",
            options=["sqrt", "log2", "all"],
            value="sqrt",
            help="'sqrt': raiz quadrada do número de variáveis, 'log2': log base 2, 'all': todas as variáveis"
        )
        
        if max_features == "all":
            max_features = None
    
    with col2:
        max_depth = st.slider(
            "Profundidade máxima das árvores", 
            min_value=2, 
            max_value=20, 
            value=10, 
            step=1,
            help="Limitar a profundidade ajuda a prevenir overfitting"
        )
        
        test_size = st.slider(
            "Proporção para teste", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.25, 
            step=0.05,
            help="Fração dos dados reservados para avaliar o modelo"
        )
    
    # Semente aleatória para reprodutibilidade
    random_state = st.number_input(
        "Semente aleatória (random_state)", 
        value=42, 
        help="Para garantir reprodutibilidade dos resultados"
    )
    
    # ETAPA 4: Execução do Modelo
    st.subheader("4️⃣ Execução do Modelo")
    
    # Botão para executar o modelo
    if not proceed_to_model:
        st.error("⚠️ Corrija os problemas acima para executar o modelo.")
        model_submitted = False
    else:
        model_submitted = st.button("🚀 Executar Modelo")
    
    # Executar o modelo quando o botão for pressionado
    if model_submitted:
        with st.spinner("🔄 Preparando dados e treinando o modelo..."):
            try:
                # Criação de função para extrair dados bioclimáticos para o conjunto de pontos
                def extrair_bio_para_df(df, bio_nums):
                    """Extrai valores das variáveis bioclimáticas para um DataFrame de pontos."""
                    bio_data = {}
                    
                    for num in bio_nums:
                        try:
                            bio_raster = carregar_var_bioclim(num)
                            values = extrair_valores_em_cache(bio_raster, df, num)
                            bio_data[f"BIO{num}"] = values
                        except Exception as e:
                            st.error(f"Erro ao extrair BIO{num}: {str(e)}")
                            return None
                    
                    # Criar DataFrame com os valores extraídos
                    bio_df = pd.DataFrame(bio_data)
                    
                    # Verificar se há valores faltantes
                    if bio_df.isnull().any().any():
                        st.warning(f"Atenção: {bio_df.isnull().sum().sum()} valores faltantes encontrados.")
                        # Remover linhas com valores faltantes
                        bio_df = bio_df.dropna()
                        st.info(f"Removidas linhas com valores faltantes. Restaram {len(bio_df)} registros.")
                    
                    return bio_df
                
                # 1. Extrair dados bioclimáticos para pontos de presença
                presence_bio = extrair_bio_para_df(df_occ, selected_vars_nums)
                if presence_bio is None or presence_bio.empty:
                    st.error("Erro ao extrair dados bioclimáticos para pontos de presença.")
                    return
                
                # 2. Extrair dados bioclimáticos para pontos de pseudo-ausência
                absence_bio = extrair_bio_para_df(df_pseudo, selected_vars_nums)
                if absence_bio is None or absence_bio.empty:
                    st.error("Erro ao extrair dados bioclimáticos para pontos de pseudo-ausência.")
                    return
                
                # 3. Criar matrizes de características (X) e rótulos (y)
                X_presence = presence_bio.values
                y_presence = np.ones(len(X_presence))
                
                X_absence = absence_bio.values
                y_absence = np.zeros(len(X_absence))
                
                # Coordenadas dos pontos para visualização no mapa
                coords_presence = df_occ[['latitude', 'longitude']].values
                coords_absence = df_pseudo[['latitude', 'longitude']].values
                
                # 4. Combinar os dados de presença e ausência
                X = np.vstack([X_presence, X_absence])
                y = np.hstack([y_presence, y_absence])
                coords = np.vstack([coords_presence, coords_absence])
                
                # 5. Divisão em conjuntos de treino e teste (estratificada)
                X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
                    X, y, coords, test_size=test_size, random_state=random_state, stratify=y
                )
                
                st.info(f"Dados divididos em {len(X_train)} registros para treino e {len(X_test)} para teste.")
                
                # 6. Treinar o modelo Random Forest
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,  # Usar todos os processadores disponíveis
                    min_samples_split=5,  # Exigir pelo menos 5 amostras para dividir um nó
                    min_samples_leaf=2,  # Exigir pelo menos 2 amostras em cada nó folha
                )
                
                # Treinar o modelo
                model.fit(X_train, y_train)
                
                # 7. Avaliação no conjunto de treino
                y_train_pred = model.predict(X_train)
                y_train_proba = model.predict_proba(X_train)[:, 1]
                
                # Métricas de treino
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_precision = precision_score(y_train, y_train_pred)
                train_recall = recall_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred)
                
                # Curva ROC para treino
                train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
                train_auc = auc(train_fpr, train_tpr)
                
                # 8. Avaliação no conjunto de teste
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Métricas de teste
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Curva ROC para teste
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                auc_value = auc(fpr, tpr)
                
                # Matriz de confusão
                cm = confusion_matrix(y_test, y_pred)
                
                # 9. Importância das variáveis
                feature_names = [f"BIO{num}" for num in selected_vars_nums]
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # 10. Armazenar resultados
                results = {
                    "model": model,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_pred_proba": y_pred_proba,
                    "metrics": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "auc": auc_value,
                        "train_accuracy": train_accuracy,
                        "train_precision": train_precision,
                        "train_recall": train_recall,
                        "train_f1": train_f1,
                        "train_auc": train_auc
                    },
                    "roc": {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "train_fpr": train_fpr,
                        "train_tpr": train_tpr
                    },
                    "confusion_matrix": cm,
                    "feature_importance": feature_importance,
                    "selected_variables": feature_names,
                    "coords_test": coords_test,
                    "params": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "max_features": max_features,
                        "test_size": test_size,
                        "random_state": random_state
                    }
                }
                
                # Armazenar resultados na sessão
                st.session_state["model_results"] = results
                
                st.success("✅ Modelo treinado com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao treinar o modelo: {str(e)}")
                logger.error(f"Erro ao treinar modelo: {str(e)}", exc_info=True)
                return
    
    # ETAPA 5: Visualização dos Resultados
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        metrics = results["metrics"]
        
        st.subheader("5️⃣ Resultados do Modelo")
        
        with st.expander("Como interpretar os resultados?", expanded=True):
            st.write("""
            Os resultados do modelo apresentam diferentes métricas e visualizações:
            
            - **Métricas de desempenho**: Acurácia, precisão, recall, F1-score e AUC 
            - **Matriz de confusão**: Mostra verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos
            - **Curva ROC**: Avalia o desempenho do modelo em diferentes limiares de classificação
            - **Importância das variáveis**: Indica quais variáveis contribuem mais para o modelo
            - **Curvas de resposta**: Mostram como cada variável afeta a probabilidade de ocorrência da espécie
            """)
        
        # Métricas de desempenho
        st.subheader("Métricas de Desempenho")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Acurácia", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precisão", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("AUC", f"{metrics['auc']:.3f}")
        
        # Visualizações em duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            # Matriz de Confusão
            st.write("### Matriz de Confusão")
            cm = results["confusion_matrix"]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Previsão')
            ax.set_ylabel('Real')
            ax.set_title('Matriz de Confusão')
            ax.set_xticklabels(['Ausência', 'Presença'])
            ax.set_yticklabels(['Ausência', 'Presença'])
            st.pyplot(fig)
        
        with col2:
            # Curva ROC
            st.write("### Curva ROC")
            fpr, tpr = results["roc"]["fpr"], results["roc"]["tpr"]
            train_fpr, train_tpr = results["roc"]["train_fpr"], results["roc"]["train_tpr"]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, 'b-', label=f'Teste (AUC = {metrics["auc"]:.3f})')
            ax.plot(train_fpr, train_tpr, 'r--', label=f'Treino (AUC = {metrics["train_auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
            ax.set_xlabel('Taxa de Falsos Positivos')
            ax.set_ylabel('Taxa de Verdadeiros Positivos')
            ax.set_title('Curva ROC')
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Visualização de treino vs teste
            st.write("### Treino vs. Teste")
            
            # Dados para o gráfico 
            comparison_data = {
                'Conjunto': ['Treino', 'Teste', 'Treino', 'Teste', 'Treino', 'Teste'],
                'Métrica': ['AUC', 'AUC', 'Acurácia', 'Acurácia', 'F1-Score', 'F1-Score'],
                'Valor': [
                    metrics['train_auc'], 
                    metrics['auc'],
                    metrics['train_accuracy'],
                    metrics['accuracy'],
                    metrics['train_f1'],
                    metrics['f1']
                ]
            }
            
            compare_df = pd.DataFrame(comparison_data)
            
            # Plotar o gráfico de barras
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(x='Métrica', y='Valor', hue='Conjunto', data=compare_df, ax=ax)
            ax.set_ylim(0, 1.05)
            ax.set_title('Comparação de Desempenho: Treino vs. Teste')
            ax.set_ylabel('Valor')
            st.pyplot(fig)
            
            # Configurações usadas
            st.write("### Configurações do Modelo")
            for param, value in results["params"].items():
                st.write(f"**{param}:** {value}")
        
        # Importância das variáveis
        st.subheader("Importância das Variáveis")
        
        with st.expander("Como interpretar a importância?"):
            st.write("""
            O gráfico mostra a contribuição relativa de cada variável para o modelo. Variáveis com maior importância:
            
            - Têm maior influência na previsão do modelo
            - São mais determinantes para prever a ocorrência da espécie
            - Provavelmente representam fatores ambientais mais limitantes para a distribuição da espécie
            """)
        
        feature_importance = results["feature_importance"]
        
        # Adicionar descrições às features
        feature_importance['Descrição'] = feature_importance['Feature'].apply(
            lambda x: CONFIG['descricoes_bio'].get(int(x.replace("BIO", "")), "")
        )
        
        # Renomear colunas para exibição
        feature_importance_display = feature_importance.copy()
        feature_importance_display['Feature_Display'] = feature_importance_display.apply(
            lambda row: f"{row['Feature']}: {row['Descrição'][:30]}{'...' if len(row['Descrição']) > 30 else ''}", 
            axis=1
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature_Display', data=feature_importance_display, ax=ax)
        ax.set_xlabel('Importância Relativa')
        ax.set_ylabel('Variável')
        ax.set_title('Importância das Variáveis no Modelo')
        st.pyplot(fig)
        
        # Curvas de resposta para cada variável climática
        st.subheader("Curvas de Resposta")
        st.write("Relação entre cada variável climática e a probabilidade de ocorrência")
        
        with st.expander("Como interpretar curvas de resposta?"):
            st.write("""
            As curvas de resposta mostram como a probabilidade de ocorrência muda conforme variam os valores de cada variável bioclimática:
            
            - **Eixo X**: Valores da variável bioclimática
            - **Eixo Y**: Probabilidade média de ocorrência
            - **Área sombreada**: Densidade dos dados (frequência dos valores)
            
            #### Padrões comuns:
            - **Curva ascendente**: A espécie prefere valores mais altos da variável
            - **Curva descendente**: A espécie prefere valores mais baixos da variável
            - **Curva em forma de sino**: A espécie prefere valores intermediários (tem um "ótimo")
            - **Curva plana**: A variável tem pouca influência na distribuição da espécie
            """)
        
        model = results.get("model")
        X_test = results.get("X_test")
        selected_variables = results.get("selected_variables", [])
        
        if model and len(X_test) > 0 and len(selected_variables) > 0:
            try:
                # Número de variáveis a exibir (máximo 9 em um grid 3x3)
                num_vars = min(len(selected_variables), 9)
                
                # Calcular número de linhas e colunas para o grid
                n_cols = min(3, num_vars)
                n_rows = (num_vars + n_cols - 1) // n_cols
                
                # Criar um grid de subplots
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                
                # Verificar se axes é um array ou um único eixo
                if num_vars == 1:
                    axes = np.array([axes])
                
                # Achatar para iterar mais facilmente
                axes = axes.flatten()
                
                # Para cada variável, gerar uma curva de resposta
                for i, feature_name in enumerate(selected_variables[:num_vars]):
                    if i < len(axes):
                        ax = axes[i]
                        
                        # Obter o índice da feature
                        feature_idx = selected_variables.index(feature_name)
                        
                        # Criar uma grade de valores para a variável
                        feat_values = X_test[:, feature_idx]
                        unique_values = np.sort(np.unique(feat_values))
                        
                        if len(unique_values) > 100:
                            # Se houver muitos valores únicos, usar um range linear
                            grid = np.linspace(np.min(feat_values), np.max(feat_values), 100)
                        else:
                            grid = unique_values
                        
                        # Calcular a média e o desvio padrão das outras variáveis
                        X_temp = X_test.copy()
                        
                        # Inicializar arrays para armazenar predições
                        mean_preds = []
                        
                        # Para cada valor na grade
                        for val in grid:
                            # Substituir a coluna pela variável com o valor único
                            X_temp_mod = X_temp.copy()
                            X_temp_mod[:, feature_idx] = val
                            
                            # Obter predições
                            y_pred = model.predict_proba(X_temp_mod)[:, 1]
                            
                            # Armazenar a média das predições
                            mean_preds.append(np.mean(y_pred))
                        
                        # Plotar a curva de resposta
                        ax.plot(grid, mean_preds, 'b-')
                        
                        try:
                            # Adicionar área sombreada para indicar a densidade dos dados
                            if len(feat_values) > 5:  # Precisa de pelo menos alguns pontos para KDE
                                kde = gaussian_kde(feat_values)
                                density = kde(grid)
                                scaled_density = 0.1 * density / density.max()
                                ax.fill_between(grid, 0, scaled_density, alpha=0.3, color='gray')
                        except Exception:
                            # Silenciosamente ignora erro de KDE
                            pass
                        
                        # Labels e títulos
                        ax.set_xlabel(f"{feature_name}")
                        ax.set_ylabel("Probabilidade de Ocorrência")
                        
                        # Tentar obter descrição da variável
                        if feature_name.startswith("BIO") and len(feature_name) > 3:
                            try:
                                var_num = int(feature_name[3:])
                                if var_num in CONFIG['descricoes_bio']:
                                    title = f"{feature_name}: {CONFIG['descricoes_bio'][var_num]}"
                                    # Truncar título longo
                                    if len(title) > 40:
                                        title = title[:37] + "..."
                                    ax.set_title(title, fontsize=9)
                                else:
                                    ax.set_title(feature_name)
                            except ValueError:
                                ax.set_title(feature_name)
                        else:
                            ax.set_title(feature_name)
                        
                        ax.grid(True, linestyle='--', alpha=0.6)
                
                # Ocultar eixos não utilizados
                for i in range(num_vars, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erro ao gerar curvas de resposta: {str(e)}")
        else:
            st.warning("Não foi possível gerar as curvas de resposta. Modelo ou dados incompletos.")
        
        # Previsões no conjunto de teste
        st.subheader("Previsões no Conjunto de Teste")
        threshold = st.slider("Limiar de Classificação", 0.0, 1.0, 0.5, 0.01,
                             help="Limiar para classificar como presença")
        
        y_binary = results["y_pred_proba"] >= threshold
        accuracy_threshold = accuracy_score(results["y_test"], y_binary)
        
        st.write(f"**Acurácia com limiar {threshold:.2f}:** {accuracy_threshold:.3f}")
        
        # Visualizar pontos no mapa
        try:
            # Mapa de pontos de teste com previsões
            coords_test = results["coords_test"]
            y_test = results["y_test"]
            y_pred_proba = results["y_pred_proba"]
            
            # Criar DataFrames para visualização
            test_points = pd.DataFrame({
                "latitude": coords_test[:, 0],
                "longitude": coords_test[:, 1],
                "actual": y_test,
                "predicted_prob": y_pred_proba,
                "predicted": y_binary
            })
            
            # Classificar pontos para visualização
            test_points["status"] = "Erro"
            test_points.loc[(test_points["actual"] == 1) & (test_points["predicted"] == 1), "status"] = "Verdadeiro Positivo"
            test_points.loc[(test_points["actual"] == 0) & (test_points["predicted"] == 0), "status"] = "Verdadeiro Negativo"
            test_points.loc[(test_points["actual"] == 0) & (test_points["predicted"] == 1), "status"] = "Falso Positivo"
            test_points.loc[(test_points["actual"] == 1) & (test_points["predicted"] == 0), "status"] = "Falso Negativo"
            
            # Cores por status
            color_map = {
                "Verdadeiro Positivo": "green",
                "Verdadeiro Negativo": "blue",
                "Falso Positivo": "orange",
                "Falso Negativo": "red"
            }
            
            test_points["color"] = test_points["status"].map(color_map)
            
            # Criar o mapa
            min_lat, max_lat = test_points["latitude"].min(), test_points["latitude"].max()
            min_lon, max_lon = test_points["longitude"].min(), test_points["longitude"].max()
            center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
            
            m = folium.Map(location=center)
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
            
            # Adicionar grupos por status
            for status in color_map.keys():
                feature_group = folium.FeatureGroup(name=status)
                status_points = test_points[test_points["status"] == status]
                
                for _, row in status_points.iterrows():
                    folium.CircleMarker(
                        location=[row["latitude"], row["longitude"]],
                        radius=6,
                        color=row["color"],
                        fill=True,
                        fill_color=row["color"],
                        fill_opacity=0.7,
                        popup=f"Status: {row['status']}<br>Prob: {row['predicted_prob']:.3f}"
                    ).add_to(feature_group)
                
                feature_group.add_to(m)
            
            # Adicionar controle de camadas
            folium.LayerControl().add_to(m)
            
            # Exibir o mapa
            st.write("### Visualização das Previsões no Mapa")
            st_folium(m, width=700, height=500)
            
        except Exception as e:
            st.error(f"Erro ao criar mapa: {str(e)}")
            logger.error(f"Erro na visualização: {str(e)}", exc_info=True)
        
        # Salvar o modelo
        with st.expander("Download do Modelo"):
            nome_modelo = st.text_input("Nome do Modelo", 
                                      value=f"RF_Model_{st.session_state.get('especie', 'especie').replace(' ', '_')}")
            
            if st.button("Download do Modelo"):
                try:
                    # Preparar o modelo para download
                    model_bytes = pickle.dumps(results["model"])
                    
                    # Preparar metadados
                    metadata = {
                        "metrics": results["metrics"],
                        "selected_variables": results["selected_variables"],
                        "params": results["params"],
                        "feature_importance": [
                            {"feature": row["Feature"], "importance": float(row["Importance"])}
                            for _, row in results["feature_importance"].iterrows()
                        ],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "species": st.session_state.get("especie", "unknown_species")
                    }
                    
                    # Botão para download do modelo
                    st.download_button(
                        "Download do Modelo (.pkl)",
                        model_bytes,
                        file_name=f"{nome_modelo}.pkl",
                        mime="application/octet-stream"
                    )
                    
                    # Botão para download dos metadados
                    st.download_button(
                        "Download dos Metadados (.json)",
                        json.dumps(metadata, indent=4),
                        file_name=f"{nome_modelo}_metadata.json",
                        mime="application/json"
                    )
                    
                    st.success(f"Clique nos botões acima para baixar o modelo e seus metadados.")
                    
                except Exception as e:
                    st.error(f"Erro ao preparar modelo para download: {str(e)}")
                    logger.error(f"Erro no download: {str(e)}", exc_info=True)

def resultados():
    """
    Página para visualização de resultados.
    """
    st.title("Resultados do Modelo")
    st.warning("AVISO: Esta seção está em desenvolvimento. Os resultados são simulados.")
    
    if "model_results" in st.session_state:
        results = st.session_state["model_results"]
        
        # Tabs para diferentes visualizações
        tabs = st.tabs(["Mapa de Probabilidade", "Estatísticas", "Avaliação", "Download"])
        
        with tabs[0]:
            st.subheader("Mapa de Distribuição Potencial")
            
            # Controles de visualização
            threshold = st.slider(
                "Limiar de Corte (threshold)", 
                0.0, 1.0, 0.5, 0.01,
                help="Valor de corte para presença/ausência"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Mapa de probabilidade contínua
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                cax = ax1.imshow(results["map"], cmap="viridis", interpolation='nearest')
                ax1.set_title("Probabilidade de Ocorrência")
                fig1.colorbar(cax, label="Probabilidade")
                st.pyplot(fig1)
            
            with col2:
                # Mapa binário (acima/abaixo do threshold)
                binary_map = results["map"] >= threshold
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                cax2 = ax2.imshow(binary_map, cmap="Greens", interpolation='nearest')
                ax2.set_title(f"Presença/Ausência (Limiar: {threshold:.2f})")
                st.pyplot(fig2)
            
            # Estatísticas da previsão
            proporção_prevista = np.mean(binary_map)
            st.write(f"**Proporção da área prevista como adequada:** {proporção_prevista:.2%}")
        
        with tabs[1]:
            st.subheader("Estatísticas do Modelo")
            
            # Distribuição de valores de predição (histograma)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(results["map"].flatten(), bins=20, kde=True, ax=ax)
            ax.set_xlabel("Probabilidade Prevista")
            ax.set_ylabel("Frequência")
            ax.set_title("Distribuição de Valores de Predição")
            st.pyplot(fig)
            
            # Estatísticas descritivas
            st.write("**Estatísticas dos Valores Previstos:**")
            pred_values = results["map"].flatten()
            stats = {
                "Mínimo": np.min(pred_values),
                "1º Quartil": np.percentile(pred_values, 25),
                "Mediana": np.median(pred_values),
                "Média": np.mean(pred_values),
                "3º Quartil": np.percentile(pred_values, 75),
                "Máximo": np.max(pred_values),
                "Desvio Padrão": np.std(pred_values)
            }
            
            st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "Valor"}))
        
        with tabs[2]:
            st.subheader("Avaliação do Modelo")
            
            # ROC Curve (simulada)
            st.write("### Curva ROC")
            
            fpr = np.linspace(0, 1, 100)
            tpr_train = fpr ** (1 / (results["auc_train"] * 5))  # Simulação de curva ROC
            tpr_test = fpr ** (1 / (results["auc_test"] * 5))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr_train, 'b-', label=f'Treino (AUC = {results["auc_train"]:.3f})')
            ax.plot(fpr, tpr_test, 'r-', label=f'Teste (AUC = {results["auc_test"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
            ax.set_xlabel('Taxa de Falsos Positivos')
            ax.set_ylabel('Taxa de Verdadeiros Positivos')
            ax.set_title('Curva ROC')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Tabela de importância de variáveis (simulada)
            st.write("### Importância de Variáveis")
            
            # Cria dados simulados de importância
            var_names = ["BIO1", "BIO12", "BIO15", "BIO4", "BIO7"]
            importance = np.random.rand(len(var_names))
            importance = importance / importance.sum() * 100  # Normaliza para percentual
            
            var_import_df = pd.DataFrame({
                'Variável': var_names,
                'Importância (%)': importance,
                'Contribuição': importance,
                'Permutação': importance * (0.8 + 0.4 * np.random.rand(len(var_names)))
            })
            
            var_import_df = var_import_df.sort_values('Importância (%)', ascending=False)
            var_import_df['Importância (%)'] = var_import_df['Importância (%)'].round(2)
            var_import_df['Contribuição'] = var_import_df['Contribuição'].round(2)
            var_import_df['Permutação'] = var_import_df['Permutação'].round(2)
            
            st.dataframe(var_import_df)
            
            # Gráfico de barras de importância
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importância (%)', y='Variável', data=var_import_df, ax=ax)
            ax.set_title('Importância das Variáveis')
            ax.set_xlabel('Importância (%)')
            st.pyplot(fig)
        
        with tabs[3]:
            st.subheader("Download de Resultados")
            
            st.write("**Em uma implementação completa, seriam disponibilizadas opções para download de:**")
            
            st.write("- Arquivo do modelo (formato .asc)")
            st.write("- Relatório em PDF com detalhes da modelagem")
            st.write("- Mapas em formato GeoTIFF")
            st.write("- Tabelas de resultados em CSV")
            
            # Botões de download simulados
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download do Mapa (simulado)"):
                    st.success("Download iniciado (simulação)")
            
            with col2:
                if st.button("Download do Relatório (simulado)"):
                    st.success("Relatório gerado (simulação)")
    else:
        st.warning("Nenhum resultado encontrado. Execute o modelo na seção 'Executar Modelo'.")
        
        # Link para a página de execução
        st.markdown("[Ir para Executar Modelo](#Executar-Modelo)")

def projecao_futura():
    """
    Página para projeção futura.
    """
    st.title("Projeção Futura")
    st.warning("AVISO: Esta seção está em desenvolvimento. As projeções são simuladas.")
    
    if "model_results" not in st.session_state:
        st.warning("Nenhum modelo encontrado. Execute o modelo na seção 'Executar Modelo'.")
        return
    
    # Opções de cenários climáticos
    st.subheader("Configuração da Projeção")
    
    col1, col2 = st.columns(2)
    
    with col1:
        periodo = st.selectbox(
            "Período Futuro",
            options=["2041-2060 (2050)", "2061-2080 (2070)", "2081-2100 (2090)"],
            index=0
        )
    
    with col2:
        cenario = st.selectbox(
            "Cenário de Emissão",
            options=["SSP1-2.6 (Otimista)", "SSP2-4.5 (Intermediário)", "SSP5-8.5 (Pessimista)"],
            index=1
        )
    
    # Modelos climáticos
    modelo_climatico = st.selectbox(
        "Modelo Climático Global",
        options=["Média dos modelos", "CMIP6-ACESS", "CMIP6-MPI-ESM", "CMIP6-NCAR"],
        index=0
    )
    
    # Execução da projeção
    if st.button("Executar Projeção"):
        with st.spinner(f"Processando projeção para {periodo}, cenário {cenario}..."):
            # Simulação de processamento
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)  # Simula processamento
                progress_bar.progress((i + 1) / 100)
            
            # Cria um mapa futuro simulado baseado no mapa atual, mas com alterações
            base_map = st.session_state["model_results"]["map"]
            
            # Fatores de alteração baseados no cenário selecionado
            if "SSP1-2.6" in cenario:
                # Cenário otimista: pequena redução na área adequada
                factor = 0.9
                shift = 0.1  # Deslocamento para o sul
            elif "SSP2-4.5" in cenario:
                # Cenário intermediário: redução moderada
                factor = 0.7
                shift = 0.2
            else:
                # Cenário pessimista: grande redução
                factor = 0.5
                shift = 0.3
            
            # Adiciona um fator temporal
            if "2050" in periodo:
                time_factor = 1.0
            elif "2070" in periodo:
                time_factor = 1.5
            else:  # 2090
                time_factor = 2.0
            
            # Cria o mapa futuro com deslocamento e alteração na intensidade
            rows, cols = base_map.shape
            simulated_future_map = np.zeros_like(base_map)
            
            # Aplica transformação no mapa
            shift_pixels = int(shift * rows * time_factor)
            for i in range(rows):
                for j in range(cols):
                    # Índice de origem com deslocamento para o sul/pólos
                    orig_i = max(0, min(rows-1, i - shift_pixels))
                    # Reduz a intensidade com base nos fatores
                    reduction = factor / time_factor
                    simulated_future_map[i, j] = base_map[orig_i, j] * reduction
            
            # Adiciona alguma variação aleatória
            noise = np.random.normal(0, 0.05, size=simulated_future_map.shape)
            simulated_future_map = np.clip(simulated_future_map + noise, 0, 1)
            
            # Armazena os resultados da projeção
            st.session_state["future_projection"] = {
                "map": simulated_future_map,
                "base_map": base_map,
                "params": {
                    "period": periodo,
                    "scenario": cenario,
                    "model": modelo_climatico
                }
            }
            
            st.success("Projeção realizada com sucesso!")
        
        # Exibe os resultados da projeção
        if "future_projection" in st.session_state:
            proj = st.session_state["future_projection"]
            
            st.subheader("Resultados da Projeção")
            
            # Tabs para diferentes visualizações
            proj_tabs = st.tabs(["Comparação", "Mudanças", "Estatísticas"])
            
            with proj_tabs[0]:
                st.write("### Comparação: Presente vs Futuro")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Mapa presente
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    cax1 = ax1.imshow(proj["base_map"], cmap="viridis", interpolation='nearest')
                    ax1.set_title("Distribuição Atual")
                    fig1.colorbar(cax1, label="Adequabilidade")
                    st.pyplot(fig1)
                
                with col2:
                    # Mapa futuro
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    cax2 = ax2.imshow(proj["map"], cmap="viridis", interpolation='nearest')
                    ax2.set_title(f"Projeção Futura ({proj['params']['period']})")
                    fig2.colorbar(cax2, label="Adequabilidade")
                    st.pyplot(fig2)
            
            with proj_tabs[1]:
                st.write("### Mapa de Mudanças")
                
                # Cálculo da diferença
                diff_map = proj["map"] - proj["base_map"]
                
                # Mapa de diferenças
                fig, ax = plt.subplots(figsize=(10, 8))
                cax = ax.imshow(diff_map, cmap="RdBu_r", interpolation='nearest', vmin=-1, vmax=1)
                ax.set_title("Mudanças na Adequabilidade (Futuro - Presente)")
                cbar = fig.colorbar(cax, label="Mudança")
                cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
                cbar.set_ticklabels(['Perda total', 'Perda parcial', 'Sem mudança', 'Ganho parcial', 'Ganho total'])
                st.pyplot(fig)
                
                # Estatísticas de mudança
                total_area = diff_map.size
                perda = np.sum(diff_map < -0.2) / total_area
                ganho = np.sum(diff_map > 0.2) / total_area
                estavel = 1 - perda - ganho
                
                st.write(f"**Proporção de área com perda de adequabilidade:** {perda:.2%}")
                st.write(f"**Proporção de área com ganho de adequabilidade:** {ganho:.2%}")
                st.write(f"**Proporção de área estável:** {estavel:.2%}")
            
            with proj_tabs[2]:
                st.write("### Estatísticas Comparativas")
                
                # Métricas de mudança de área adequada
                threshold = 0.5  # Limiar para considerar área adequada
                
                area_atual = np.mean(proj["base_map"] >= threshold)
                area_futura = np.mean(proj["map"] >= threshold)
                mudanca_relativa = (area_futura - area_atual) / area_atual * 100
                
                # Estatísticas descritivas
                stats_atual = {
                    "Área adequada (%)": area_atual * 100,
                    "Probabilidade média": np.mean(proj["base_map"]),
                    "Probabilidade máxima": np.max(proj["base_map"])
                }
                
                stats_futura = {
                    "Área adequada (%)": area_futura * 100,
                    "Probabilidade média": np.mean(proj["map"]),
                    "Probabilidade máxima": np.max(proj["map"])
                }
                
                stats_df = pd.DataFrame({
                    "Presente": stats_atual,
                    "Futuro": stats_futura,
                    "Mudança (%)": {
                        "Área adequada (%)": mudanca_relativa,
                        "Probabilidade média": (stats_futura["Probabilidade média"] / stats_atual["Probabilidade média"] - 1) * 100,
                        "Probabilidade máxima": (stats_futura["Probabilidade máxima"] / stats_atual["Probabilidade máxima"] - 1) * 100
                    }
                })
                
                st.dataframe(stats_df)
                
                # Histograma comparativo
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(proj["base_map"].flatten(), bins=20, kde=True, 
                             label="Presente", alpha=0.6, ax=ax)
                sns.histplot(proj["map"].flatten(), bins=20, kde=True, 
                             label="Futuro", alpha=0.6, ax=ax)
                ax.set_xlabel("Adequabilidade")
                ax.set_ylabel("Frequência")
                ax.set_title("Comparação da Distribuição de Adequabilidade")
                ax.legend()
                st.pyplot(fig)


# =============================================================================
# Menu de Navegação (Sidebar) e Execução do App
# =============================================================================

def main():
    """
    Função principal que controla o fluxo do aplicativo e configura a barra lateral.
    """
    
    # Configuração da barra lateral
    st.sidebar.title("TAIPA - Navegação")
    
    # Logo (se existir)
    try:
        st.sidebar.image("logo_taipa.png", width=200)
    except:
        pass
    
    # Menu de navegação
    pagina = st.sidebar.selectbox(
        "Selecione a página", 
        ["Home", "Busca Ocorrência", "Pseudoausências", 
        "Bioclima", "Executar Modelo", "Resultados", "Projeção Futura"]
    )
    
    # Informações na barra lateral
    with st.sidebar.expander("Sobre o TAIPA"):
        st.write("""
        **TAIPA** (Tecnologia de Aprendizado Interativo em Predição Ambiental) 
        é uma plataforma educacional para modelagem de distribuição de espécies.
        
        Desenvolvido como ferramenta de ensino e pesquisa.
        """)
    
    # Exibe a página selecionada
    if pagina == "Home":
        pagina_inicial()
    elif pagina == "Busca Ocorrência":
        pagina_busca_api()
    elif pagina == "Pseudoausências":
        pagina_pseudoausencias()
    elif pagina == "Bioclima":
        pagina_variaveis_ambientais()
    elif pagina == "Executar Modelo":
        executar_modelo()
    elif pagina == "Resultados":
        resultados()
    elif pagina == "Projeção Futura":
        projecao_futura()
    
    # Rodapé
    st.sidebar.markdown("---")
    st.sidebar.markdown("<small>TAIPA v1.0.0 | 2025</small>", unsafe_allow_html=True)
    st.sidebar.markdown("<small>Desenvolvido por Pedro Higuchi</small>", unsafe_allow_html=True)

# Executa o aplicativo
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erro não tratado: {str(e)}")
        logger.error(f"Erro não tratado na aplicação: {str(e)}", exc_info=True)