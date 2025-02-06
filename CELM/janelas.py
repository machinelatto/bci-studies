#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Criado em: 14 de Novembro de 2024
Autor: Manoel
Descrição: Script para segmentar os dados de EEG em janelas de 1 segundo.
"""

# Importação das bibliotecas necessárias
import numpy as np
import scipy.io

# **Função de segmentação dos dados EEG em janelas**
# A função 'janelas' segmenta os dados EEG em janelas de tamanho fixo de 1 segundo.


def janelas(EEG, num_amostras_janela, delay):
    # Obtém as dimensões dos dados EEG (tensor)
    num_eletrodos, num_amostras, num_freqs, num_trials = EEG.shape

    # Calcula o número de janelas possíveis, com base no número de amostras
    num_janelas = int((num_amostras - 250) / num_amostras_janela)

    # Cria um tensor vazio para armazenar os dados segmentados em janelas
    tensor = np.zeros(
        [num_eletrodos, num_amostras_janela, num_freqs, num_trials, num_janelas]
    )

    # itera sobre as frequências de estimulação
    for f in range(num_freqs):
        # Itera sobre as trials (sessões de aquisição)
        for t in range(num_trials):
            # Extrai o trial de interesse, aplicando o delay (tempo de latência) para segmentação
            trial = dados[:, slice(delay, delay + 1250), f, t]

            # Itera sobre o numero de janelas, onde cada janela corresponde a uma seção temporal do sinal EEG de 1 segundo
            for i in range(num_janelas):
                # Define os índices de início e fim da janela
                inicio = i * num_amostras_janela
                fim = (i + 1) * num_amostras_janela
                janela = trial[:, inicio:fim]  # Extrai os dados da janela

                # Salva a janela no tensor
                tensor[:, :, f, t, i] = janela
    return tensor


# **Carregamento dos dados EEG do sujeito 1 (S1)**
# Os dados são carregados a partir de arquivos no formato .npy. Cada arquivo contém um tensor de 4 dimensões:
# - Número de eletrodos
# - Número de amostras no tempo
# - Número de frequências de estimulação
# - Número de trials (sessões de aquisição)
# Neste caso, os dados já estão filtrados

for subject in range(2, 36):

    dados = np.load(f"S{subject}_passa-banda_6_52_Hz.npy")

    # **Carregamento das informações das frequências e fases associadas aos estímulos visuais**
    # O arquivo 'Freq_Phase.mat' contém as frequências e fases associadas aos estímulos visuais.
    # As frequências são arredondadas para duas casas decimais e as fases são extraídas do arquivo.

    frequencias_fases = scipy.io.loadmat("Freq_Phase.mat")
    frequencias = np.round(
        frequencias_fases["freqs"], 2
    )  # Frequências de estimulação, arredondadas
    fases = frequencias_fases["phases"]  # Fases associadas às frequências

    # **Segmentação dos dados a partir de 640ms (500ms sem estimulação + 140ms de latência)**
    # A segmentação é realizada a partir de 640ms, levando em consideração a taxa de amostragem de 250Hz.
    # Como a taxa de amostragem é de 250Hz, isso significa que cada amostra corresponde a 4ms.
    # Portanto, para segmentar os dados a partir de 640ms (que corresponde a 160 amostras),
    # passamos o valor 160 como parâmetro para a função de segmentação. Esse valor de 160 é o número de amostras
    # equivalentes a 640ms, com base na taxa de amostragem de 250Hz.
    # O parâmetro 250 refere-se ao tamanho das janelas, ou seja, 250 amostras correspondem a 1 segundo.
    # O parâmetro 'dados' é o tensor de dados de EEG que estamos segmentando em janelas de 250 amostras.

    dados_segmentados = janelas(dados, 250, 160)

    # O tensor resultante 'dados_segmentados' possui 5 dimensões [64,250,40,6,5]:
    # - 64 eletrodos
    # - 250 amostras por janela (1 segundo)
    # - 40 frequências de estimulação
    # - 6 trials
    # - 5 janelas de 1 segundo por trial

    # **Verificação e salvamento dos dados segmentados**
    # Verifica se há valores nulos (NaN) ou zeros no tensor de dados segmentados.
    # Se não houver problemas, o tensor é salvo em um novo arquivo .npy.

    if np.any(np.isnan(dados_segmentados)) or np.any(dados_segmentados == 0):
        print("O tensor possui valores nulos ou 'não numéricos' (NaN).")
    else:
        print("O tensor não possui valores nulos ou 'não numéricos' (NaN).")

        # O arquivo será salvo com o nome baseado no arquivo original
        nome_do_arquivo = f"S{subject}_passa-banda_6_52_Hz" + "_janelas"
        np.save(nome_do_arquivo + ".npy", dados_segmentados)

    # **Reconstrução de um trial para verificação**
    # Para garantir que a segmentação em janelas foi feita corretamente, uma trial específica é reconstruída a partir das janelas.
    # A reconstrução do trial é feita concatenando as janelas de volta para o formato original e comparando com o trial original.

    # Seleciona a trial número 0 para a frequência de 8Hz
    trial_original = dados[:, slice(125 + 35, 125 + 35 + 1250), 0, 0]

    # Reconstrução da trial original a partir das janelas de 1 segundo
    trial_reconstruida = []

    for k in range(5):
        trial = dados_segmentados[:, :, 0, 0, k]
        trial_reconstruida.append(trial)

    trial_reconstruida = np.concatenate(trial_reconstruida, axis=1)

    # Verifica se a trial reconstruída é igual a trial original
    if np.array_equal(trial_original, trial_reconstruida):
        print("Trial reconstruída a partir das janelas é igual ao trial original")
    else:
        print("Trial reconstruída a partir das janelas é diferente do trial original")
