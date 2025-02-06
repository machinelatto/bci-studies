#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Criado em: 14 de Novembro de 2024
Autor: Manoel
Descrição: Script para filtragem de dados EEG utilizando um filtro Butterworth passa-banda.
"""

# Importação das bibliotecas necessárias
import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt

# **Definição dos parâmetros do filtro**
# Este script utiliza um filtro Butterworth passa-banda para filtrar sinais de EEG na faixa de frequência [6-52] Hz.
# A frequência de amostragem dos sinais é de 250 Hz.

# Parâmetros do filtro
taxa_amostragem = 250  # Frequência de amostragem (Hz)
freq_corte_low = 6  # Frequência de corte inferior (Hz)
freq_corte_high = 52  # Frequência de corte superior (Hz)
ordem_filtro = 10  # Ordem do filtro


# **Construção do filtro passa-banda**
# Cria o filtro passa-banda com os parâmetros especificados. Foi empregada a função butter da biblioteca SciPy.

b, a = signal.butter(
    ordem_filtro,
    [freq_corte_low, freq_corte_high],
    btype="bandpass",
    analog=False,
    output="ba",
    fs=taxa_amostragem,
)

# **Plot da resposta em frequência do filtro**

plt.figure(1, dpi=300)
w, h = signal.freqz(b, a, worN=2000)  # Calcula a resposta em frequência
plt.plot((taxa_amostragem * 0.5 / np.pi) * w, abs(h))  # Converte para Hz no eixo x
plt.title("Resposta em Frequência do Filtro")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# **Carregamento dos dados EEG do sujeito 1 (S1)**
# Os dados são carregados a partir de arquivos no formato .mat. O arquivo é  um tensor de 4 dimensões:
# - Número de eletrodos
# - Número de amostras no tempo
# - Número de frequências de estimulação
# - Número de trials (sessões de aquisição)

for subject in range(2, 36):

    dados = scipy.io.loadmat(
        f"C:\\Users\\machi\\Documents\\Mestrado\\repos\\bci-studies\\data\\benchmark\\S{subject}.mat"
    )["data"]

    # Carrega o arquivo que contém informações das frequências e fases associadas aos estímulos visuais.
    frequencias_fases = scipy.io.loadmat("Freq_Phase.mat")
    frequencias = np.round(
        frequencias_fases["freqs"], 2
    )  # Frequências de estimulação, arredondadas
    fases = frequencias_fases["phases"]  # Fases associadas às frequências

    # **Filtragem dos dados**
    # Aqui é realizado o processo de filtragem para todas as frequências, trials e eletrodos.

    num_eletrodos, num_amostras, num_freqs, num_trials = dados.shape

    # Foi empregada a função filfilt da biblioteca Scipy para criar um filtro de fase zero, garantindo que a fase original dos sinais EEG
    # não seja distorcida durante o processo de filtragem.

    for f in range(num_freqs):  # Para cada frequência de estimulação
        for trial in range(num_trials):  # Para cada trial
            for eletrodo in range(num_eletrodos):  # Para cada eletrodo
                eletrodo_filtrado = signal.filtfilt(
                    b, a, dados[eletrodo, :, f, trial]
                )  # Filtragem com filtfilt
                dados[eletrodo, :, f, trial] = (
                    eletrodo_filtrado  # Substitui o dado original pelo filtrado
                )

    freq_selecionadas = np.array([8, 10, 12, 15])  # Frequências de interesse

    # **Análise de Exemplo: Espectro de Frequência**
    # O objetivo aqui é ilustrar o espectro de frequência de um sinal de EEG após o processo de filtragem.
    # Foi selecionado o eletrodo Oz, a frequência de estimulação de 10 Hz e um trial para plotar o espectro do sinal.
    # eletrodo_Oz = 61  # Índice do eletrodo Oz

    # sinal_10hz = dados[
    #     eletrodo_Oz, :, np.where(frequencias == freq_selecionadas[1])[1], 0
    # ].ravel()
    # fft_sinal = np.fft.fft(sinal_10hz)
    # eixo_frequencias = np.fft.fftfreq(len(sinal_10hz), d=1 / taxa_amostragem)
    # frequencias_pos = eixo_frequencias[eixo_frequencias >= 0]
    # fft_sinal_pos = np.abs(fft_sinal[eixo_frequencias >= 0])

    # # Plote o espectro de frequência
    # plt.figure(figsize=(10, 5), dpi=300)
    # plt.plot(frequencias_pos, fft_sinal_pos)
    # plt.title("Espectro de Frequência do Eletrodo Oz: Estimulação Visual em 10 Hz")
    # plt.xlabel("Frequência (Hz)")
    # plt.ylabel("Magnitude")
    # plt.xlim(0, 120)
    # plt.grid()
    # plt.show()

    # **Verificação e salvamento dos dados**
    # Verifica se há valores nulos ou não numéricos (NaN) no tensor de dados após a filtragem.
    # Caso não seja encontrado nenhum problema, salva em um novo arquivo no formato .npy que pode ser aberto com a biblioteca numpy.
    # O tensor de salvo tem as mesmas dimensões do tensor original
    if np.any(np.isnan(dados)) or np.any(dados == 0):
        print("O tensor possui valores nulos ou 'não numéricos' (NaN).")
    else:
        print("O tensor não possui valores nulos ou 'não numéricos' (NaN).")

        nome_do_arquivo = (
            f"S{subject}"
            + "_passa-banda_"
            + str(freq_corte_low)
            + "_"
            + str(freq_corte_high)
            + "_Hz"
        )
        np.save(nome_do_arquivo + ".npy", dados)
