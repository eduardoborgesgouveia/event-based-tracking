
import os
import json
from matplotlib.cbook import to_filehandle
import numpy as np
from numpy.core.fromnumeric import size
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib.pyplot import figure

from scipy.stats import pearsonr


def main():
    path_init = "analise_resultados_re_refeitos/"
    folderNames = ['experimento_1','experimento_2','experimento_3','experimento_4','experimento_5']
    resultExperimentos = []
    ensaios =[]
    for folderName in folderNames:
      onlyfiles = [filename for filename in os.listdir(path_init+folderName) if filename.endswith(r".json")]
      ensaios = []
      for file in onlyfiles:
          f = open(path_init + folderName + "/" + file, "r")
          Py_object = json.load(f)
          f.close()
          ensaios.append(Py_object)

      cores = ["darkturquoise","steelblue","darkseagreen","goldenrod","coral" ]
      media_taxa_acerto_percentual_tracking = []
      media_taxa_acerto_percentual_deteccao = []
      media_taxa_deteccoes_acc_temp = []
      media_percentagem_sacada_frames = []
      media_percentagem_sacada_tempo = []
      media_tempo_sacada_raw =[]
      deteccoes_validas = []
      deteccoes_sacadas = []
      deteccoes_invalidas = []
      media_taxa_deteccoes_acc_temp_frames_totais = []
      media_tempo_total = []
      quantidade_frames = []
      dt = ensaios[0]["frame_time"]
      dt = dt/100000
      for ensaio in ensaios:
          print(ensaio["id"])
          media_taxa_acerto_percentual_tracking.append(ensaio["taxa_acerto_percentual_tracking"])
          media_taxa_acerto_percentual_deteccao.append(ensaio["taxa_acerto_percentual_deteccoes"])
          media_taxa_deteccoes_acc_temp.append(ensaio["taxa_deteccoes_acc_temp"])
          media_tempo_sacada_raw.append(ensaio["tempo_sacadas"])
          media_percentagem_sacada_frames.append(ensaio["percentagem_detec_sacada"])
          media_percentagem_sacada_tempo.append(ensaio["porcentagem_tempo_sacada"])
          deteccoes_validas.append(ensaio["percentagem_detec_valida"])
          deteccoes_sacadas.append(ensaio["percentagem_detec_sacada"])
          deteccoes_invalidas.append(ensaio["percentagem_detec_invalida"])
          media_tempo_total.append(ensaio["tempo_alcance"])
          quantidade_frames.append(ensaio["quantidade_frames"])
          detec_consecutiva = []
          count = 0
          for i in range(len(ensaio["tipo_deteccao"])):
            if ensaio["tipo_deteccao"][i] == "valida":
              count += 1
            else:
              detec_consecutiva.append(count)
              count = 0
          detec_consecutiva = [detec for detec in detec_consecutiva if detec != 0]

          j = 0
          qtde_frames_cinco_deteccoes = []
          vetor_frames = []
          _qtde_frames = 0
          for i in range(len(ensaio["qtde_deteccoes_acc_temp_validas"])):
            while ensaio["qtde_deteccoes_acc_temp_validas"][i] > _qtde_frames:
              vetor_frames.append(ensaio["qtde_predicoes_tensor"][j])
              if ensaio["qtde_predicoes_tensor"][j] <=1:
                _qtde_frames += ensaio["qtde_predicoes_tensor"][j]
              else:
                _qtde_frames += 1
              j += 1

            qtde_frames_cinco_deteccoes.append(ensaio["qtde_deteccoes_acc_temp_validas"][i]/len(vetor_frames))
            # vetor_frames = []
            j == 0
          media_taxa_deteccoes_acc_temp_frames_totais.append(qtde_frames_cinco_deteccoes)










      # print(media_taxa_deteccoes_acc_temp)
      comp_results = {
        "experimento": folderName,
        "media_taxa_acerto_percentual_tracking": media_taxa_acerto_percentual_tracking,
        "media_taxa_acerto_percentual_deteccao": media_taxa_acerto_percentual_deteccao,
        "media_taxa_deteccoes_acc_temp": tolerant_mean(media_taxa_deteccoes_acc_temp),
        "desvio_padrao_taxa_deteccoes_acc_temp":tolerant_std(media_taxa_deteccoes_acc_temp),
        "media_taxa_deteccoes_acc_temp_frames_totais": tolerant_mean(media_taxa_deteccoes_acc_temp_frames_totais),
        "desvio_padrao_taxa_deteccoes_acc_temp_frames_totais":tolerant_std(media_taxa_deteccoes_acc_temp_frames_totais),
        "percentagem_sacada_tempo":media_percentagem_sacada_tempo,
        "media_percentagem_sacada_tempo":np.mean(media_percentagem_sacada_tempo),
        "media_percentagem_sacada_frame":media_percentagem_sacada_frames,
        "media_tempos_sacadas":tolerant_mean(media_tempo_sacada_raw),
        "desvio_padrao_tempos_sacadas":tolerant_std(media_tempo_sacada_raw),
        "media_deteccoes_consecutivas":detec_consecutiva,
        "media_deteccoes_validas":np.mean(deteccoes_validas),
        "media_deteccoes_invalidas":np.mean(deteccoes_invalidas),
        "media_deteccoes_sacadas":np.mean(deteccoes_sacadas),
        "media_tempo_total":np.mean(media_tempo_total),
        "tempo_total":media_tempo_total,
        "quantidade_frames":np.mean(quantidade_frames)
      }
      resultExperimentos.append(comp_results)
    # print(resultExperimentos)

    acuracia_geral_tracking = []
    acuracia_geral_tracking_modo_2 = []
    tempo_medio_sacada = []
    tempo_medio_sacada_porcentagem = []
    media_sacada_porcentagem_frame = []
    tempo_total = []
    eixo_x_boxPlot = []
    media_tracking_tempo_deteccao = []
    std_tracking_tempo_deteccao= []
    media_tracking_tempo_deteccao_total_frames= []
    std_tracking_tempo_deteccao_total_frames= []
    media_deteccao_valida_consecutiva = []
    media_deteccao_valida = []
    media_deteccao_invalida = []
    media_deteccao_sacada = []
    tempo_sacada_raw = []
    std_tempo_sacada_raw = []
    quantidade_frames_media = []

    for exp in resultExperimentos:
      eixo_x_boxPlot.append(exp["experimento"])
      acuracia_geral_tracking.append(exp["media_taxa_acerto_percentual_tracking"])
      acuracia_geral_tracking_modo_2.append(exp["media_taxa_acerto_percentual_deteccao"])
      tempo_medio_sacada_porcentagem.append(exp["percentagem_sacada_tempo"])
      tempo_medio_sacada.append(exp["media_percentagem_sacada_tempo"]*exp["media_tempo_total"])
      tempo_total.append(exp["tempo_total"])
      media_sacada_porcentagem_frame.append(exp["media_percentagem_sacada_frame"])
      media_tracking_tempo_deteccao.append(exp["media_taxa_deteccoes_acc_temp"])
      std_tracking_tempo_deteccao.append(exp["desvio_padrao_taxa_deteccoes_acc_temp"])
      media_tracking_tempo_deteccao_total_frames.append(exp["media_taxa_deteccoes_acc_temp_frames_totais"])
      std_tracking_tempo_deteccao_total_frames.append(exp["desvio_padrao_taxa_deteccoes_acc_temp_frames_totais"])
      media_deteccao_valida_consecutiva.append(exp["media_deteccoes_consecutivas"])
      media_deteccao_valida.append(exp["media_deteccoes_validas"])
      media_deteccao_invalida.append(exp["media_deteccoes_invalidas"])
      media_deteccao_sacada.append(exp["media_deteccoes_sacadas"])
      tempo_sacada_raw.append(exp["media_tempos_sacadas"])
      std_tempo_sacada_raw.append(exp["desvio_padrao_tempos_sacadas"])
      quantidade_frames_media.append(exp["quantidade_frames"])


    # fig = plt.figure()
    # plt.boxplot(acuracia_geral_tracking)
    # plt.show()
    plt.rcParams["figure.figsize"] = (8,4)
    if True:
      fig1, ax1 = plt.subplots()
      arr = np.array(acuracia_geral_tracking)
      arr = arr*100
      i = 1
      for exp in arr:
        bp = ax1.boxplot(exp,positions=[i],patch_artist=True,medianprops=dict(color="black"))
        for patch in bp['boxes']:
          patch.set_facecolor(cores[i-1])
          i += 1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_ylim((0, 110))
      ax1.set_title('Acurácia geral do modelo de rastreio ativo - modo 1')
      plt.xticks([1, 2, 3,4,5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Acurácia [%]")
      plt.savefig(path_init+'/acuracia_geral_modo_1.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      # bp = ax1.boxplot(acuracia_geral_tracking_modo_2,patch_artist=True,medianprops=dict(color="black"))
      arr = np.array(acuracia_geral_tracking_modo_2)
      arr = arr*100
      i = 1
      for exp in arr:
        bp = ax1.boxplot(exp,positions=[i],patch_artist=True,medianprops=dict(color="black"))
        for patch in bp['boxes']:
          patch.set_facecolor(cores[i-1])
          i += 1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_ylim((0, 110))
      ax1.set_title('Acurácia geral do modelo de rastreio ativo - modo 2')
      plt.xticks([1, 2, 3, 4, 5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Acurácia [%]")
      plt.savefig(path_init+'/acuracia_geral_modo_2.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      bp = ax1.boxplot(tempo_medio_sacada_porcentagem,patch_artist=True,medianprops=dict(color="black"))
      i = 0
      for patch in bp['boxes']:
        patch.set_facecolor(cores[i])
        i += 1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Tempo médio do movimento sacádico')
      plt.xticks([1, 2, 3, 4, 5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Tempo [s]")
      plt.savefig(path_init+'/tempo_medio_movimento_sacadico.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      bp = ax1.boxplot(tempo_total,patch_artist=True,medianprops=dict(color="black"))
      i = 0
      for patch in bp['boxes']:
        patch.set_facecolor(cores[i])
        i += 1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Tempo médio de duração do alcance')
      plt.xticks([1, 2, 3,4,5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Tempo médio [s]")
      plt.savefig(path_init+'/tempo_medio_experimentos.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      tempo_total
      # bp = ax1.boxplot(media_sacada_porcentagem_frame,patch_artist=True,medianprops=dict(color="black"))
      arr = np.array(media_sacada_porcentagem_frame)
      #arr = arr*tempo_total
      i = 1
      for exp in arr:
        bp = ax1.boxplot(exp*tempo_total[i-1],positions=[i],patch_artist=True,medianprops=dict(color="black"))
        for patch in bp['boxes']:
          patch.set_facecolor(cores[i-1])
          i += 1

      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Média de tempo de movimentos sacádicos')
      plt.xticks([1, 2, 3, 4, 5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Tempo [s]")
      plt.savefig(path_init+'/media_tempo_movimento_sacadico.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      tempo_total
      # bp = ax1.boxplot(media_sacada_porcentagem_frame,patch_artist=True,medianprops=dict(color="black"))
      arr = np.array(media_sacada_porcentagem_frame)
      #arr = arr*tempo_total
      i = 1
      for exp in arr:
        bp = ax1.boxplot(exp*quantidade_frames_media[i-1],positions=[i],patch_artist=True,medianprops=dict(color="black"))
        for patch in bp['boxes']:
          patch.set_facecolor(cores[i-1])
          i += 1

      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Quantidade de movimentos sacádicos')
      plt.xticks([1, 2, 3, 4, 5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Quantidade [un]")
      plt.savefig(path_init+'/quantidade_media_movimento_sacadico.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      bp = ax1.boxplot(media_deteccao_valida_consecutiva,patch_artist=True,medianprops=dict(color="black"))
      i = 0
      for patch in bp['boxes']:
        patch.set_facecolor(cores[i])
        i += 1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Média de detecções válidas consecutivas')
      plt.xticks([1, 2, 3, 4, 5], eixo_x_boxPlot)
      # plt.xlabel("Experimentos")
      plt.ylabel("Quantidade [un]")
      plt.savefig(path_init+'/quantidade_deteccoes_validas_consecutivas.png')
      plt.show()


      plt.rcParams["figure.figsize"] = (12,8)
      experimentos = []
      for i in range(len(media_deteccao_invalida)):
        max_value = max(media_deteccao_invalida[i],media_deteccao_sacada[i],media_deteccao_valida[i])
        media_deteccao_invalida[i] =( media_deteccao_invalida[i]/max_value )* 100
        media_deteccao_sacada[i] = (media_deteccao_sacada[i]/max_value )* 100
        media_deteccao_valida[i] = (media_deteccao_valida[i]/max_value) * 100
        experimentos.append([media_deteccao_sacada[i],media_deteccao_valida[i],media_deteccao_invalida[i]])

      categories = ["Sacada", "Válida", "Invalida"]
      N = len(categories)
      j = 0
      for exp in experimentos:
        values = np.array([exp[0], exp[1], exp[2]]).tolist()
        j += 1
        values += values[:1]
        angles = [n/float(N)*2* math.pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(230+j, polar=True)
        plt.polar(angles,values,marker='.',color=cores[j-1])
        plt.fill(angles,values,alpha=0.3, color=cores[j-1])
        plt.xticks(angles[:-1], categories)
        plt.yticks([0,50,100],color="gray",size= 7)
        plt.ylim(0,105)
        ax.set_rlabel_position(25.5)
        ax.set_title("Experimento - " + str(j), va='bottom')

      plt.savefig(path_init+'/distribuicao_tipos_deteccao.png')
      plt.show()


      plt.rcParams["figure.figsize"] = (9,5)
    # plotar uma série temporal dos valores de tempo de cada sacada ao longo do movimento de alcance ##
      media_das_media = tolerant_mean(media_tracking_tempo_deteccao)
      std__media_das_media = tolerant_std(media_tracking_tempo_deteccao)
      fig1, ax1 = plt.subplots()
      maior = 0
      for ser in media_tracking_tempo_deteccao:
        tamanho = len(ser)
        if tamanho > maior:
          maior = tamanho
      tempo = np.linspace(0,maior,maior)
      i = 0
      for ser in media_tracking_tempo_deteccao:
        plt.plot(tempo[:len(ser)]*(dt),ser*100,label=eixo_x_boxPlot[i],color=cores[i])
        plt.fill_between(tempo[:len(ser)]*(dt),(ser*100-std_tracking_tempo_deteccao[i]*100),(ser*100+std_tracking_tempo_deteccao[i]*100),color=cores[i],alpha=0.09)
        i+=1
      tempo_media = len(media_das_media)
      tempo_media = np.linspace(0,tempo_media,tempo_media)
      plt.plot(tempo_media*dt,media_das_media*100,label="média de acurácia",color="grey")
      plt.fill_between(tempo_media*dt,(media_das_media*100-std__media_das_media*100),(media_das_media*100+std__media_das_media*100),color="grey",alpha=0.09)
      ax1.set_title('Rastreio ativo - acurácia rastreio ao longo do tempo - modo 2')
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      plt.xlabel("Tempo [s]")
      plt.ylabel("Acurácia [%]")
      plt.legend()
      plt.savefig(path_init+'/acc_rastreio_ao_longo_do_tempo_modo_2.png')
      plt.show()


      plt.rcParams["figure.figsize"] = (12,10)
      media_das_media = tolerant_mean(media_tracking_tempo_deteccao_total_frames)
      std__media_das_media = tolerant_std(media_tracking_tempo_deteccao_total_frames)
      fig1, ax1 = plt.subplots()
      maior = 0
      for ser in media_tracking_tempo_deteccao_total_frames:
        tamanho = len(ser)
        if tamanho > maior:
          maior = tamanho
      tempo = np.linspace(0,maior,maior)
      i = 0
      for ser in media_tracking_tempo_deteccao_total_frames:
        plt.plot(tempo[:len(ser)]*(dt),ser*100,label=eixo_x_boxPlot[i],color=cores[i])
        plt.fill_between(tempo[:len(ser)]*(dt),(ser*100-std_tracking_tempo_deteccao_total_frames[i]*100),(ser*100+std_tracking_tempo_deteccao_total_frames[i]*100),color=cores[i],alpha=0.09)
        i+=1
      tempo_media = len(media_das_media)
      tempo_media = np.linspace(0,tempo_media,tempo_media)
      plt.plot(tempo_media*dt,media_das_media*100,label="média de acurácia",color="grey")
      plt.fill_between(tempo_media*dt,(media_das_media*100-std__media_das_media*100),(media_das_media*100+std__media_das_media*100),color="grey",alpha=0.09)
      ax1.set_title('Rastreio ativo - acurácia rastreio ao longo do tempo - modo 1')
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      plt.xlabel("Tempo [s]")
      plt.ylabel("Acurácia [%]")
      plt.legend(loc=1)
      plt.savefig(path_init+'/acc_rastreio_ao_longo_do_tempo_modo_1.png')
      plt.show()


      plt.rcParams["figure.figsize"] = (20,6)
      media_das_media_tempo_sacada_raw = tolerant_mean(tempo_sacada_raw)
      std__media_tempo_sacada_raw = tolerant_std(tempo_sacada_raw)
      fig1, ax1 = plt.subplots()
      maior = 0
      for ser in tempo_sacada_raw:
        tamanho = len(ser)
        if tamanho > maior:
          maior = tamanho
      tempo = np.linspace(0,maior,maior)
      i = 0
      for ser in tempo_sacada_raw:
        plt.scatter(tempo[:len(ser)],ser*1000,s=std_tempo_sacada_raw[i]*1000,label=eixo_x_boxPlot[i],color=cores[i])
        i+=1
      tempo_media = len(media_das_media_tempo_sacada_raw)
      tempo_media = np.linspace(0,tempo_media,tempo_media)
      plt.plot(tempo_media,media_das_media_tempo_sacada_raw*1000,label="média do tempo da sacada",color="grey")
      plt.fill_between(tempo_media,(media_das_media_tempo_sacada_raw-std__media_tempo_sacada_raw)*1000,(media_das_media_tempo_sacada_raw+std__media_tempo_sacada_raw)*1000,color="grey",alpha=0.15)
      ax1.set_title('Tempo gasto com movimentos sacadicos ao longo do alcance')
      plt.xlabel("Movimento")
      plt.ylabel("Tempo [ms]")
      plt.legend()
      plt.savefig(path_init+'/tempo_gasto_movimento_sacadico.png')
      plt.show()

    ########################################################################################################################################
    ##Gráfico comparando o experimento 3 e o 5 ###
    ########################################################################################################################################


    cores = ["darkseagreen","coral" ]

    plt.rcParams["figure.figsize"] = (8,4)
    if True:
      fig1, ax1 = plt.subplots()
      arr = np.array(acuracia_geral_tracking)
      arr = arr*100
      i = 1
      j = 0
      for exp in arr:
        if j == 2 or j == 4:
          bp = ax1.boxplot(exp,positions=[i],patch_artist=True,medianprops=dict(color="black"))
          for patch in bp['boxes']:
            patch.set_facecolor(cores[i-1])
            i += 1
        j +=1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_ylim((0, 110))
      ax1.set_title('Acurácia geral do modelo de rastreio ativo - modo 1')
      plt.xticks([1,2],[eixo_x_boxPlot[2],eixo_x_boxPlot[4]])
      # plt.xlabel("Experimentos")
      plt.ylabel("Acurácia [%]")
      plt.savefig(path_init+'/exp_3_5_acuracia_geral_modo_1.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      # bp = ax1.boxplot(acuracia_geral_tracking_modo_2,patch_artist=True,medianprops=dict(color="black"))
      arr = np.array(acuracia_geral_tracking_modo_2)
      arr = arr*100
      i = 1
      j = 0
      for exp in arr:
        if j == 2 or j == 4:
          bp = ax1.boxplot(exp,positions=[i],patch_artist=True,medianprops=dict(color="black"))
          for patch in bp['boxes']:
            patch.set_facecolor(cores[i-1])
            i += 1
        j +=1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_ylim((0, 110))
      ax1.set_title('Acurácia geral do modelo de rastreio ativo - modo 2')
      plt.xticks([1,2],[eixo_x_boxPlot[2],eixo_x_boxPlot[4]])
      plt.ylabel("Acurácia [%]")
      plt.savefig(path_init+'/exp_3_5_acuracia_geral_modo_2.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      bp = ax1.boxplot(tempo_medio_sacada_porcentagem[2],positions=[1],patch_artist=True,medianprops=dict(color="black"))
      i = 0
      for patch in bp['boxes']:
        patch.set_facecolor(cores[i])
        i += 1
      bp = ax1.boxplot(tempo_medio_sacada_porcentagem[4],positions=[2],patch_artist=True,medianprops=dict(color="black"))
      for patch in bp['boxes']:
        patch.set_facecolor(cores[i])
        i += 1
      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Tempo médio do movimento sacádico')
      plt.xticks([1,2],[eixo_x_boxPlot[2],eixo_x_boxPlot[4]])
      plt.ylabel("Tempo [s]")
      plt.savefig(path_init+'/exp_3_5_tempo_medio_movimento_sacadico.png')
      plt.show()

      fig1, ax1 = plt.subplots()
      tempo_total
      arr = np.array(media_sacada_porcentagem_frame)
      i = 1
      j = 0
      for exp in arr:
        if j == 2 or j == 4:
          bp = ax1.boxplot(exp*tempo_total[i-1],positions=[i],patch_artist=True,medianprops=dict(color="black"))
          for patch in bp['boxes']:
            patch.set_facecolor(cores[i-1])
            i += 1
        j +=1

      ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
      ax1.spines['top'].set_visible(False)
      ax1.spines['right'].set_visible(False)
      ax1.spines['left'].set_visible(False)
      ax1.set_title('Média de tempo de movimentos sacádicos')
      plt.xticks([1,2],[eixo_x_boxPlot[2],eixo_x_boxPlot[4]])
      plt.ylabel("Tempo [s]")
      plt.savefig(path_init+'/exp_3_5_media_tempo_movimento_sacadico.png')
      plt.show()




      if True:
        fig1, ax1 = plt.subplots()
        tempo_total
        # bp = ax1.boxplot(media_sacada_porcentagem_frame,patch_artist=True,medianprops=dict(color="black"))
        arr = np.array(media_sacada_porcentagem_frame)
        #arr = arr*tempo_total
        i = 1
        j = 0
        for exp in arr:
          if j == 2 or j == 4:
            bp = ax1.boxplot(exp*quantidade_frames_media[i-1],positions=[i],patch_artist=True,medianprops=dict(color="black"))
            for patch in bp['boxes']:
              patch.set_facecolor(cores[i-1])
              i += 1
          j += 1
        ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_title('Quantidade de movimentos sacádicos')
        plt.xticks([1,2],[eixo_x_boxPlot[2],eixo_x_boxPlot[4]])
        # plt.xlabel("Experimentos")
        plt.ylabel("Quantidade [un]")
        plt.savefig(path_init+'/exp_3_5_quantidade_media_movimento_sacadico.png')
        plt.show()

        fig1, ax1 = plt.subplots()
        bp = ax1.boxplot(media_deteccao_valida_consecutiva[2],positions=[1],patch_artist=True,medianprops=dict(color="black"))
        i = 0
        for patch in bp['boxes']:
          patch.set_facecolor(cores[i])
          i += 1
        bp = ax1.boxplot(media_deteccao_valida_consecutiva[4],positions=[2],patch_artist=True,medianprops=dict(color="black"))
        for patch in bp['boxes']:
          patch.set_facecolor(cores[i])
          i += 1
        ax1.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_title('Média de detecções válidas consecutivas')
        plt.xticks([1,2],[eixo_x_boxPlot[2],eixo_x_boxPlot[4]])
        # plt.xlabel("Experimentos")
        plt.ylabel("Quantidade [un]")
        plt.savefig(path_init+'/exp_3_5_quantidade_deteccoes_validas_consecutivas.png')
        plt.show()


        plt.rcParams["figure.figsize"] = (9,5)
      # plotar uma série temporal dos valores de tempo de cada sacada ao longo do movimento de alcance ##
        media_das_media = tolerant_mean(media_tracking_tempo_deteccao)
        std__media_das_media = tolerant_std(media_tracking_tempo_deteccao)
        fig1, ax1 = plt.subplots()
        maior = 0
        for ser in media_tracking_tempo_deteccao:
          tamanho = len(ser)
          if tamanho > maior:
            maior = tamanho
        tempo = np.linspace(0,maior,maior)
        i = 0
        j = 0
        for ser in media_tracking_tempo_deteccao:
          if i == 2 or i == 4:
            plt.plot(tempo[:len(ser)]*(dt),ser*100,label=eixo_x_boxPlot[i],color=cores[j])
            plt.fill_between(tempo[:len(ser)]*(dt),(ser*100-std_tracking_tempo_deteccao[i]*100),(ser*100+std_tracking_tempo_deteccao[i]*100),color=cores[j],alpha=0.09)
            j += 1
          i+=1
        # tempo_media = len(media_das_media)
        # tempo_media = np.linspace(0,tempo_media,tempo_media)
        # plt.plot(tempo_media*dt,media_das_media*100,label="média de acurácia",color="grey")
        # plt.fill_between(tempo_media*dt,(media_das_media*100-std__media_das_media*100),(media_das_media*100+std__media_das_media*100),color="grey",alpha=0.09)
        ax1.set_title('Rastreio ativo - acurácia rastreio ao longo do tempo - modo 2 - deteccao')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        plt.xlabel("Tempo [s]")
        plt.ylabel("Acurácia [%]")
        plt.legend()
        plt.savefig(path_init+'/exp_3_5_acc_rastreio_ao_longo_do_tempo_modo_2.png')
        plt.show()


        plt.rcParams["figure.figsize"] = (12,10)
        media_das_media = tolerant_mean(media_tracking_tempo_deteccao_total_frames)
        std__media_das_media = tolerant_std(media_tracking_tempo_deteccao_total_frames)
        fig1, ax1 = plt.subplots()
        maior = 0
        for ser in media_tracking_tempo_deteccao_total_frames:
          tamanho = len(ser)
          if tamanho > maior:
            maior = tamanho
        tempo = np.linspace(0,maior,maior)
        i = 0
        j = 0
        for ser in media_tracking_tempo_deteccao_total_frames:
          if i == 2 or i == 4:
            plt.plot(tempo[:len(ser)]*(dt),ser*100,label=eixo_x_boxPlot[i],color=cores[j])
            plt.fill_between(tempo[:len(ser)]*(dt),(ser*100-std_tracking_tempo_deteccao_total_frames[i]*100),(ser*100+std_tracking_tempo_deteccao_total_frames[i]*100),color=cores[j],alpha=0.09)
            j +=1
          i+=1
        # tempo_media = len(media_das_media)
        # tempo_media = np.linspace(0,tempo_media,tempo_media)
        # plt.plot(tempo_media*dt,media_das_media*100,label="média de acurácia",color="grey")
        # plt.fill_between(tempo_media*dt,(media_das_media*100-std__media_das_media*100),(media_das_media*100+std__media_das_media*100),color="grey",alpha=0.09)
        ax1.set_title('Rastreio ativo - acurácia rastreio ao longo do tempo - modo 1 - total frames')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        plt.xlabel("Tempo [s]")
        plt.ylabel("Acurácia [%]")
        plt.legend(loc=1)
        plt.savefig(path_init+'/exp_3_5_acc_rastreio_ao_longo_do_tempo_modo_1.png')
        plt.show()




      plt.rcParams["figure.figsize"] = (20,6)
      fig1, ax1 = plt.subplots()
      maior = 0
      for ser in tempo_sacada_raw:
        tamanho = len(ser)
        if tamanho > maior:
          maior = tamanho
      tempo = np.linspace(0,maior,maior)
      data1 = []
      data2 =[]
      i = 0
      j = 0
      for ser in tempo_sacada_raw:
        if i == 2 or i == 4:
          # val = np.polyfit(tempo[:len(ser)], ser*1000, 8)
          # y = np.polyval(val,tempo[:len(ser)])
          # correlation = np.corrcoef(val,tempo[:len(ser)])[0,1]
          # det = correlation*correlation
          plt.scatter(tempo[:len(ser)],ser*1000,s=std_tempo_sacada_raw[i]*1000,label=eixo_x_boxPlot[i],color=cores[j])
          # plt.plot(tempo[:len(ser)], y,color=cores[j],label=str(det))
          if i == 4:
            data1 =ser*1000
          elif i == 2:
            data2 = ser*1000
          j +=1

        i+=1
      # data2_m = data2[:len(data1)]
      # res = -data2_m + data1
      # print("média dos tempos gastos com sacadas entre o experimento 3 e 5: ",np.mean((res)), "ms - porcentagem: ", 100*(np.mean(res)/np.max((np.max(data1),np.max(data2_m)))), "%" )
      if False:
        plt.plot(tempo[:len(res)],np.ones(len(res))*np.mean((res)),label="média do valor de erro entre as duas curvas",color="grey")
      # corr, _ = pearsonr(data1, data2[:len(data1)])
      # print(print('Pearsons correlation: %.3f' % corr))
      ax1.set_title('Tempo gasto com movimentos sacadicos ao longo do alcance')
      plt.xlabel("Movimento")
      plt.ylabel("Tempo [ms]")
      plt.legend()
      plt.savefig(path_init+'/exp_3_5_tempo_gasto_movimento_sacadico.png')
      plt.show()













def tolerant_mean(arrs):
  lens = [len(i) for i in arrs]
  arr = np.ma.empty((np.max(lens),len(arrs)))
  arr.mask = True
  for idx, l in enumerate(arrs):
      arr[:len(l),idx] = l
  return arr.mean(axis = -1)

def tolerant_std(arrs):
  lens = [len(i) for i in arrs]
  arr = np.ma.empty((np.max(lens),len(arrs)))
  arr.mask = True
  for idx, l in enumerate(arrs):
      arr[:len(l),idx] = l
  return  arr.std(axis=-1)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
