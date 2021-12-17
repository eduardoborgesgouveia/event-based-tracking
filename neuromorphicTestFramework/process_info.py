
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd




def main():

    folderNames = ['experimento_1','experimento_2','experimento_3','experimento_4_1','experimento_5']
    resultExperimentos = []
    ensaios =[]
    for folderName in folderNames:
      onlyfiles = [filename for filename in os.listdir(folderName) if filename.endswith(r".json")]
      ensaios = []
      for file in onlyfiles:
          f = open(folderName + "/" + file, "r")
          Py_object = json.load(f)
          f.close()
          ensaios.append(Py_object)

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
            if ensaio["qtde_deteccoes_acc_temp_validas"][i] != _qtde_frames:
              vetor_frames.append(ensaio["qtde_predicoes_tensor"][j])
              _qtde_frames += ensaio["qtde_predicoes_tensor"][j]
              j += 1
            else:
              qtde_frames_cinco_deteccoes.append(ensaio["qtde_deteccoes_acc_temp_validas"][i]/len(vetor_frames))
              vetor_frames = []
              _qtde_frames = 0
              j == 0
          media_taxa_deteccoes_acc_temp_frames_totais.append(qtde_frames_cinco_deteccoes)










      # print(media_taxa_deteccoes_acc_temp)
      comp_results = {
        "experimento": folderName,
        "media_taxa_acerto_percentual_tracking": np.mean(media_taxa_acerto_percentual_tracking),
        "desvio_padrao_taxa_acerto_percentual_tracking":np.std(media_taxa_acerto_percentual_tracking),
        "media_taxa_acerto_percentual_deteccao": np.mean(media_taxa_acerto_percentual_deteccao),
        "desvio_padrao_taxa_acerto_percentual_deteccao":np.std(media_taxa_acerto_percentual_deteccao),
        "media_taxa_deteccoes_acc_temp": tolerant_mean(media_taxa_deteccoes_acc_temp),
        "desvio_padrao_taxa_deteccoes_acc_temp":tolerant_std(media_taxa_deteccoes_acc_temp),
        "media_taxa_deteccoes_acc_temp_frames_totais": tolerant_mean(media_taxa_deteccoes_acc_temp_frames_totais),
        "desvio_padrao_taxa_deteccoes_acc_temp_frames_totais":tolerant_std(media_taxa_deteccoes_acc_temp_frames_totais),
        "media_percentagem_sacada_tempo":np.mean(media_percentagem_sacada_tempo),
        "desvio_padrao_percentagem_sacada_tempo":np.std(media_percentagem_sacada_tempo),
        "media_percentagem_sacada_frame":np.mean(media_percentagem_sacada_frames),
        "desvio_padrao_percentagem_sacada_frame":np.std(media_percentagem_sacada_frames),
        "media_tempos_sacadas":tolerant_mean(media_tempo_sacada_raw),
        "desvio_padrao_tempos_sacadas":tolerant_std(media_tempo_sacada_raw),
        "media_deteccoes_consecutivas":np.mean(detec_consecutiva),
        "desvio_padrao_deteccoes_consecutivas":np.std(detec_consecutiva),
        "media_deteccoes_validas":np.mean(deteccoes_validas),
        "media_deteccoes_invalidas":np.mean(deteccoes_invalidas),
        "media_deteccoes_sacadas":np.mean(deteccoes_sacadas),
        "média_tempo_total":np.mean(media_tempo_total),
        "desvio_padrao_tempo_total": np.std(media_tempo_total)
      }
      resultExperimentos.append(comp_results)
    # print(resultExperimentos)
    
    acuracia_geral_tracking = []
    std_acuracia_geral_tracking = []
    acuracia_geral_tracking_modo_2 = []
    std_acuracia_geral_tracking_modo_2 = []
    tempo_medio_sacada = []
    std_tempo_medio_sacada = []
    tempo_medio_sacada_porcentagem = []
    std_tempo_medio_sacada_porcentagem = []
    media_sacada_porcentagem_frame = []
    std_sacada_porcentagem_frame = []
    tempo_total = []
    std_tempo_total = []
    eixo_x_boxPlot = []
    media_tracking_tempo_deteccao = []
    std_tracking_tempo_deteccao= []
    media_tracking_tempo_deteccao_total_frames= []
    std_tracking_tempo_deteccao_total_frames= []
    media_deteccao_valida_consecutiva = []
    std_deteccao_valida_consecutiva = []
    media_deteccao_valida = []
    media_deteccao_invalida = []
    media_deteccao_sacada = []
    tempo_sacada_raw = []
    std_tempo_sacada_raw = []

    for exp in resultExperimentos:
      eixo_x_boxPlot.append(exp["experimento"])
      acuracia_geral_tracking.append(exp["media_taxa_acerto_percentual_tracking"])
      std_acuracia_geral_tracking.append(exp["desvio_padrao_taxa_acerto_percentual_tracking"])
      acuracia_geral_tracking_modo_2.append(exp["media_taxa_acerto_percentual_deteccao"])
      std_acuracia_geral_tracking_modo_2.append(exp["desvio_padrao_taxa_acerto_percentual_deteccao"])
      tempo_medio_sacada_porcentagem.append(exp["media_percentagem_sacada_tempo"])
      std_tempo_medio_sacada_porcentagem.append(exp["desvio_padrao_percentagem_sacada_tempo"])
      tempo_medio_sacada.append(exp["media_percentagem_sacada_tempo"]*exp["média_tempo_total"])
      std_tempo_medio_sacada.append(exp["desvio_padrao_percentagem_sacada_tempo"]*exp["média_tempo_total"])
      tempo_total.append(exp["média_tempo_total"])
      std_tempo_total.append(exp["desvio_padrao_tempo_total"])
      media_sacada_porcentagem_frame.append(exp["media_percentagem_sacada_frame"])
      std_sacada_porcentagem_frame.append(exp["desvio_padrao_percentagem_sacada_frame"])
      media_tracking_tempo_deteccao.append(exp["media_taxa_deteccoes_acc_temp"])
      std_tracking_tempo_deteccao.append(exp["desvio_padrao_taxa_deteccoes_acc_temp"])
      media_tracking_tempo_deteccao_total_frames.append(exp["media_taxa_deteccoes_acc_temp_frames_totais"])
      std_tracking_tempo_deteccao_total_frames.append(exp["desvio_padrao_taxa_deteccoes_acc_temp_frames_totais"])
      media_deteccao_valida_consecutiva.append(exp["media_deteccoes_consecutivas"])
      std_deteccao_valida_consecutiva.append(exp["desvio_padrao_deteccoes_consecutivas"])
      media_deteccao_valida.append(exp["media_deteccoes_validas"])
      media_deteccao_invalida.append(exp["media_deteccoes_invalidas"])
      media_deteccao_sacada.append(exp["media_deteccoes_sacadas"])
      tempo_sacada_raw.append(exp["media_tempos_sacadas"])
      std_tempo_sacada_raw.append(exp["desvio_padrao_tempos_sacadas"])


    sns.set_theme(style="ticks", color_codes=True)
    df = pd.DataFrame({'Acuracia':acuracia_geral_tracking,'Experimentos':eixo_x_boxPlot})
    g = sns.catplot(x="Experimentos", y="Acuracia", kind="violin", inner=None, data=df)
    sns.swarmplot(x="Experimentos", y="Acuracia", color="k", size=3, data=df, ax=g.ax)
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
