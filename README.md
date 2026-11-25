# Classificação de Tumores Cerebrais em Ressonância Magnética com Deep Learning

Projeto da disciplina **SIN 393 – Introdução à Visão Computacional (2026)** · UFV – Campus Rio Paranaíba · Prof. João Fernando Mari

**Integrantes:** Hugo Resende Babos 9858 e Lucas Santos de Freitas 9859

**Vídeo de apresentação:** [COLOCAR AQUI O LINK DO YOUTUBE]

## Descrição

Classificação de imagens de ressonância magnética (RM) cerebral em quatro classes (`glioma_tumor`, `meningioma_tumor`, `pituitary_tumor` e `normal`) com CNNs do torchvision, em PyTorch.

**Questão experimental:** qual o impacto da arquitetura (ResNet-50 vs. EfficientNet-B0) e do data augmentation no desempenho?

**Dataset:** [Brain tumors 256x256 (Kaggle)](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256), em uma [Versão limpa](https://drive.google.com/drive/folders/19TAW66GtVFAyEz0gczUHGZLkUIrkkMAZ?usp=sharing), da qual foram removidas previamente imagens duplicadas e muito semelhantes utilizando [difPy](https://difpy.app/) (2.821 imagens após a limpeza).

## Metodologia

- **Divisão:** 70% treino (1.974) / 15% validação (423) / 15% teste (424), com `train_test_split` estratificado e semente fixa (42). O teste é usado apenas na avaliação final.
- **Pré-processamento:** conversão para RGB, resize 224×224 e normalização do ImageNet.
- **Data augmentation (só no treino):** recorte aleatório (`RandomResizedCrop`, escala 0,8–1,0), flip horizontal, rotação de até 10° e `ColorJitter` leve (brilho e contraste). Sem flip vertical e sem saturação/matiz.
- **Modelos:** ResNet-50 e EfficientNet-B0, com transfer learning (ImageNet) e fine-tuning completo.
- **Treino:** `CrossEntropyLoss`, AdamW (lr 1e-4, weight decay 1e-2), batch 32, até 30 épocas com early stopping (paciência 5, sobre a acurácia de validação). O melhor checkpoint é escolhido pela acurácia de validação.
- **Experimentos:** {ResNet-50, EfficientNet-B0} × {com, sem augmentation}, uma execução por configuração.
- **Avaliação:** matriz de confusão, acurácia, precisão, recall e F1 (agregação macro e weighted) e análise qualitativa de acertos e erros.

## Como executar

**Requisitos:** Python 3.10+, `torch`, `torchvision`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `pillow` e `jupyter` (`requirements.txt`). GPU recomendada.

### Google Colab (recomendado)

1. Abra `SIN393_brain_tumor_classification_adamw_es.ipynb` no Colab e ative a GPU (*Ambiente de execução > Alterar tipo > GPU*).
2. Baixe o dataset (de preferência a versão limpa) e envie-o ao Google Drive (`.zip`). O notebook monta o Drive e extrai o arquivo no disco local do Colab.
3. Ajuste o caminho do `.zip` e `DATA_DIR` (seção 2 do notebook) para a pasta que contém diretamente as pastas das classes.
4. Execute as células em ordem. Dica: teste antes com `EPOCHS = 2`.

### Localmente

```bash
git clone <URL-DO-REPOSITORIO> && cd <NOME-DO-REPOSITORIO>
pip install -r requirements.txt
jupyter notebook SIN393_brain_tumor_classification_adamw_es.ipynb   # remova as células do Google Drive e ajuste DATA_DIR para o caminho local do dataset
```

## Resultados

Avaliação no conjunto de teste (424 imagens). O artigo completo está em `artigo_tumores_cerebrais.pdf`.

| Configuração | Acurácia | F1 macro | F1 weighted |
|---|---|---|---|
| ResNet-50 + augmentation | 0,9575 | 0,9610 | 0,9576 |
| ResNet-50 sem augmentation | 0,9575 | 0,9628 | 0,9576 |
| EfficientNet-B0 + augmentation | **0,9646** | **0,9705** | **0,9646** |
| EfficientNet-B0 sem augmentation | 0,9410 | 0,9469 | 0,9409 |

A principal confusão ocorre entre glioma e meningioma. As classes `normal` e `pituitary_tumor` são as mais fáceis.

## Limitações

Execução única por configuração (uma divisão e uma semente), com intervalos de confiança sobrepostos; hiperparâmetros não otimizados; conjunto de teste pequeno e classe `normal` com poucos exemplos (44 no teste). O dataset não traz identificação de paciente, então fatias do mesmo paciente podem cair em partições diferentes mesmo após a remoção de duplicatas. Não há validação externa nem avaliação clínica: os modelos não devem ser usados para diagnóstico.
