# Sistema de Recomendacão por Imagens com YOLOv8 e FAISS

## Descrição do Projeto

Este projeto implementa um sistema de recomendação baseado em imagens utilizando redes neurais profundas e técnicas de busca por similaridade. O objetivo é, a partir de uma imagem de entrada, sugerir imagens similares baseando-se em características visuais como formato, cor e textura, sem depender de informações textuais como preço ou marca.

## Tecnologias Utilizadas
- **YOLOv8** para detecção de objetos e extração de features.
- **FAISS** (Facebook AI Similarity Search) para indexação e busca eficiente de imagens similares.
- **Python** como linguagem principal.
- **Google Colab** para execução do treinamento e testes.

## Estrutura do Projeto
```
├── dataset/                   # Diretório contendo imagens e anotações
│   ├── images/               # Imagens de treinamento e validação
│   ├── labels/               # Arquivos de anotação no formato YOLO
│   └── dataset.yaml          # Arquivo de configuração para treino do modelo
├── models/                    # Modelos treinados salvos
├── notebooks/                 # Códigos em Jupyter Notebook para treino e inferência
├── app/                       # Implementação da API ou interface para recomendação
└── README.md                  # Documento atual
```

## Como Executar o Projeto

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/nome-do-projeto.git
cd nome-do-projeto
```

### 2. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 3. Baixar o Dataset e Treinar a YOLOv8
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Carregar modelo pré-treinado
model.train(data="/content/dataset/dataset.yaml", epochs=50, imgsz=640, batch=16)
```

### 4. Extração de Features e Indexação com FAISS
```python
import faiss
import numpy as np

# Criar um index FAISS
index = faiss.IndexFlatL2(512)  # 512 é o número de features extraídas

# Adicionar embeddings ao index
index.add(embeddings)
```

### 5. Buscar Imagens Similares
```python
distances, indices = index.search(query_embedding, k=5)  # Retorna as 5 imagens mais similares
```

## Resultados e Melhorias Futuras
- Melhorar a qualidade do dataset para aumentar a precisão.
- Testar diferentes modelos de extração de features.
- Implementar uma interface web para busca visual de produtos.



