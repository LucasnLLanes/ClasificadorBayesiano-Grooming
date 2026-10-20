# Detección de Grooming con Clasificador Naive Bayes

## Descripción

Este proyecto implementa un clasificador de Naive Bayes para la detección de grooming en mensajes de texto. El grooming se refiere a comportamientos manipuladores en línea dirigidos a menores. El modelo utiliza el algoritmo Multinomial Naive Bayes para clasificar mensajes como "grooming" o "no grooming" basándose en el contenido textual.

El proyecto incluye:
- Preprocesamiento de texto con CountVectorizer
- Entrenamiento del modelo con validación cruzada estratificada
- Evaluación del rendimiento con métricas de clasificación y matriz de confusión
- Predicción en nuevos mensajes

## Autor

- **Lucas Nahuel Llanes**

## Materia

Aprendizaje Automático — Tecnicatura en Ciencia de Datos e Inteligencia Artificial — INSTITUTO SUPERIOR DE FORMACIÓN TÉCNICA N° 197 — Año 2026.

## Docente

- Scudero, Yanina Ximena

## Dataset

El dataset utilizado es un conjunto de ejemplo con 20 mensajes etiquetados manualmente:
- 10 mensajes clasificados como "grooming"
- 10 mensajes clasificados como "no grooming"

Los mensajes incluyen frases comunes en conversaciones en línea que pueden indicar intentos de grooming o interacciones inocentes.

## Requisitos

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/grooming-naive-bayes.git
   cd grooming-naive-bayes# ClasificadorBayesiano-Grooming
