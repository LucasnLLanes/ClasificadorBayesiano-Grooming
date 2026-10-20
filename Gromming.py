# ============================================================
# TRABAJO PRÁCTICO: Detección de Grooming con Naive Bayes
# Materia: Aprendizaje Automático
# Docente: Scudero, Yanina Ximena
# Alumno: Lucas Nahuel Llanes
# Carrera: Tecnicatura en Ciencia de Datos e Inteligencia Artificial
# Institución: INSTITUTO SUPERIOR DE FORMACIÓN TÉCNICA N° 197
# Año: 2026
# ============================================================

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. Carga del corpus de ejemplo
# ============================================================
data = {
    'text': [
        "¿Cuántos años tenés? No le digas a nadie que hablamos.",
        "Hola, ¿cómo estás? ¿Querés jugar algo online?",
        "Sos muy linda, no le muestres esto a tus papás.",
        "¿Terminaste la tarea de matemáticas?",
        "Podes mandarme una foto tuya, pero que sea solo para mí.",
        "¿Querés que te ayude con el juego que no podías pasar?",
        "No le contés a nadie que hablamos por acá, ¿sí?",
        "¿Te gustó la película de ayer?",
        "¿Estás solo en casa ahora?",
        "¿Querés hacer una videollamada para estudiar juntos?",
        "No le digas a tus padres que te escribí, es nuestro secreto.",
        "¿Cómo te fue en el examen de historia?",
        "Mandame una foto tuya, pero que nadie más la vea.",
        "¿Jugamos Minecraft esta tarde?",
        "Sos muy especial para mí, no le cuentes a nadie que hablamos.",
        "¿Querés que te pase los apuntes de biología?",
        "¿Podes mostrarme cómo estás vestida ahora?",
        "¿Tenés ganas de salir a andar en bici mañana?",
        "No hace falta que le digas a nadie que hablamos tanto.",
        "¿Querés que estudiemos juntos para el parcial?"
    ],
    'label': [
        "grooming", "no grooming", "grooming", "no grooming", "grooming",
        "no grooming", "grooming", "no grooming", "grooming", "no grooming",
        "grooming", "no grooming", "grooming", "no grooming", "grooming",
        "no grooming", "grooming", "no grooming", "grooming", "no grooming"
    ]
}

df = pd.DataFrame(data)
X = df['text']
y = df['label']
print(df)

# ============================================================
# 2. Pipeline: vectorización + clasificador
# ============================================================
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# ============================================================
# 3. Validación cruzada estratificada (5 folds)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print("Precisión en cada fold:", scores)
print("Precisión media:", scores.mean())
print("Desviación estándar:", scores.std())

# ============================================================
# 4. Entrenamiento sobre el conjunto completo
# ============================================================
pipeline.fit(X, y)
model = pipeline.named_steps['classifier']
vectorizer = pipeline.named_steps['vectorizer']

# ============================================================
# 5. Evaluación sobre split train/test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

pipeline_split = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
pipeline_split.fit(X_train, y_train)
y_pred = pipeline_split.predict(X_test)

print("Reporte de clasificación en conjunto de prueba:\n",
      classification_report(y_test, y_pred))
print("Precisión en conjunto de prueba:", accuracy_score(y_test, y_pred))

# ============================================================
# 6. Matriz de confusión (estilo y color personalizados)
# ============================================================
clases = pipeline_split.named_steps['classifier'].classes_
cm = confusion_matrix(y_test, y_pred, labels=clases)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrBr',
    linewidths=1.2,
    linecolor='white',
    cbar_kws={'label': 'Cantidad de mensajes'},
    annot_kws={'size': 14, 'weight': 'bold', 'color': 'black'},
    xticklabels=clases,
    yticklabels=clases
)
plt.title('Matriz de Confusión — Clasificador Naive Bayes', fontsize=12, pad=12)
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta real')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# 7. Predicción sobre mensajes nuevos
# ============================================================
nuevas_noticias = [
    "¿Podés mandarme una foto tuya? No se la muestres a nadie.",
    "¿Jugamos Minecraft esta tarde?",
    "No le digas a tus papás que hablamos, ¿sí?",
    "¿Terminaste el trabajo práctico de historia?"
]

nuevas_noticias_vectorized = vectorizer.transform(nuevas_noticias)
predicciones = model.predict(nuevas_noticias_vectorized)

df_preds = pd.DataFrame({
    'texto': nuevas_noticias,
    'prediccion': predicciones
})
print(df_preds)