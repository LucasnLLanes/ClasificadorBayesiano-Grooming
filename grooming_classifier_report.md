# 📊 Informe Técnico: Clasificador de Detección de Grooming
## Naive Bayes con Validación Cruzada

**Autor:** Lucas Nahuel Llanes  
**Materia:** Aprendizaje Automático  
**Docente:** Scudero, Yanina Ximena  
**Institución:** INSTITUTO SUPERIOR DE FORMACIÓN TÉCNICA N° 197  
**Año:** 2026

---

## 🎯 1. OBJETIVO DEL PROYECTO

Desarrollar un **clasificador de machine learning** capaz de identificar mensajes potencialmente peligrosos de **grooming** (acoso sexual infantil en línea) utilizando el algoritmo **Naive Bayes Multinomial**.

### Contexto
El grooming es una problemática seria donde adultos manipulan a menores a través de mensajes aparentemente inocentes que esconden intenciones maliciosas. Este clasificador busca detectar patrones lingüísticos característicos de este comportamiento.

---

## 📁 2. DATASET

| Métrica | Valor |
|---------|-------|
| **Total de mensajes** | 20 |
| **Mensajes de grooming** | 10 (50%) |
| **Mensajes normales** | 10 (50%) |
| **Balance de clases** | ✅ Perfectamente balanceado |
| **Idioma** | Español (Argentina) |
| **Tipo de datos** | Texto conversacional |

### Ejemplos del Dataset

**Mensajes de Grooming:**
- "¿Cuántos años tenés? No le digas a nadie que hablamos."
- "Podes mandarme una foto tuya, pero que sea solo para mí."
- "No le contés a nadie que hablamos por acá, ¿sí?"

**Mensajes Normales:**
- "Hola, ¿cómo estás? ¿Querés jugar algo online?"
- "¿Terminaste la tarea de matemáticas?"
- "¿Querés que te pase los apuntes de biología?"

### Patrones Identificables en Grooming

- 🚫 Solicitudes de secreto ("No le digas a nadie")
- 📸 Peticiones de fotos o información personal
- 🎭 Creación de intimidad falsa ("Sos especial para mí")
- 🔒 Aislamiento del menor de sus padres/tutores
- ⚠️ Preguntas sobre privacidad ("¿Estás solo en casa?")

---

## 🔧 3. METODOLOGÍA

### 3.1 Arquitectura del Modelo
