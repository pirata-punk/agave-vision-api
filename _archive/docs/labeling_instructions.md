# AGAVE VISION — Manual de Etiquetado

## 0) Pre‑work obligatorio (archivos locales + Label Studio)

Label Studio bloquea el servicio de archivos locales si no se habilita. Completa cada paso antes empezar a etiquetar.

### 0.1 Instalar Python y Label Studio (una sola vez)

- Verifica Python: `python3 --version` (≥ 3.10; 3.11 recomendado).
- Si falta o está desactualizado:

  - macOS: `brew install python` o https://www.python.org/downloads/macos/
  - Windows: https://www.python.org/downloads/windows/ <br> _(marca “Add Python to PATH”)._

- Instala Label Studio:
  ```bash
  pip install --upgrade pip
  pip install label-studio
  ```

### 0.2 Verificar carpetas del dataset

- Patrón: `agave-vision-api/data/tiles_round<round_number>/images/`
- Ejemplos: `.../tiles_round1/images/`, `.../tiles_round2/images/`, `.../tiles_round3/images/`, `.../tiles_round4/images/`
- Cada ronda usa su propio proyecto en Label Studio.

### 0.3 Establecer variables de entorno requeridas

La raíz debe ser `tiles_roundN` (NO la carpeta `images`). Ejemplo para ronda 1:

```bash
export LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

export LOCAL_FILES_DOCUMENT_ROOT="/Users/<tu-usuario>/path/agave-vision-api/data/tiles_round1"
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/<tu-usuario>/path/agave-vision-api/data/tiles_round1"
```

Reemplaza `<tu-usuario>` y la ruta completa según corresponda.

### 0.7 Iniciar Label Studio

```bash
label-studio start
```

### 0.8 Conectar archivos locales en la UI

- Tipo de almacenamiento: Local Files.
- Ruta absoluta: `/Users/<tu-usuario>/path/agave-vision-api/data/tiles_round1/images`
- Debe estar **dentro** de la raíz configurada en las variables.
- Si root = `tiles_round1`, entonces:
  - `tiles_round1/images` ✅
  - `tiles_round1` (no es carpeta de imágenes)

---

### Troubleshooting:

### A) Terminar procesos previos de Label Studio

Ejecuta todo; ignora mensajes de “no se encontró proceso”:

```bash
pkill -f label-studio
pkill -f label_studio
pkill -f "label studio"
```

### B) Limpiar variables de entorno en conflicto

```bash
unset LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED
unset LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
unset LOCAL_FILES_SERVING_ENABLED
unset LOCAL_FILES_DOCUMENT_ROOT
```

## 1) Nombres de proyecto (obligatorio)

- Formato: `agave-vision-tiles-round-<round_number>`
- Ejemplos: `agave-vision-tiles-round-1`, `agave-vision-tiles-round-2`, `agave-vision-tiles-round-3`, `agave-vision-tiles-round-4`

## 2) Nombres de dataset (almacenamiento Local Files)

- Formato: `tiles-round-<round_number>` (ej.: `tiles-round-1`, `tiles-round-2`, `tiles-round-3`, `tiles-round-4`)

## 3) Interfaz de etiquetado requerida (XML)

Usa exactamente esto:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="pina" background="green"/>
    <Label value="worker" background="yellow"/>
    <Label value="object" background="red"/>
  </RectangleLabels>
</View>
```

Notas: los nombres de clase deben coincidir (`pina`, `worker`, `object`); conserva los colores.

## 4) Objetivo de anotación

- Entrenar YOLOv8 para detectar: `pina`, `worker`, `object`.
- Exportar bounding boxes y tiles vacíos. Los tiles vacíos deben quedar sin etiqueta.

## 5) Definiciones de clase

- **pina**: Piña de agave reconocible (completa o parcial, cercana o lejana).
- **worker**: Persona con PPE (casco, chaleco, botas, postura de trabajador).
- **object**: Elementos removibles y no estructurales (conos, herramientas, cascos sueltos, mangueras/cables en el piso, llantas no fijadas, tablas, cajas, rocas, escombros grandes).

## 6) Nunca etiquetar

- Paredes de difusor/hopper; rejillas de piso; paredes de concreto; barandales/estructuras metálicas.
- Carrocería del camión o llantas fijas; cables permanentes.
- Sombras; reflejos; manchas de agua; escombros muy pequeños; formas ambiguas.

## 7) Reglas de etiquetado

- **worker — etiqueta si:** PPE identificable; parte superior visible; piernas + chaleco visibles; agachado; parcialmente recortado pero identificable.  
  **No etiquetes:** extremidades no identificables; sombras; reflejos; fragmentos diminutos.
- **pina — etiqueta si:** piñas completas; piñas parciales; piñas lejanas si la forma es clara; piñas superpuestas (una por piña visible).  
  **No etiquetes:** chips; montones indistinguibles; trozos de madera similares.
- **object — etiqueta si:** casco en el piso; conos; herramientas; cables en el piso; llantas sueltas; cajas/tablas/cubetas; escombros grandes.  
  **No etiquetes:** componentes fijos del camión; tuberías estructurales; cables permanentes; polvo/residuos pequeños.

## 8) Manejo de tiles vacíos

Si el tile no tiene `pina`, `worker` ni `object`, déjalo vacío. Los tiles vacíos son negativos obligatorios para entrenamiento.

## 9) Lógica de decisión (flujo)

- ¿Es un trabajador? → `worker`
- Si no: ¿es una piña? → `pina`
- Si no: ¿es un objeto removible? → `object`
- Si no: dejar vacío

## 10) Ejemplos rápidos

| Escenario                         | Clase  |
| --------------------------------- | ------ |
| Trabajador sentado, casco visible | worker |
| Pierna + chaleco visibles         | worker |
| Piñas vista clara                 | pina   |
| Piñas lejanas identificables      | pina   |
| Casco en el piso                  | object |
| Cono en pasillo                   | object |
| Cable sobre el piso               | object |
| Caja de camión, sin objetos       | empty  |
| Fosa negra                        | empty  |
| Placa metálica                    | empty  |
| Sombras o reflejos                | ignore |
