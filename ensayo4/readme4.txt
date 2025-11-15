🧪 Ensayo 4 – Qué cambia respecto al Ensayo 3 y por qué es importante

El Ensayo 4 representa una variación más profunda y minimalista del modelo usado en el Ensayo 3, y su propósito principal es estudiar cómo cambia el comportamiento de la red cuando:

- Se aumentan las capas convolucionales.
- Se reduce drásticamente el número de filtros.
- Se utiliza una activación distinta (GELU en lugar de SiLU).
- Se comprime más la representación (GAP a 1×1).
- Se cambia el optimizador (Adamax en lugar de RMSprop**).

A continuación se explica qué aporta cada uno de estos cambios cuando comparamos directamente Ensayo 4 vs Ensayo 3.

🔍 1. Profundidad: 5 capas vs 2 capas

- Ensayo 3: 2 capas (32 y 64 filtros).
- Ensayo 4: 5 capas (1→2→4→8→16 filtros).

El Ensayo 4 explora si más profundidad, incluso con filtros muy pequeños, puede capturar mejor patrones jerárquicos.

➡️ Hipótesis a probar:
“Una red más profunda aunque con menos filtros puede aprender mejor que una red corta con filtros más anchos.”

🔍 2. Número de filtros: crecimiento mínimo vs convencional

- Ensayo 3: 32 → 64 (convencional).
- Ensayo 4: 1 → 2 → 4 → 8 → 16 (minimalista extremo).

Aquí el Ensayo 4 lleva al límite la idea de “menos es más”:

✔ Menos parámetros.
✔ Menos memoria.
✔ Menos riesgo de sobreajuste.

Pero a costa de una capacidad representacional mucho menor.

➡️ Lo que compara Ensayo 4:
¿Una red muy ligera puede competir en rendimiento con la arquitectura base?

🔍 3. Activación: GELU vs SiLU

- Ensayo 3 usa SiLU (suave, derivada estable).
- Ensayo 4 usa GELU (más expresiva en redes profundas).

GELU tiende a funcionar mejor cuando hay muchas capas, porque:

✔ permite flujos de gradiente más adaptativos,
✔ introduce una no linealidad más rica que SiLU.

➡️ El Ensayo 4 prueba:
Si la activación GELU compensa la baja cantidad de filtros gracias a su mayor capacidad expresiva.

🔍 4. Pooling y compresión espacial

- Ensayo 3: GAP a (2×2).
- Ensayo 4: GAP a (1×1).

El Ensayo 4 comprime la imagen hasta el punto máximo, convirtiendo todo el mapa en un único valor por canal.

✔ Representación súper compacta.
✔ Muy pocas características entran al clasificador.

Pero esto implica:

⚠ Se pierde información espacial fina.
⚠ El clasificador recibe un vector más pobre (solo 16 valores).

➡️ Ensayo 4 evalúa:
¿Qué tan lejos se puede llevar la compresión sin destruir el rendimiento?

🔍 5. Optimizador: Adamax vs RMSprop

- Ensayo 3 usa RMSprop.
- Ensayo 4 usa Adamax.

Adamax funciona especialmente bien con:
✔ modelos pequeños,
✔ gradientes ruidosos,
✔ LazyModules (como en esta arquitectura).

➡️ La pregunta del Ensayo 4 es:
¿Puede Adamax estabilizar la convergencia de una red muy pequeña y profunda donde RMSprop quizá no sea óptimo?

🎯 Conclusión centrada en Ensayo 4

El Ensayo 4 no pretende ser una red mejor que la del Ensayo 3.
Su rol es experimental: estresar el concepto de CNN minimalista para medir los límites de:
- capacidad con muy pocos filtros,
- profundidad extrema en redes ligeras,
- compresión agresiva de características,
- diferentes activaciones y optimizadores en un entorno reducido.
