✦Comparación General – Ensayos 1, 2, 3 y 4✦
================================================

➤Evolución de arquitectura, capacidad, activación, regularización y filosofía de diseño

Los cuatro ensayos representan una línea progresiva de experimentación donde se estudia cómo cambia el rendimiento y comportamiento de una CNN al modificar la profundidad, la cantidad de filtros, la estructura del clasificador, la activación y el optimizador.
Los Ensayos 2 y 4 son rediseños más profundos respecto a sus contrapartes (1 y 3), pero cada pareja explora enfoques distintos.

------------------------------------------------------------------------------------------------

💠1. Arquitectura general: de modelos convencionales a arquitecturas minimalistas

Los Ensayos 1 y 2 forman una pareja donde el Ensayo 2 simplifica y refina el modelo previo:
Eliminación de clasificadores densos grandes.
Uso de 4 bloques convolucionales 16→32→32→64 con BatchNorm, Mish y Dropout2d.
Clasificación basada en GAP + Linear, mucho más ligera y estable.

Los Ensayos 3 y 4, en cambio, analizan la arquitectura desde una perspectiva distinta:
El Ensayo 3 usa una arquitectura corta y convencional (32→64).
El Ensayo 4 profundiza mucho más (5 capas) pero con filtros extremadamente pequeños (1→2→4→8→16), llevando el minimalismo al límite.

🔹Diferencia clave entre parejas:

La pareja 1–2 busca estabilidad, ligereza razonable y eficiencia.
La pareja 3–4 busca experimentar con reducción extrema de capacidad y profundidad inusual.

------------------------------------------------------------------------------------------------

💠2. Capacidad convolucional: estrategias opuestas en ambos grupos

🟦 Ensayos 1–2: reducción moderada pero estratégica
El Ensayo 2 mantiene una arquitectura “normal”:
Número de filtros razonable (16–64).
Bloque doble de 32 para mejorar gradientes y detalle fino.
Capacidad suficiente para reconocer gestos complejos.

🟧 Ensayos 3–4: reducción drástica
El Ensayo 4 reformula por completo la capacidad al pasar a:
5 capas muy pequeñas: 1→2→4→8→16.
Representación extremadamente compacta (solo 16 features finales).

🔹Dos filosofías distintas:
Ensayo 2: “ligero pero competente”.
Ensayo 4: “mínimo absoluto para estudiar límites”.

------------------------------------------------------------------------------------------------

💠3. Funciones de activación: comparación de tres enfoques

Los ensayos exploran diferentes activaciones según la arquitectura:

🟦 Ensayo 2
Mish, elegida por su suavidad y mejor propagación del gradiente en modelos pequeños/medianos.

🟧 Ensayo 3
SiLU, una activación suave bien establecida para CNN de tamaño moderado.

🟩 Ensayo 4
GELU, que suele funcionar mejor en redes más profundas gracias a su no linealidad más expresiva.

🔹El contraste global muestra que:
Ensayo 2 prioriza estabilidad y riqueza de representación.
Ensayo 3 mantiene una opción estándar.
Ensayo 4 busca compensar la baja capacidad con una activación más fuerte.

------------------------------------------------------------------------------------------------

💠4. Regularización y compresión espacial

🟦 Ensayo 2
Implementa un enfoque equilibrado:
Dropout2d(0.1) + Dropout(0.3).
GAP a 1×1 tras convoluciones de tamaño razonable.

🟧 Ensayo 3
Utiliza GAP a 2×2, conservando algo más de información espacial.

🟩Ensayo 4
Lleva la compresión al extremo:
GAP a 1×1 a pesar de tener muy pocos filtros.
Esto crea una representación ultra compacta (solo 16 valores).

La pareja 3–4 explora específicamente cuánto detalle puede eliminarse sin destruir rendimiento.
La pareja 1–2 busca una regularización moderada y estable.

------------------------------------------------------------------------------------------------

💠5. Optimizadores: distintas elecciones según la arquitectura

🟦 Ensayo 2 utiliza AdamW, ideal para separar gradiente y regularización.

🟧 Ensayo 3 utiliza RMSprop.

🟩 Ensayo 4 cambia a Adamax, optimizado para gradientes ruidosos y modelos pequeños.

La pareja 3–4 explora si cambiar el optimizador puede estabilizar arquitecturas muy pequeñas.
La pareja 1–2 se centra más en robustez y generalización.

------------------------------------------------------------------------------------------------

💠6. Procesamiento de datos y tamaño de imagen

Solo los Ensayos 1–2 mencionan explícitamente modificaciones de data augmentation y resolución.

Ensayo 2 reduce la resolución a 96×96 y suaviza el augmentation.

La pareja 3–4 no reporta cambios en este aspecto.

🔹En la comparativa general: los Ensayos 1–2 dedican más atención al preprocesamiento como parte del diseño.

❖Conclusión general❖

Los 4 ensayos reflejan dos líneas de investigación paralelas:

🟦 Ensayos 1 y 2
Buscan optimizar una arquitectura razonablemente pequeña, logrando: menos parámetros, mejor estabilidad, mejor generalización, un clasificador mucho más eficiente.
El Ensayo 2 representa una versión pulida, ligera y equilibrada del Ensayo 1.

🟧 Ensayos 3 y 4
Exploran el extremo del minimalismo: filtros mínimos, profundidad máxima para la cantidad de canales, compresión agresiva, activaciones y optimizadores alternativos.
El Ensayo 4 “estresa” el modelo para medir los límites de cuánta capacidad se puede sacrificar manteniendo un comportamiento razonable.
