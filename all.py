import os
import argparse
from PIL import Image
import numpy as np
from scipy import ndimage

def quitar_fondo(imagen, color_fondo=None, tolerancia=30):
    """
    Quita el fondo de una imagen convirtiéndolo en transparente.
    Detecta el fondo como la región más grande conectada al borde.
    Preserva todos los píxeles que pertenecen a sprites (regiones más pequeñas).
    
    Args:
        imagen: Imagen PIL a procesar
        color_fondo: Color del fondo a quitar (R, G, B). Si es None, se detecta automáticamente
        tolerancia: Tolerancia para la detección del color (0-255)
    
    Returns:
        Imagen con fondo transparente
    """
    # Convertir a RGBA si no lo está
    if imagen.mode != 'RGBA':
        imagen = imagen.convert('RGBA')
    
    datos = np.array(imagen)
    alto, ancho = datos.shape[:2]
    
    # Si no se especifica color de fondo, detectarlo de las esquinas
    if color_fondo is None:
        # Tomar muestras de las esquinas (más confiable que todo el borde)
        esquinas = [
            datos[0, 0, :3],
            datos[0, -1, :3],
            datos[-1, 0, :3],
            datos[-1, -1, :3]
        ]
        color_fondo = np.median(esquinas, axis=0).astype(int)
        print(f"Color de fondo detectado: RGB{tuple(color_fondo)}")
    
    # Calcular la diferencia entre cada pixel y el color de fondo
    r_diff = (datos[:, :, 0].astype(float) - color_fondo[0]) ** 2
    g_diff = (datos[:, :, 1].astype(float) - color_fondo[1]) ** 2
    b_diff = (datos[:, :, 2].astype(float) - color_fondo[2]) ** 2
    diferencia = np.sqrt(r_diff + g_diff + b_diff)
    
    # Máscara de píxeles que NO coinciden con el fondo (son contenido/sprite)
    es_contenido = diferencia > tolerancia
    
    # Etiquetar regiones conectadas de CONTENIDO (no fondo)
    etiquetas_contenido, num_regiones = ndimage.label(es_contenido)
    
    print(f"Detectadas {num_regiones} regiones de contenido")
    
    # Crear máscara de sprites: incluir todas las regiones de contenido
    # y también los píxeles oscuros/negros que están DENTRO de esas regiones
    mascara_sprites = np.zeros((alto, ancho), dtype=bool)
    
    for i in range(1, num_regiones + 1):
        region = etiquetas_contenido == i
        area = np.sum(region)
        
        # Si la región tiene un área mínima, es un sprite
        if area >= 20:  # Al menos 20 píxeles
            # Encontrar el bounding box de esta región
            filas = np.any(region, axis=1)
            cols = np.any(region, axis=0)
            y1, y2 = np.argmax(filas), len(filas) - np.argmax(filas[::-1])
            x1, x2 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
            
            # Incluir TODOS los píxeles dentro del bounding box que no son fondo puro
            # Esto preserva los trazos negros internos
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # Si el píxel está en la región de contenido O
                    # si está rodeado por contenido (es un trazo interno)
                    if region[y, x]:
                        mascara_sprites[y, x] = True
    
    # Ahora, rellenar huecos dentro de cada sprite usando morfología
    # Esto captura los trazos negros que están rodeados por el sprite
    from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion
    
    # Para cada región de sprite, rellenar los huecos internos
    etiquetas_sprites, num_sprites = ndimage.label(mascara_sprites)
    
    mascara_final = np.zeros((alto, ancho), dtype=bool)
    
    for i in range(1, num_sprites + 1):
        sprite_region = etiquetas_sprites == i
        
        # Dilatar ligeramente, rellenar huecos, y volver al tamaño original
        sprite_dilatado = binary_dilation(sprite_region, iterations=2)
        sprite_relleno = binary_fill_holes(sprite_dilatado)
        
        # Aplicar la máscara rellena pero solo donde había contenido original o está rodeado
        mascara_final |= sprite_relleno
    
    # Erosionar un poco para quitar la dilatación extra pero mantener los huecos rellenos
    # Solo en los bordes externos
    mascara_final_limpia = binary_erosion(mascara_final, iterations=1)
    
    # Combinar: mantener todo lo que está en mascara_final_limpia O en mascara_sprites original
    mascara_final = mascara_final_limpia | mascara_sprites
    
    # Calcular cuántos píxeles se eliminan
    pixeles_eliminados = np.sum(~mascara_final)
    print(f"Píxeles de fondo eliminados: {pixeles_eliminados}")
    print(f"Píxeles de sprite preservados: {np.sum(mascara_final)}")
    
    # Aplicar transparencia
    alpha = datos[:, :, 3].copy()
    alpha[~mascara_final] = 0
    datos[:, :, 3] = alpha
    
    return Image.fromarray(datos, 'RGBA')

def recortar_sprite(sprite):
    """
    Recorta un sprite eliminando los bordes transparentes.
    
    Args:
        sprite: Imagen PIL a recortar
    
    Returns:
        Sprite recortado
    """
    # Convertir a RGBA si no lo está
    if sprite.mode != 'RGBA':
        sprite = sprite.convert('RGBA')
    
    # Obtener el bounding box del contenido no transparente
    bbox = sprite.getbbox()
    
    if bbox:
        return sprite.crop(bbox)
    else:
        return sprite

def detectar_sprites_como_blobs(imagen, min_ancho=5, min_alto=5, min_area=50, filtrar_texto=True):
    """
    Detecta sprites individuales como regiones conectadas (blobs).
    Este método es el más robusto para detectar sprites de cualquier forma y posición.
    
    Args:
        imagen: Imagen PIL a analizar
        min_ancho: Ancho mínimo para considerar un sprite válido
        min_alto: Alto mínimo para considerar un sprite válido
        min_area: Área mínima en pixels para considerar un sprite válido
        filtrar_texto: Si True, intenta filtrar texto pequeño
    
    Returns:
        Lista de sprites detectados
    """
    if imagen.mode != 'RGBA':
        imagen = imagen.convert('RGBA')
    
    datos = np.array(imagen)
    alpha = datos[:, :, 3]
    
    # Crear máscara binaria de contenido
    mascara = alpha > 20
    
    # Etiquetar regiones conectadas (blobs)
    etiquetas, num_sprites = ndimage.label(mascara)
    
    print(f"\nDetectando sprites como blobs conectados...")
    print(f"Encontrados {num_sprites} blobs en la imagen")
    
    # Analizar cada blob
    sprites_info = []
    
    for i in range(1, num_sprites + 1):
        # Obtener máscara de este blob
        blob_mask = etiquetas == i
        
        # Encontrar bounding box del blob
        filas = np.any(blob_mask, axis=1)
        cols = np.any(blob_mask, axis=0)
        
        if not np.any(filas) or not np.any(cols):
            continue
        
        y1 = np.argmax(filas)
        y2 = len(filas) - np.argmax(filas[::-1])
        x1 = np.argmax(cols)
        x2 = len(cols) - np.argmax(cols[::-1])
        
        ancho = x2 - x1
        alto = y2 - y1
        area = np.sum(blob_mask)
        
        # Calcular ratio y densidad para clasificar el blob
        ratio = ancho / alto if alto > 0 else 0
        densidad = area / (ancho * alto) if (ancho * alto) > 0 else 0
        
        # Filtros
        if ancho < min_ancho or alto < min_alto or area < min_area:
            print(f"  ⊗ Blob {i}: {ancho}x{alto}px, área={area}, descartado (muy pequeño)")
            continue
        
        # Filtrar texto/números (generalmente tienen ratio alto o muy bajo y baja densidad)
        if filtrar_texto:
            # El texto suele ser muy alto y estrecho o muy ancho y bajo
            if (ratio > 3 or ratio < 0.3) and area < 200:
                print(f"  ⊗ Blob {i}: {ancho}x{alto}px, ratio={ratio:.2f}, descartado (posible texto)")
                continue
            
            # Texto también suele tener baja densidad (muchos espacios)
            if densidad < 0.3 and area < 300:
                print(f"  ⊗ Blob {i}: {ancho}x{alto}px, densidad={densidad:.2f}, descartado (baja densidad)")
                continue
        
        sprites_info.append({
            'id': i,
            'bbox': (x1, y1, x2, y2),
            'ancho': ancho,
            'alto': alto,
            'area': area,
            'densidad': densidad,
            'ratio': ratio,
            'y1': y1  # Para ordenar verticalmente
        })
    
    # Ordenar sprites por posición vertical (de arriba a abajo)
    sprites_info.sort(key=lambda s: s['y1'])
    
    print(f"\n{len(sprites_info)} sprites válidos detectados")
    
    # Extraer sprites
    sprites = []
    for info in sprites_info:
        x1, y1, x2, y2 = info['bbox']
        sprite = imagen.crop((x1, y1, x2, y2))
        sprites.append(sprite)
        print(f"  ✓ Sprite {len(sprites)}: {info['ancho']}x{info['alto']}px, área={info['area']}, pos=({x1},{y1})")
    
    return sprites

def separar_sprites_en_region(imagen, inicio_x, fin_x, min_ancho=8, min_alto=8):
    """
    Intenta separar múltiples sprites dentro de una región analizando patrones.
    """
    if imagen.mode != 'RGBA':
        imagen = imagen.convert('RGBA')
    
    datos = np.array(imagen)
    alpha = datos[:, :, 3]
    contenido = alpha > 20
    
    # Extraer solo la región especificada
    region = contenido[:, inicio_x:fin_x]
    ancho_region = fin_x - inicio_x
    
    # Si la región es muy estrecha, no intentar separar
    if ancho_region < min_ancho * 2:
        return [(inicio_x, fin_x)]
    
    # Analizar la proyección horizontal dentro de la región
    proyeccion = np.sum(region, axis=0)
    
    # Calcular el ancho promedio esperado de un sprite
    # Buscar patrones de repetición o valles en la densidad
    densidad_media = np.mean(proyeccion[proyeccion > 0]) if np.any(proyeccion > 0) else 1
    densidad_max = np.max(proyeccion) if np.any(proyeccion > 0) else 1
    densidad_umbral = densidad_media * 0.5  # 50% de la densidad media
    
    # Buscar columnas con baja densidad (potenciales separadores)
    columnas_bajas = proyeccion < densidad_umbral
    
    # Encontrar secuencias de columnas bajas
    cambios = np.diff(np.concatenate(([False], columnas_bajas, [False])).astype(int))
    inicios_gap = np.where(cambios == 1)[0]
    finales_gap = np.where(cambios == -1)[0]
    
    # Filtrar gaps muy pequeños (ruido)
    gaps_validos = []
    for i_gap, f_gap in zip(inicios_gap, finales_gap):
        if f_gap - i_gap >= 1:  # Al menos 1 columna de gap
            gaps_validos.append((i_gap + inicio_x, f_gap + inicio_x))
    
    if len(gaps_validos) == 0:
        return [(inicio_x, fin_x)]
    
    # Crear sub-regiones basadas en los gaps
    sub_regiones = []
    inicio_actual = inicio_x
    
    for gap_inicio, gap_fin in gaps_validos:
        if gap_inicio - inicio_actual >= min_ancho:
            sub_regiones.append((inicio_actual, gap_inicio))
        inicio_actual = gap_fin
    
    # Agregar la última región
    if fin_x - inicio_actual >= min_ancho:
        sub_regiones.append((inicio_actual, fin_x))
    
    return sub_regiones if len(sub_regiones) > 0 else [(inicio_x, fin_x)]

def detectar_sprites_automaticamente(imagen, min_ancho=8, min_alto=8, gap_minimo=2):
    """
    Detecta sprites automáticamente basándose en el contenido de la imagen.
    Busca columnas vacías para separar sprites.
    
    Args:
        imagen: Imagen PIL a analizar
        min_ancho: Ancho mínimo para considerar un sprite válido
        min_alto: Alto mínimo para considerar un sprite válido
        gap_minimo: Mínimo de columnas vacías entre sprites para separarlos
    
    Returns:
        Lista de sprites detectados
    """
    # Convertir a RGBA
    if imagen.mode != 'RGBA':
        imagen = imagen.convert('RGBA')
    
    datos = np.array(imagen)
    alpha = datos[:, :, 3]
    
    # Crear máscara de contenido (donde alpha > umbral para evitar ruido)
    contenido = alpha > 20  # Umbral para considerar un pixel como contenido
    
    # Proyectar horizontalmente (sumar columnas)
    proyeccion_horizontal = np.sum(contenido, axis=0)
    
    # Detectar columnas vacías (o casi vacías)
    columnas_vacias = proyeccion_horizontal <= 2  # Permitir hasta 2 pixels de ruido
    
    print(f"\nDetectando sprites automáticamente...")
    print(f"Analizando imagen de {imagen.size[0]}x{imagen.size[1]} pixels")
    print(f"Proyección horizontal: min={proyeccion_horizontal.min()}, max={proyeccion_horizontal.max()}")
    
    # Buscar gaps (secuencias de columnas vacías)
    # Un gap debe tener al menos gap_minimo columnas consecutivas vacías
    en_gap = False
    inicio_gap = 0
    gaps = []
    
    for i, vacia in enumerate(columnas_vacias):
        if vacia and not en_gap:
            # Inicio de un gap
            inicio_gap = i
            en_gap = True
        elif not vacia and en_gap:
            # Fin de un gap
            longitud_gap = i - inicio_gap
            if longitud_gap >= gap_minimo:
                gaps.append((inicio_gap, i))
            en_gap = False
    
    print(f"Encontrados {len(gaps)} gaps significativos")
    
    # Si no hay gaps, intentar dividir por densidad mínima
    if len(gaps) == 0:
        print("No se encontraron gaps claros, usando análisis de densidad...")
        # Buscar mínimos locales en la proyección
        from scipy import signal
        if len(proyeccion_horizontal) > 10:
            # Suavizar la señal
            kernel_size = max(3, len(proyeccion_horizontal) // 20)
            kernel = np.ones(kernel_size) / kernel_size
            proyeccion_suave = np.convolve(proyeccion_horizontal, kernel, mode='same')
            
            # Encontrar mínimos locales
            minimos = signal.argrelextrema(proyeccion_suave, np.less, order=3)[0]
            
            # Filtrar mínimos que sean suficientemente bajos
            umbral = np.percentile(proyeccion_suave, 25)  # 25% más bajo
            minimos_significativos = [m for m in minimos if proyeccion_suave[m] < umbral]
            
            print(f"Encontrados {len(minimos_significativos)} mínimos locales significativos")
            
            # Crear gaps artificiales en los mínimos
            gaps = [(m-1, m+1) for m in minimos_significativos]
    
    # Usar los gaps para definir los límites de los sprites
    if len(gaps) == 0:
        # Si todavía no hay gaps, devolver la imagen completa como un sprite
        print("⚠ No se pudieron detectar separaciones, devolviendo imagen completa")
        return [imagen]
    
    # Crear lista de límites de sprites basados en los gaps
    limites_sprites_inicial = []
    inicio_sprite = 0
    
    for gap_inicio, gap_fin in gaps:
        if gap_inicio - inicio_sprite >= min_ancho:
            limites_sprites_inicial.append((inicio_sprite, gap_inicio))
        inicio_sprite = gap_fin
    
    # Agregar el último sprite
    if imagen.size[0] - inicio_sprite >= min_ancho:
        limites_sprites_inicial.append((inicio_sprite, imagen.size[0]))
    
    print(f"Detectadas {len(limites_sprites_inicial)} regiones iniciales")
    
    # FASE 2: Intentar subdividir regiones grandes que puedan contener múltiples sprites
    limites_sprites = []
    for inicio, fin in limites_sprites_inicial:
        ancho = fin - inicio
        # Si una región es muy ancha, intentar subdividirla
        if ancho > min_ancho * 1.8:  # Si es más de 1.8 veces el ancho mínimo (más agresivo)
            print(f"  Analizando región grande: X={inicio}->{fin} ({ancho}px)")
            sub_regiones = separar_sprites_en_region(imagen, inicio, fin, min_ancho, min_alto)
            if len(sub_regiones) > 1:
                print(f"    → Subdividida en {len(sub_regiones)} sprites")
                limites_sprites.extend(sub_regiones)
            else:
                limites_sprites.append((inicio, fin))
        else:
            limites_sprites.append((inicio, fin))
    
    print(f"\nTotal de sprites después del análisis: {len(limites_sprites)}")
    
    # Extraer sprites
    sprites = []
    for i, (inicio, final) in enumerate(limites_sprites, 1):
        ancho = final - inicio
        
        # Extraer la región
        region = contenido[:, inicio:final]
        
        # Encontrar límites verticales
        proyeccion_vertical = np.sum(region, axis=1)
        filas_con_contenido = proyeccion_vertical > 0
        
        if not np.any(filas_con_contenido):
            print(f"  - Sprite {i} descartado (sin contenido)")
            continue
        
        fila_inicio = np.argmax(filas_con_contenido)
        fila_final = len(filas_con_contenido) - np.argmax(filas_con_contenido[::-1])
        
        alto = fila_final - fila_inicio
        
        if alto < min_alto or ancho < min_ancho:
            print(f"  - Sprite {i} descartado (dimensiones: {ancho}x{alto}px)")
            continue
        
        # Extraer el sprite
        sprite = imagen.crop((inicio, fila_inicio, final, fila_final))
        sprites.append(sprite)
        print(f"  ✓ Sprite {len(sprites)}: X={inicio:3d}->{final:3d} ({ancho:2d}px), Y={fila_inicio:2d}->{fila_final:2d} ({alto:2d}px)")
    
    return sprites

def dividir_imagen(imagen, num_horizontal=6, num_vertical=1, recortar=True):
    """
    Divide una imagen en sprites individuales uniformemente.
    
    Args:
        imagen: Imagen PIL a dividir
        num_horizontal: Número de sprites horizontalmente
        num_vertical: Número de sprites verticalmente
        recortar: Si True, recorta los bordes transparentes de cada sprite
    
    Returns:
        Lista de sprites (imágenes PIL)
    """
    ancho_imagen, alto_imagen = imagen.size
    ancho_sprite = ancho_imagen // num_horizontal
    alto_sprite = alto_imagen // num_vertical
    
    sprites = []
    
    for y in range(num_vertical):
        for x in range(num_horizontal):
            left = x * ancho_sprite
            upper = y * alto_sprite
            right = left + ancho_sprite
            lower = upper + alto_sprite
            
            sprite = imagen.crop((left, upper, right, lower))
            
            # Recortar bordes transparentes si se solicita
            if recortar:
                sprite = recortar_sprite(sprite)
            
            sprites.append(sprite)
    
    return sprites

def detectar_por_cuadricula_inteligente(imagen, num_sprites_total=None, num_horizontal=None, num_vertical=None, recortar=True):
    """
    Detecta sprites usando una cuadrícula inteligente que se ajusta al contenido.
    Divide la imagen en regiones y detecta el contenido dentro de cada celda.
    
    Args:
        imagen: Imagen PIL a dividir
        num_sprites_total: Número total de sprites esperados (opcional)
        num_horizontal: Número de sprites por fila (requerido si se especifica num_vertical)
        num_vertical: Número de filas (requerido si se especifica num_horizontal)
        recortar: Si True, recorta los bordes transparentes
    
    Returns:
        Lista de sprites detectados
    """
    if imagen.mode != 'RGBA':
        imagen = imagen.convert('RGBA')
    
    datos = np.array(imagen)
    alpha = datos[:, :, 3]
    contenido = alpha > 20
    
    ancho_imagen, alto_imagen = imagen.size
    
    # Si se especificó num_horizontal y num_vertical, usar cuadrícula uniforme pero detectando contenido
    if num_horizontal and num_vertical:
        print(f"Usando cuadrícula de {num_horizontal}x{num_vertical}")
        ancho_celda = ancho_imagen / num_horizontal
        alto_celda = alto_imagen / num_vertical
        
        sprites = []
        
        for fila in range(num_vertical):
            for col in range(num_horizontal):
                # Calcular límites de la celda
                x1 = int(col * ancho_celda)
                x2 = int((col + 1) * ancho_celda)
                y1 = int(fila * alto_celda)
                y2 = int((fila + 1) * alto_celda)
                
                # Extraer región de contenido en esta celda
                region_contenido = contenido[y1:y2, x1:x2]
                
                if not np.any(region_contenido):
                    print(f"  ⊗ Celda [{fila},{col}]: vacía, omitida")
                    continue
                
                # Encontrar el bounding box del contenido en esta celda
                filas_con_contenido = np.any(region_contenido, axis=1)
                cols_con_contenido = np.any(region_contenido, axis=0)
                
                if not np.any(filas_con_contenido) or not np.any(cols_con_contenido):
                    print(f"  ⊗ Celda [{fila},{col}]: sin contenido válido")
                    continue
                
                fila_inicio = np.argmax(filas_con_contenido) + y1
                fila_fin = y2 - np.argmax(filas_con_contenido[::-1])
                col_inicio = np.argmax(cols_con_contenido) + x1
                col_fin = x2 - np.argmax(cols_con_contenido[::-1])
                
                # Extraer sprite
                sprite = imagen.crop((col_inicio, fila_inicio, col_fin, fila_fin))
                
                if recortar:
                    sprite = recortar_sprite(sprite)
                
                sprites.append(sprite)
                ancho_s = col_fin - col_inicio
                alto_s = fila_fin - fila_inicio
                print(f"  ✓ Celda [{fila},{col}]: {ancho_s}x{alto_s}px en posición ({col_inicio},{fila_inicio})")
        
        return sprites
    
    # Si solo se especificó el total, intentar calcular la mejor cuadrícula
    elif num_sprites_total:
        print(f"Detectando cuadrícula óptima para {num_sprites_total} sprites...")
        
        # Probar diferentes configuraciones de cuadrícula
        mejor_config = None
        for h in range(1, num_sprites_total + 1):
            if num_sprites_total % h == 0:
                v = num_sprites_total // h
                # Calcular qué tan bien se ajusta esta configuración
                ratio_imagen = ancho_imagen / alto_imagen
                ratio_config = (ancho_imagen / h) / (alto_imagen / v)
                
                if mejor_config is None or abs(ratio_config - 1) < abs(mejor_config[2] - 1):
                    mejor_config = (h, v, ratio_config)
        
        if mejor_config:
            num_horizontal, num_vertical = mejor_config[0], mejor_config[1]
            print(f"Mejor configuración detectada: {num_horizontal}x{num_vertical}")
            return detectar_por_cuadricula_inteligente(imagen, None, num_horizontal, num_vertical, recortar)
    
    return []

def procesar_imagen_completo(ruta_entrada, directorio_salida="output", 
                             color_fondo=None, tolerancia=30, 
                             num_horizontal=None, num_vertical=None,
                             nombre_base="sprite", recortar=True, auto_detectar=True,
                             gap_minimo=2, min_ancho=8):
    """
    Procesa una imagen: quita el fondo y la divide en sprites.
    
    Args:
        ruta_entrada: Ruta de la imagen a procesar
        directorio_salida: Directorio donde guardar los resultados
        color_fondo: Color del fondo a quitar (R, G, B) o None para autodetección
        tolerancia: Tolerancia para la detección del color
        num_horizontal: Número de sprites horizontalmente (None para detección automática)
        num_vertical: Número de sprites verticalmente (None para detección automática)
        nombre_base: Nombre base para los sprites guardados
        recortar: Si True, recorta los bordes transparentes de cada sprite
        auto_detectar: Si True, detecta automáticamente los sprites por contenido
    """
    try:
        # Cargar imagen
        print(f"Cargando imagen: {ruta_entrada}")
        imagen = Image.open(ruta_entrada)
        print(f"Tamaño original: {imagen.size[0]}x{imagen.size[1]} pixels")
        
        # Paso 1: Quitar el fondo
        print("\n" + "="*60)
        print("PASO 1: Quitando fondo...")
        print("="*60)
        imagen_sin_fondo = quitar_fondo(imagen, color_fondo, tolerancia)
        
        # Crear directorio de salida si no existe
        os.makedirs(directorio_salida, exist_ok=True)
        
        # Guardar imagen sin fondo
        ruta_sin_fondo = os.path.join(directorio_salida, "imagen_sin_fondo.png")
        imagen_sin_fondo.save(ruta_sin_fondo)
        print(f"✓ Imagen sin fondo guardada: {ruta_sin_fondo}")
        
        # Paso 2: Dividir en sprites
        print("\n" + "="*60)
        print("PASO 2: Dividiendo en sprites...")
        print("="*60)
        
        if num_horizontal or num_vertical:
            # Modo cuadrícula inteligente (con dimensiones especificadas)
            if num_horizontal and num_vertical:
                print(f"Modo: Cuadrícula inteligente {num_horizontal}x{num_vertical}")
            else:
                print(f"Modo: Cuadrícula semi-inteligente")
            sprites = detectar_por_cuadricula_inteligente(imagen_sin_fondo, None, num_horizontal, num_vertical, recortar)
        elif auto_detectar:
            print("Modo: Detección automática por BLOBS (regiones conectadas)")
            sprites = detectar_sprites_como_blobs(imagen_sin_fondo, min_ancho=min_ancho, min_alto=8, min_area=50)
        else:
            # Modo división simple uniforme (fallback)
            num_h = 6 if num_horizontal is None else num_horizontal
            num_v = 1 if num_vertical is None else num_vertical
            print(f"Modo: División uniforme simple {num_h}x{num_v}")
            sprites = dividir_imagen(imagen_sin_fondo, num_h, num_v, recortar)
        
        # Guardar sprites individuales
        print("\n" + "="*60)
        print("PASO 3: Guardando sprites...")
        print("="*60)
        for i, sprite in enumerate(sprites, 1):
            ruta_sprite = os.path.join(directorio_salida, f"{nombre_base}_{i:02d}.png")
            sprite.save(ruta_sprite)
            print(f"  ✓ Sprite {i:2d}: {sprite.size[0]:3d}x{sprite.size[1]:3d}px -> {os.path.basename(ruta_sprite)}")
        
        print(f"\n{'='*60}")
        print(f"✓ ¡COMPLETADO! {len(sprites)} sprites guardados en '{directorio_salida}'")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Quita el fondo de una imagen y la divide en sprites individuales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Detección automática inteligente por blobs (RECOMENDADO para sprites dispersos)
  python all.py imagen.png
  python all.py imagen.png --tolerancia 50
  
  # Cuadrícula inteligente - cuando los sprites están en una cuadrícula uniforme
  python all.py imagen.png --horizontal 6 --vertical 1
  python all.py imagen.png --horizontal 4 --vertical 2
  
  # Con parámetros personalizados
  python all.py imagen.png --min-ancho 10 --tolerancia 50
  python all.py imagen.png --color 90 90 90 --salida mis_sprites
        """
    )
    parser.add_argument("imagen", help="Ruta de la imagen a procesar")
    parser.add_argument("--salida", default="output",
                        help="Directorio de salida (por defecto: 'output')")
    parser.add_argument("--nombre", default="sprite",
                        help="Nombre base para los sprites (por defecto: 'sprite')")
    parser.add_argument("--manual", action="store_true",
                        help="Usar división uniforme simple (no inteligente)")
    parser.add_argument("--horizontal", type=int, default=None, dest="num_horizontal",
                        help="Número de columnas de sprites (ej: 6 sprites en horizontal)")
    parser.add_argument("--vertical", type=int, default=None, dest="num_vertical",
                        help="Número de filas de sprites (ej: 2 filas de sprites)")
    parser.add_argument("--tolerancia", type=int, default=30,
                        help="Tolerancia para detección de fondo 0-255 (por defecto: 30)")
    parser.add_argument("--color", nargs=3, type=int, metavar=("R", "G", "B"),
                        help="Color RGB del fondo a quitar (ej: --color 128 128 128)")
    parser.add_argument("--no-recortar", action="store_true",
                        help="No recortar los bordes transparentes de cada sprite")
    parser.add_argument("--gap", type=int, default=2,
                        help="Mínimo de columnas vacías entre sprites (por defecto: 2)")
    parser.add_argument("--min-ancho", type=int, default=8,
                        help="Ancho mínimo de un sprite en pixels (por defecto: 8)")
    
    args = parser.parse_args()
    
    procesar_imagen_completo(
        args.imagen,
        args.salida,
        tuple(args.color) if args.color else None,
        args.tolerancia,
        args.num_horizontal,
        args.num_vertical,
        args.nombre,
        not args.no_recortar,
        not args.manual and not (args.num_horizontal or args.num_vertical),  # auto_detectar solo si no hay dimensiones
        args.gap,
        args.min_ancho
    )

if __name__ == "__main__":
    main()
