import os
import argparse
from PIL import Image
import numpy as np
from scipy import ndimage
import cv2

def cargar_imagen(ruta_imagen):
    """Carga una imagen desde una ruta especificada."""
    try:
        imagen = Image.open(ruta_imagen)
        if imagen.mode != "RGBA" and "A" in imagen.getbands():
            imagen = imagen.convert("RGBA")
        return imagen
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None

def detectar_sprites_horizontal(imagen):
    """Detecta los límites horizontales de cada sprite en una imagen."""
    datos = np.array(imagen)
    altura, anchura = datos.shape[:2]
    
    columnas_no_vacias = []
    inicio_sprite = None
    
    for x in range(anchura):
        columna = datos[:, x, 3]
        if np.any(columna > 0):
            if inicio_sprite is None:
                inicio_sprite = x
        elif inicio_sprite is not None:
            columnas_no_vacias.append((inicio_sprite, x))
            inicio_sprite = None
    
    if inicio_sprite is not None:
        columnas_no_vacias.append((inicio_sprite, anchura))
    
    return columnas_no_vacias

def detectar_sprites_vertical(imagen):
    """Detecta los límites verticales de cada sprite en una imagen."""
    datos = np.array(imagen)
    altura, anchura = datos.shape[:2]
    
    filas_no_vacias = []
    inicio_sprite = None
    
    for y in range(altura):
        fila = datos[y, :, 3]
        if np.any(fila > 0):
            if inicio_sprite is None:
                inicio_sprite = y
        elif inicio_sprite is not None:
            filas_no_vacias.append((inicio_sprite, y))
            inicio_sprite = None
    
    if inicio_sprite is not None:
        filas_no_vacias.append((inicio_sprite, altura))
    
    return filas_no_vacias

def detectar_sprites(imagen):
    """Detecta los límites de cada sprite en una imagen, tanto horizontal como verticalmente."""
    limites_horizontales = detectar_sprites_horizontal(imagen)
    limites_verticales = detectar_sprites_vertical(imagen)
    
    sprites = []
    limites = []
    
    for x_inicio, x_fin in limites_horizontales:
        for y_inicio, y_fin in limites_verticales:
            sprite = imagen.crop((x_inicio, y_inicio, x_fin, y_fin))
            sprites.append(sprite)
            limites.append((x_inicio, y_inicio, x_fin, y_fin))
    
    return sprites, limites

def recortar_sprites(imagen, modo='horizontal'):
    """Recorta los sprites individuales de la imagen."""
    if modo == 'horizontal':
        limites = detectar_sprites_horizontal(imagen)
        sprites = [imagen.crop((x_inicio, 0, x_fin, imagen.height)) for x_inicio, x_fin in limites]
    else:
        limites = detectar_sprites_vertical(imagen)
        sprites = [imagen.crop((0, y_inicio, imagen.width, y_fin)) for y_inicio, y_fin in limites]
    
    return sprites, limites

def dividir_sprites_fijos(imagen, ancho, alto):
    """Divide una imagen en sprites de tamaño fijo."""
    sprites = []
    limites = []
    
    imagen_ancho, imagen_alto = imagen.size
    
    num_sprites_x = imagen_ancho // ancho
    num_sprites_y = imagen_alto // alto
    
    for y in range(num_sprites_y):
        for x in range(num_sprites_x):
            left = x * ancho
            upper = y * alto
            right = left + ancho
            lower = upper + alto
            
            sprite = imagen.crop((left, upper, right, lower))
            sprites.append(sprite)
            limites.append((left, upper, right, lower))
    
    return sprites, limites

def detectar_sprites_por_numero(imagen, num_horizontal, num_vertical):
    """Detecta los límites de cada sprite en una imagen basado en el número de sprites horizontal y verticalmente."""
    imagen_ancho, imagen_alto = imagen.size
    ancho_sprite = imagen_ancho // num_horizontal
    alto_sprite = imagen_alto // num_vertical
    
    sprites = []
    limites = []
    
    for y in range(num_vertical):
        for x in range(num_horizontal):
            left = x * ancho_sprite
            upper = y * alto_sprite
            right = left + ancho_sprite
            lower = upper + alto_sprite
            
            sprite = imagen.crop((left, upper, right, lower))
            sprites.append(sprite)
            limites.append((left, upper, right, lower))
    
    return sprites, limites

def detectar_sprites_por_dimensiones(imagen, ancho_sprite, alto_sprite):
    """Detecta todos los sprites con dimensiones específicas en una imagen."""
    datos = np.array(imagen)
    altura, anchura = datos.shape[:2]
    
    sprites = []
    limites = []
    
    imagen_ancho, imagen_alto = imagen.size
    if (ancho_sprite >= imagen_ancho * 0.9 and alto_sprite >= imagen_alto * 0.9) or (ancho_sprite == imagen_ancho and alto_sprite == imagen_alto):
        sprite = imagen.crop((0, 0, imagen_ancho, imagen_alto))
        sprites.append(sprite)
        limites.append((0, 0, imagen_ancho, imagen_alto))
        return sprites, limites
    
    regiones_incluidas = np.zeros((altura, anchura), dtype=bool)
    
    for y in range(0, altura - alto_sprite + 1, max(1, alto_sprite // 10)):
        for x in range(0, anchura - ancho_sprite + 1, max(1, ancho_sprite // 10)):
            if regiones_incluidas[y:y+alto_sprite, x:x+ancho_sprite].any():
                continue
                
            region = datos[y:y+alto_sprite, x:x+ancho_sprite, 3]
            
            pixeles_no_transparentes = np.sum(region > 0)
            if pixeles_no_transparentes > (ancho_sprite * alto_sprite * 0.05):
                mejor_x, mejor_y = x, y
                
                sprite = imagen.crop((mejor_x, mejor_y, mejor_x + ancho_sprite, mejor_y + alto_sprite))
                sprites.append(sprite)
                limites.append((mejor_x, mejor_y, mejor_x + ancho_sprite, mejor_y + alto_sprite))
                
                regiones_incluidas[mejor_y:mejor_y+alto_sprite, mejor_x:mejor_x+ancho_sprite] = True
    
    return sprites, limites

def detectar_sprites_por_color(imagen, umbral_color=10, min_area=100):
    """Detecta sprites basados en la diferencia de color con el fondo."""
    if imagen.mode == 'RGBA':
        rgb_imagen = np.array(imagen.convert('RGB'))
        alpha = np.array(imagen.split()[3])
        mascara = alpha > 0
    else:
        rgb_imagen = np.array(imagen.convert('RGB'))
        bordes = np.concatenate([
            rgb_imagen[0, :],
            rgb_imagen[-1, :],
            rgb_imagen[:, 0],
            rgb_imagen[:, -1]
        ], axis=0)
        
        color_fondo = np.median(bordes, axis=0).astype(int)
        diferencia = np.sqrt(np.sum((rgb_imagen - color_fondo) ** 2, axis=2))
        mascara = diferencia > umbral_color
    
    etiquetas, num_etiquetas = ndimage.label(mascara)
    
    sprites = []
    limites = []
    
    for i in range(1, num_etiquetas + 1):
        componente = etiquetas == i
        area = np.sum(componente)
        
        if area < min_area:
            continue
        
        y_indices, x_indices = np.where(componente)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        margen = 1
        x_min = max(0, x_min - margen)
        y_min = max(0, y_min - margen)
        x_max = min(imagen.width - 1, x_max + margen)
        y_max = min(imagen.height - 1, y_max + margen)
        
        sprite = imagen.crop((x_min, y_min, x_max + 1, y_max + 1))
        sprites.append(sprite)
        limites.append((x_min, y_min, x_max + 1, y_max + 1))
    
    return sprites, limites

def detectar_sprites_avanzado(imagen, min_area=100, max_area=None):
    """Utiliza OpenCV para detección avanzada de contornos y encuentra sprites incluso en fondos complejos."""
    if imagen.mode == 'RGBA':
        np_img = np.array(imagen)
        alpha = np_img[:, :, 3]
        mascara = alpha > 0
        img_cv = cv2.cvtColor(np.array(imagen.convert('RGB')), cv2.COLOR_RGB2BGR)
    else:
        img_cv = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(gris, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mascara = mascara > 0
    
    mascara = mascara.astype(np.uint8) * 255
    
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sprites = []
    limites = []
    
    altura, anchura = mascara.shape[:2]
    if max_area is None:
        max_area = altura * anchura
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contorno)
        
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(imagen.width, x + w)
        y_max = min(imagen.height, y + h)
        
        sprite = imagen.crop((x_min, y_min, x_max, y_max))
        sprites.append(sprite)
        limites.append((x_min, y_min, x_max, y_max))
    
    return sprites, limites

def guardar_sprites(sprites, directorio_salida, nombre_base):
    """Guarda los sprites recortados como archivos individuales."""
    os.makedirs(directorio_salida, exist_ok=True)
    
    for i, sprite in enumerate(sprites):
        nombre_archivo = f"{nombre_base}_{i+1}.png"
        ruta_completa = os.path.join(directorio_salida, nombre_archivo)
        sprite.save(ruta_completa)
        print(f"Sprite guardado: {ruta_completa}")

def main():
    parser = argparse.ArgumentParser(description="Divide una hoja de sprites en sprites individuales.")
    parser.add_argument("imagen", help="Ruta de la imagen a procesar")
    parser.add_argument("--modo", choices=["horizontal", "vertical", "fijo", "auto", "numero", "dimensiones", "color", "avanzado"], 
                        default="fijo", help="Modo de división")
    parser.add_argument("--salida", default="sprites_recortados",
                        help="Directorio donde guardar los sprites (predeterminado: 'sprites_recortados')")
    parser.add_argument("--nombre", default="sprite",
                        help="Nombre base para los sprites guardados (predeterminado: 'sprite')")
    parser.add_argument("--ancho", type=int, default=0,
                        help="Ancho en píxeles de cada sprite (para modo 'fijo' y 'dimensiones')")
    parser.add_argument("--alto", type=int, default=0,
                        help="Alto en píxeles de cada sprite (para modo 'fijo' y 'dimensiones')")
    parser.add_argument("--num_horizontal", type=int, default=0,
                        help="Número de sprites horizontalmente (para modo 'numero')")
    parser.add_argument("--num_vertical", type=int, default=0,
                        help="Número de sprites verticalmente (para modo 'numero')")
    parser.add_argument("--umbral_color", type=int, default=10,
                        help="Umbral para detección por color (para modo 'color')")
    parser.add_argument("--min_area", type=int, default=100,
                        help="Área mínima para considerar un componente como sprite (para modos 'color' y 'avanzado')")
    parser.add_argument("--max_area", type=int, default=None,
                        help="Área máxima para considerar un componente como sprite (para modo 'avanzado')")
    
    args = parser.parse_args()
    
    imagen = cargar_imagen(args.imagen)
    if imagen:
        if args.modo == "fijo":
            if args.ancho <= 0 or args.alto <= 0:
                print("Error: Para el modo 'fijo', debe proporcionar un ancho y alto válidos (mayores que 0).")
                return
                
            sprites, limites = dividir_sprites_fijos(imagen, args.ancho, args.alto)
            print(f"Se han generado {len(sprites)} sprites con dimensiones {args.ancho}x{args.alto} píxeles.")
            for i, limite in enumerate(limites):
                left, upper, right, lower = limite
                print(f"Sprite {i+1}: Coordenadas (x1={left}, y1={upper}, x2={right}, y2={lower})")
        elif args.modo == "dimensiones":
            if args.ancho <= 0 or args.alto <= 0:
                print("Error: Para el modo 'dimensiones', debe proporcionar un ancho y alto válidos (mayores que 0).")
                return
                
            sprites, limites = detectar_sprites_por_dimensiones(imagen, args.ancho, args.alto)
            print(f"Se han detectado {len(sprites)} sprites con dimensiones {args.ancho}x{args.alto} píxeles.")
            for i, limite in enumerate(limites):
                left, upper, right, lower = limite
                print(f"Sprite {i+1}: Coordenadas (x1={left}, y1={upper}, x2={right}, y2={lower})")
        elif args.modo == "color":
            min_area = args.min_area if args.min_area > 0 else 100
            sprites, limites = detectar_sprites_por_color(imagen, args.umbral_color, min_area)
            print(f"Se han detectado {len(sprites)} sprites usando detección basada en color.")
            for i, limite in enumerate(limites):
                left, upper, right, lower = limite
                print(f"Sprite {i+1}: Coordenadas (x1={left}, y1={upper}, x2={right}, y2={lower})")
        elif args.modo == "avanzado":
            min_area = args.min_area if args.min_area > 0 else 100
            max_area = args.max_area
            sprites, limites = detectar_sprites_avanzado(imagen, min_area, max_area)
            print(f"Se han detectado {len(sprites)} sprites usando detección avanzada.")
            for i, limite in enumerate(limites):
                left, upper, right, lower = limite
                print(f"Sprite {i+1}: Coordenadas (x1={left}, y1={upper}, x2={right}, y2={lower})")
        elif args.modo == "auto":
            sprites, limites = detectar_sprites(imagen)
            print(f"Se han detectado {len(sprites)} sprites.")
            for i, limite in enumerate(limites):
                print(f"Sprite {i+1}: Coordenadas (x1={limite[0]}, y1={limite[1]}, x2={limite[2]}, y2={limite[3]})")
        elif args.modo == "numero":
            if args.num_horizontal <= 0 or args.num_vertical <= 0:
                print("Error: Para el modo 'numero', debe proporcionar un número válido de sprites horizontal y verticalmente (mayores que 0).")
                return
                
            sprites, limites = detectar_sprites_por_numero(imagen, args.num_horizontal, args.num_vertical)
            print(f"Se han generado {len(sprites)} sprites con {args.num_horizontal}x{args.num_vertical} divisiones.")
            for i, limite in enumerate(limites):
                left, upper, right, lower = limite
                print(f"Sprite {i+1}: Coordenadas (x1={left}, y1={upper}, x2={right}, y2={lower})")
        else:
            sprites, limites = recortar_sprites(imagen, args.modo)
            print(f"Se han detectado {len(sprites)} sprites.")
            for i, limite in enumerate(limites):
                if args.modo == "horizontal":
                    print(f"Sprite {i+1}: Desde x={limite[0]} hasta x={limite[1]}")
                else:
                    print(f"Sprite {i+1}: Desde y={limite[0]} hasta y={limite[1]}")
        
        guardar_sprites(sprites, args.salida, args.nombre)
        print("Proceso completado con éxito.")

if __name__ == "__main__":
    main()
