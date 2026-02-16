import os
import argparse
from PIL import Image

def cargar_imagen(ruta_imagen):
    """Carga una imagen desde una ruta especificada."""
    try:
        imagen = Image.open(ruta_imagen)
        return imagen
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None

def aplicar_fondo(imagen_frente, imagen_fondo, modo_ajuste="estirar"):
    """
    Aplica una imagen de fondo a una imagen con transparencia.
    
    Modos de ajuste:
    - estirar: Estira el fondo para que coincida con las dimensiones de la imagen frontal
    - centrar: Centra el fondo sin modificar su tamaño
    - mosaico: Repite el fondo en mosaico para cubrir toda la imagen
    - escalar: Escala el fondo proporcionalmente para cubrir la imagen
    """
    # Asegurarse de que la imagen frontal tiene canal alpha
    if imagen_frente.mode != 'RGBA':
        imagen_frente = imagen_frente.convert('RGBA')
    
    ancho_frente, alto_frente = imagen_frente.size
    
    # Preparar el fondo según el modo de ajuste
    if modo_ajuste == "estirar":
        fondo_ajustado = imagen_fondo.resize((ancho_frente, alto_frente))
    
    elif modo_ajuste == "centrar":
        # Crear un nuevo lienzo del tamaño de la imagen frontal
        fondo_ajustado = Image.new('RGBA', (ancho_frente, alto_frente), (0, 0, 0, 0))
        ancho_fondo, alto_fondo = imagen_fondo.size
        
        # Calcular la posición para centrar
        x = (ancho_frente - ancho_fondo) // 2
        y = (alto_frente - alto_fondo) // 2
        
        # Pegar el fondo en el centro
        fondo_ajustado.paste(imagen_fondo, (max(0, x), max(0, y)))
    
    elif modo_ajuste == "mosaico":
        fondo_ajustado = Image.new('RGBA', (ancho_frente, alto_frente), (0, 0, 0, 0))
        ancho_fondo, alto_fondo = imagen_fondo.size
        
        # Repetir el fondo en mosaico
        for y in range(0, alto_frente, alto_fondo):
            for x in range(0, ancho_frente, ancho_fondo):
                fondo_ajustado.paste(imagen_fondo, (x, y))
    
    elif modo_ajuste == "escalar":
        ancho_fondo, alto_fondo = imagen_fondo.size
        
        # Calcular la relación de aspecto
        relacion_frente = ancho_frente / alto_frente
        relacion_fondo = ancho_fondo / alto_fondo
        
        if relacion_fondo > relacion_frente:
            # El fondo es más ancho proporcionalmente
            nuevo_alto = alto_frente
            nuevo_ancho = int(nuevo_alto * relacion_fondo)
        else:
            # El fondo es más alto proporcionalmente
            nuevo_ancho = ancho_frente
            nuevo_alto = int(nuevo_ancho / relacion_fondo)
        
        # Redimensionar el fondo manteniendo la proporción
        fondo_redimensionado = imagen_fondo.resize((nuevo_ancho, nuevo_alto))
        
        # Recortar el centro
        left = (nuevo_ancho - ancho_frente) // 2
        top = (nuevo_alto - alto_frente) // 2
        right = left + ancho_frente
        bottom = top + alto_frente
        
        fondo_ajustado = fondo_redimensionado.crop((left, top, right, bottom))
    
    # Combinar las imágenes
    resultado = Image.new('RGBA', (ancho_frente, alto_frente), (0, 0, 0, 0))
    resultado.paste(fondo_ajustado, (0, 0))
    resultado.paste(imagen_frente, (0, 0), imagen_frente)
    
    return resultado

def recortar_imagen(imagen, ancho, alto, modo_recorte="centro"):
    """
    Recorta una imagen al tamaño especificado.
    
    Modos de recorte:
    - centro: Recorta desde el centro de la imagen
    - superior_izquierda: Recorta desde la esquina superior izquierda
    - ajustar: Redimensiona la imagen para ajustarse a las dimensiones especificadas
    """
    if ancho <= 0 or alto <= 0:
        return imagen  # No recortar si no se especifican dimensiones válidas
    
    ancho_actual, alto_actual = imagen.size
    
    if modo_recorte == "ajustar":
        return imagen.resize((ancho, alto))
    
    elif modo_recorte == "superior_izquierda":
        return imagen.crop((0, 0, min(ancho, ancho_actual), min(alto, alto_actual)))
    
    else:  # centro por defecto
        # Calcular coordenadas para recortar desde el centro
        left = max(0, (ancho_actual - ancho) // 2)
        top = max(0, (alto_actual - alto) // 2)
        right = min(ancho_actual, left + ancho)
        bottom = min(alto_actual, top + alto)
        
        return imagen.crop((left, top, right, bottom))

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

def procesar_imagenes(ruta_frente, ruta_fondo, directorio_salida, prefijo="fondo_", 
                      modo_ajuste="estirar", num_horizontal=0, num_vertical=0):
    """Procesa una o más imágenes aplicándoles un fondo común y detectando el número de sprites."""
    # Cargar la imagen de fondo
    imagen_fondo = cargar_imagen(ruta_fondo)
    if not imagen_fondo:
        return
    
    # Asegurar que existe el directorio de salida
    os.makedirs(directorio_salida, exist_ok=True)
    
    # Determinar si la ruta_frente es un directorio o un archivo
    if os.path.isdir(ruta_frente):
        # Procesar todas las imágenes en el directorio
        archivos = [f for f in os.listdir(ruta_frente) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        if not archivos:
            print(f"No se encontraron imágenes en el directorio: {ruta_frente}")
            return
        
        for archivo in archivos:
            ruta_completa = os.path.join(ruta_frente, archivo)
            imagen_frente = cargar_imagen(ruta_completa)
            
            if imagen_frente:
                # Detectar sprites por número si se especificaron
                if num_horizontal > 0 and num_vertical > 0:
                    sprites, limites = detectar_sprites_por_numero(imagen_frente, num_horizontal, num_vertical)
                    for i, sprite in enumerate(sprites):
                        # Aplicar fondo a cada sprite
                        resultado = aplicar_fondo(sprite, imagen_fondo, modo_ajuste)
                        nombre_salida = f"{prefijo}{archivo.split('.')[0]}_{i+1}.png"
                        ruta_salida = os.path.join(directorio_salida, nombre_salida)
                        resultado.save(ruta_salida)
                        print(f"Sprite procesado: {ruta_salida}")
                else:
                    # Aplicar fondo a la imagen completa
                    resultado = aplicar_fondo(imagen_frente, imagen_fondo, modo_ajuste)
                    nombre_salida = f"{prefijo}{archivo}"
                    ruta_salida = os.path.join(directorio_salida, nombre_salida)
                    resultado.save(ruta_salida)
                    print(f"Imagen procesada: {ruta_salida}")
    
    else:
        # Procesar un solo archivo
        imagen_frente = cargar_imagen(ruta_frente)
        
        if imagen_frente:
            # Detectar sprites por número si se especificaron
            if num_horizontal > 0 and num_vertical > 0:
                sprites, limites = detectar_sprites_por_numero(imagen_frente, num_horizontal, num_vertical)
                for i, sprite in enumerate(sprites):
                    # Aplicar fondo a cada sprite
                    resultado = aplicar_fondo(sprite, imagen_fondo, modo_ajuste)
                    nombre_base = os.path.basename(ruta_frente).split('.')[0]
                    nombre_salida = f"{prefijo}{nombre_base}_{i+1}.png"
                    ruta_salida = os.path.join(directorio_salida, nombre_salida)
                    resultado.save(ruta_salida)
                    print(f"Sprite procesado: {ruta_salida}")
            else:
                # Aplicar fondo a la imagen completa
                resultado = aplicar_fondo(imagen_frente, imagen_fondo, modo_ajuste)
                nombre_base = os.path.basename(ruta_frente)
                nombre_salida = f"{prefijo}{nombre_base}"
                ruta_salida = os.path.join(directorio_salida, nombre_salida)
                
                resultado.save(ruta_salida)
                print(f"Imagen procesada: {ruta_salida}")

def main():
    parser = argparse.ArgumentParser(description="Aplica un fondo a imágenes con transparencia y luego las recorta.")
    parser.add_argument("frente", help="Ruta de la imagen o directorio con imágenes frontales")
    parser.add_argument("fondo", help="Ruta de la imagen de fondo a aplicar")
    parser.add_argument("--salida", default="imagenes_con_fondo", 
                        help="Directorio donde guardar las imágenes procesadas (predeterminado: 'imagenes_con_fondo')")
    parser.add_argument("--prefijo", default="fondo_", 
                        help="Prefijo para los archivos de salida (predeterminado: 'fondo_')")
    parser.add_argument("--modo", choices=["estirar", "centrar", "mosaico", "escalar"], default="estirar",
                        help="Modo de ajuste del fondo (predeterminado: 'estirar')")
    parser.add_argument("--num_horizontal", type=int, default=0,
                        help="Número de sprites horizontalmente (para detectar sprites por número)")
    parser.add_argument("--num_vertical", type=int, default=0,
                        help="Número de sprites verticalmente (para detectar sprites por número)")
    
    args = parser.parse_args()
    
    procesar_imagenes(
        args.frente, 
        args.fondo, 
        args.salida, 
        args.prefijo, 
        args.modo,
        args.num_horizontal,
        args.num_vertical
    )
    print("Proceso completado con éxito.")

if __name__ == "__main__":
    main()

# como hago la funcion por terminal de que se duplique la imagen de fondo en la imagen frontal
# y que se guarde en un directorio de salida
# python aplicar_fondo.py sprites sprites/fondo.png --salida sprites_con_fondo --modo mosaicopython aplicar_fondo.py cartas.png fondo.png --salida cartas_con_fondo --prefijo carta_ --modo escalar --num_horizontal 13 --num_vertical 4