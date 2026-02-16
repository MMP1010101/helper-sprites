import os

# Define the colors and their order
colors = ['corazon', 'treboles', 'diamantes', 'picas']
# Define the ranks
ranks = [str(i) for i in range(2, 11)] + ['j', 'r', 'k']

def rename_files(directory):
    # Get list of files in the directory
    files = os.listdir(directory)
    print(f"Found {len(files)} files in directory")
    
    # Print first few files to understand naming format
    for i, file in enumerate(files[:5]):
        print(f"Example file {i}: {file}")
    
    # Para imágenes con formato "carta_cartasNombre_X"
    images_to_rename = []
    for file in files:
        if file.startswith('carta_cartasNombre_') and file.endswith('.png'):
            try:
                # Extraer el número después del último guion bajo
                card_num = int(file.split('_')[-1].split('.')[0])
                
                # Ajuste: restar 1 porque el mapeo comienza desde 0 internamente
                card_index = card_num - 1
                
                # Determinar color y rango
                color_index = card_index // 13
                rank_index = card_index % 13
                
                if color_index < len(colors) and rank_index < len(ranks):
                    new_name = f"{colors[color_index]}_{ranks[rank_index]}.png"
                    images_to_rename.append((file, new_name))
                    print(f"Will rename {file} to {new_name}")
                else:
                    print(f"Skipping file: {file} - index out of range")
            except (ValueError, IndexError):
                print(f"Skipping file: {file} - doesn't match expected format")
    
    # Rename files
    if images_to_rename:
        proceed = input("¿Proceder con el renombrado? (s/n): ")
        if proceed.lower() == 's':
            for old_name, new_name in images_to_rename:
                os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))
                print(f"Renamed {old_name} to {new_name}")
            print("Renombrado completado.")
        else:
            print("Operación cancelada.")
    else:
        print("No se encontraron archivos para renombrar.")

# Define the directory containing the card images
directory = 'cartas_con_fondo'
rename_files(directory)
