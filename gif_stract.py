import sys
import os
from PIL import Image


def extract_frames(gif_path: str, output_dir: str = None):
    if not os.path.isfile(gif_path):
        print(f"Error: no se encontró el archivo '{gif_path}'")
        sys.exit(1)

    gif = Image.open(gif_path)

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(gif_path))[0]
        output_dir = os.path.join(os.path.dirname(gif_path), base_name + "_frames")

    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    try:
        while True:
            frame = gif.convert("RGBA")
            out_path = os.path.join(output_dir, f"frame_{frame_index:04d}.png")
            frame.save(out_path)
            print(f"  Guardado: {out_path}")
            frame_index += 1
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    print(f"\nTotal: {frame_index} fotogramas extraídos en '{output_dir}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python gif_stract.py <archivo.gif> [carpeta_salida]")
        sys.exit(1)

    gif_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else None

    extract_frames(gif_path, output_dir)
