import os
from PIL import Image
import numpy as np

def scale_images_in_folder(folder_path, target_size=32):
    """
    Skaliert alle Bilder in einem Ordner auf 32x32 Pixel,
    behält dabei das Seitenverhältnis bei und zentriert das Bild.
    """
    if not os.path.exists(folder_path):
        print(f"Ordner nicht gefunden: {folder_path}")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Nur Bilddateien verarbeiten
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        
        try:
            # Bild öffnen
            img = Image.open(file_path)
            
            # In RGB konvertieren falls nötig
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Seitenverhältnis beibehalten und auf 30x30 skalieren
            img.thumbnail((30, 30), Image.Resampling.LANCZOS)
            
            # Weiße Canvas 32x32 erstellen
            canvas = Image.new('RGB', (target_size, target_size), 'white')
            
            # Bild zentriert einfügen
            offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
            canvas.paste(img, offset)
            
            # Skaliertes Bild speichern
            canvas.save(file_path)
            print(f"✓ {filename} skaliert")
        
        except Exception as e:
            print(f"✗ Fehler bei {filename}: {e}")



def rename_images_in_folder(letters_folder,folder_path):
    """
    Benennt alle Bilder in einem Ordner um, indem es die Dateiendung entfernt
    und die Bilder fortlaufend nummeriert (z.B. B_.png, B.png, ...).
    """
    if not os.path.exists(folder_path):
        print(f"Ordner nicht gefunden: {folder_path}")
        return
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    for index, filename in enumerate(image_files):
        file_path = os.path.join(folder_path, filename)
        new_filename = f"{os.path.basename(folder_path)}_Peintner_{index}.png"
        new_file_path = os.path.join(folder_path, new_filename)
        
        try:
            # Bild öffnen und als PNG speichern
            img = Image.open(file_path)
            img.save(new_file_path)
            
            # Originaldatei löschen
            os.remove(file_path)
            print(f"✓ {filename} umbenannt zu {new_filename}")
        
        except Exception as e:
            print(f"✗ Fehler bei {filename}: {e}")

# Pfad zum Letters-Ordner
letters_folder = "letters"

scale_images_in_folder(letters_folder)
# Alle Buchstaben-Ordner verarbeiten
for letter_folder in os.listdir(letters_folder):
    letter_path = os.path.join(letters_folder, letter_folder)
    if os.path.isdir(letter_path):
        print(f"Verarbeite Ordner: {letter_folder}")
        rename_images_in_folder(letters_folder, letter_path)

print("Fertig!")