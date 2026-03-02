import os
import cv2
import numpy as np
from PIL import Image
import argparse

def remove_black_borders_and_resize(input_dir, output_dir, target_size=(32, 32)):
    """
    Lädt Bilder aus input_dir und Unterordnern, entfernt ALLE schwarzen Ränder 
    und skaliert sie auf 32x32, sodass der Buchstabe das gesamte Bild ausfüllt.
    """
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = output_dir if relative_path == '.' else os.path.join(output_dir, relative_path)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        image_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        if image_files:
            print(f"Verarbeite Ordner: {root} ({len(image_files)} Bilder)")
        
        for img_file in image_files:
            input_path = os.path.join(root, img_file)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Fehler beim Laden: {img_file}")
                continue
            
            # 1. Initiale Schwellenwertbildung
            _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            
            # 2. Finde den Buchstaben (größte zusammenhängende Komponente)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
            
            if num_labels < 2:  # Nur Hintergrund vorhanden
                print(f"Kein Buchstabe gefunden in: {img_file}")
                continue
            
            # Finde die größte Komponente (nicht Hintergrund)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # Erstelle Maske für den Buchstaben
            mask = (labels == largest_label).astype(np.uint8) * 255
            
            # 3. Finde Grenzen des Buchstabens aus der Maske
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                print(f"Keine Pixel in Maske für: {img_file}")
                continue
                
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            
            # 4. Bescheide das ORIGINALBILD mit den Koordinaten aus der Maske
            # Füge etwas Rand hinzu (1 Pixel), um sicherzustellen, dass nichts abgeschnitten wird
            pad = 1
            h, w = img.shape
            min_y = max(0, min_y - pad)
            max_y = min(h, max_y + pad + 1)
            min_x = max(0, min_x - pad)
            max_x = min(w, max_x + pad + 1)
            
            # Bescheide das Originalbild
            cropped = img[min_y:max_y, min_x:max_x]
            
            # 5. Auf 32x32 skalieren - Buchstabe soll komplett ausfüllen
            # Zuerst bestimmen wir das Seitenverhältnis
            crop_h, crop_w = cropped.shape
            target_h, target_w = target_size
            
            # Bestimme Skalierungsfaktor, sodass der Buchstabe das Zielformat komplett ausfüllt
            # Wir skalieren so, dass die längere Seite genau 32 Pixel wird
            scale_h = target_h / crop_h
            scale_w = target_w / crop_w
            
            # Verwende den größeren Skalierungsfaktor, um sicherzustellen, dass der Buchstabe das Bild ausfüllt
            scale = max(scale_h, scale_w)
            
            # Temporäre Größe berechnen
            temp_h = int(crop_h * scale)
            temp_w = int(crop_w * scale)
            
            # Skaliere auf temporäre Größe
            resized_temp = cv2.resize(cropped, (temp_w, temp_h), 
                                     interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            
            # 6. Schneide auf 32x32 zu - Zentriere den Buchstaben
            start_y = max(0, (temp_h - target_h) // 2)
            start_x = max(0, (temp_w - target_w) // 2)
            
            # Finales Bild zuschneiden
            final_img = resized_temp[start_y:start_y+target_h, start_x:start_x+target_w]
            
            # 7. Sicherstellen, dass wirklich keine schwarzen Ränder mehr da sind
            # Erneut Schwellenwert anwenden, um schwache Ränder zu entfernen
            _, final_binary = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
            
            # Finde erneut die Grenzen im finalen Bild
            coords_final = np.where(final_binary > 0)
            if len(coords_final[0]) > 0:
                final_min_y, final_max_y = np.min(coords_final[0]), np.max(coords_final[0])
                final_min_x, final_max_x = np.min(coords_final[1]), np.max(coords_final[1])
                
                # Wenn der Buchstabe nicht das ganze Bild ausfüllt, kontrast erhöhen
                if (final_max_y - final_min_y < target_h - 5) or (final_max_x - final_min_x < target_w - 5):
                    # Kontrast erhöhen
                    final_img = cv2.convertScaleAbs(final_img, alpha=1.5, beta=0)
            
            # 8. Speichern
            output_path = os.path.join(output_subdir, img_file)
            
            # Optional: Invertieren, falls Buchstabe weiß auf schwarzem Grund sein soll
            # final_img = 255 - final_img
            
            Image.fromarray(final_img).save(output_path)
            print(f"  ✓ {img_file}")
    
    print(f"\nFertig! {len(image_files)} Bilder gespeichert in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entfernt ALLE schwarzen Ränder und skaliert auf 32x32.')
    parser.add_argument('--input', '-i', default='./letters', help='Eingabeordner mit Bildern')
    parser.add_argument('--output', '-o', default='./output', help='Ausgabeordner für verarbeitete Bilder')
    
    args = parser.parse_args()
    
    remove_black_borders_and_resize(args.input, args.output)