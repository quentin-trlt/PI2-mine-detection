"""
Script de diagnostic: Vérifier la structure du dataset téléchargé
"""
import sys
from pathlib import Path
import zipfile
import shutil

sys.path.append(str(Path(__file__).parent.parent))
import config


def explore_directory(path, level=0, max_level=3):
    """Explore récursivement la structure"""
    if level > max_level:
        return

    indent = "  " * level
    for item in sorted(path.iterdir()):
        if item.is_dir():
            num_files = len(list(item.glob('*')))
            print(f"{indent}{item.name}/ ({num_files} items)")
            if num_files < 50:
                explore_directory(item, level + 1)
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"{indent}{item.name} ({size_mb:.2f} MB)")


def find_and_extract_zips():
    """Trouve et extrait les fichiers ZIP"""
    for zip_file in config.RAW_DATA_DIR.rglob('*.zip'):
        print(f"\nExtraction de {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            extract_path = zip_file.parent / zip_file.stem
            zip_ref.extractall(extract_path)
        print(f"✓ Extrait dans {extract_path}")


def reorganize_if_nested():
    """Réorganise si les classes sont dans un sous-dossier"""
    # Chercher le vrai dossier de données
    candidates = []
    for item in config.RAW_DATA_DIR.rglob('*'):
        if item.is_dir():
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            if len(subdirs) > 2:  # Potentiellement les classes
                has_images = any(
                    list(d.glob('*.jpg')) or list(d.glob('*.png'))
                    for d in subdirs[:3]
                )
                if has_images:
                    candidates.append((item, len(subdirs)))

    if candidates:
        # Prendre celui avec le plus de sous-dossiers
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        print(f"\nDossier de classes trouvé: {best_candidate}")

        # Déplacer au bon endroit si nécessaire
        if best_candidate != config.RAW_DATA_DIR:
            print("Réorganisation...")
            for class_dir in best_candidate.iterdir():
                if class_dir.is_dir():
                    target = config.RAW_DATA_DIR / class_dir.name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(class_dir), str(target))
            print("✓ Réorganisé")


if __name__ == "__main__":
    print("=== Diagnostic du dataset ===\n")

    print("Structure actuelle:")
    explore_directory(config.RAW_DATA_DIR)

    find_and_extract_zips()
    reorganize_if_nested()

    print("\n\nStructure finale:")
    explore_directory(config.RAW_DATA_DIR, max_level=2)

    # Vérifier classes
    print("\n\nClasses détectées:")
    classes = []
    for item in config.RAW_DATA_DIR.iterdir():
        if item.is_dir():
            num_jpg = len(list(item.glob('*.jpg')))
            num_png = len(list(item.glob('*.png')))
            total = num_jpg + num_png
            if total > 0:
                classes.append(item.name)
                print(f"  {item.name}: {total} images")

    print(f"\nTotal: {len(classes)} classes")