"""
Script 02: Préparation des données (split et organisation)
"""
import sys
from pathlib import Path
import shutil
import random
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

import config


def set_seeds():
    """Fixe les seeds pour reproductibilité"""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)


def split_dataset():
    """Split le dataset en train/val/test"""
    set_seeds()

    print("Création des splits train/val/test...")

    # Parcourir chaque classe
    for class_dir in config.RAW_DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"\nTraitement de la classe: {class_name}")

        # Récupérer toutes les images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        random.shuffle(images)

        total = len(images)
        train_end = int(total * config.TRAIN_RATIO)
        val_end = train_end + int(total * config.VAL_RATIO)

        # Split
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(f"  Total: {total}")
        print(f"  Train: {len(train_images)}")
        print(f"  Val: {len(val_images)}")
        print(f"  Test: {len(test_images)}")

        # Créer répertoires cibles
        for split_name in ['train', 'val', 'test']:
            split_dir = config.SPLITS_DIR / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

        # Copier les fichiers
        copy_files(train_images, config.SPLITS_DIR / 'train' / class_name)
        copy_files(val_images, config.SPLITS_DIR / 'val' / class_name)
        copy_files(test_images, config.SPLITS_DIR / 'test' / class_name)

    print("\n✓ Splits créés avec succès")


def copy_files(file_list, target_dir):
    """Copie une liste de fichiers vers un répertoire cible"""
    for file_path in file_list:
        shutil.copy2(file_path, target_dir / file_path.name)


def verify_splits():
    """Vérifie les splits créés"""
    print("\n" + "=" * 60)
    print("VÉRIFICATION DES SPLITS")
    print("=" * 60)

    for split_name in ['train', 'val', 'test']:
        split_dir = config.SPLITS_DIR / split_name
        print(f"\n{split_name.upper()}:")

        total = 0
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                num_images = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                total += num_images
                print(f"  {class_dir.name}: {num_images} images")

        print(f"  Total: {total} images")


def create_file_lists():
    """Crée des fichiers texte avec les chemins pour chaque split"""
    print("\nCréation des listes de fichiers...")

    for split_name in ['train', 'val', 'test']:
        split_dir = config.SPLITS_DIR / split_name
        output_file = config.DATA_DIR / f'{split_name}_files.txt'

        with open(output_file, 'w') as f:
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_path in sorted(class_dir.glob('*.jpg')):
                        f.write(f"{img_path},{class_name}\n")
                    for img_path in sorted(class_dir.glob('*.png')):
                        f.write(f"{img_path},{class_name}\n")

        print(f"✓ {output_file}")


if __name__ == "__main__":
    print("=== Préparation des données ===\n")

    # Vérifier que le dataset brut existe
    if not config.RAW_DATA_DIR.exists():
        print("✗ Dataset brut non trouvé. Exécutez d'abord 01_setup_kaggle.py")
        sys.exit(1)

    split_dataset()
    verify_splits()
    create_file_lists()

    print("\n=== Terminé ===")