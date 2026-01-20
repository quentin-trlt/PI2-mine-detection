"""
Script 01: Configuration Kaggle et téléchargement du dataset
"""
import os
import sys
import json
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config


def setup_kaggle_credentials():
    """Configure les credentials Kaggle"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)

    credentials = {
        'username': 'your_username',  # Sera extrait du token
        'key': config.KAGGLE_TOKEN
    }

    kaggle_json = kaggle_dir / 'kaggle.json'
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)

    os.chmod(kaggle_json, 0o600)
    print(f"✓ Credentials Kaggle configurées dans {kaggle_json}")


def download_dataset():
    """Télécharge le dataset depuis Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print(f"Téléchargement de {config.KAGGLE_DATASET}...")

        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        api.dataset_download_files(
            config.KAGGLE_DATASET,
            path=str(config.RAW_DATA_DIR),
            unzip=True
        )

        print(f"✓ Dataset téléchargé dans {config.RAW_DATA_DIR}")

        # Vérifier la structure
        verify_dataset_structure()

    except Exception as e:
        print(f"✗ Erreur lors du téléchargement: {e}")
        sys.exit(1)


def verify_dataset_structure():
    """Vérifie la structure du dataset téléchargé"""
    print("\nStructure du dataset:")

    classes = []
    total_images = 0

    for item in sorted(config.RAW_DATA_DIR.iterdir()):
        if item.is_dir():
            num_images = len(list(item.glob('*.jpg'))) + len(list(item.glob('*.png')))
            if num_images > 0:
                classes.append(item.name)
                total_images += num_images
                print(f"  {item.name}: {num_images} images")

    print(f"\nTotal: {len(classes)} classes, {total_images} images")

    # Mettre à jour config
    update_config_classes(classes)


def update_config_classes(classes):
    """Met à jour le fichier config avec les classes trouvées"""
    config_path = Path(__file__).parent.parent / 'config.py'

    with open(config_path, 'r') as f:
        content = f.read()

    content = content.replace('CLASSES = []', f'CLASSES = {sorted(classes)}')
    content = content.replace('NUM_CLASSES = 0', f'NUM_CLASSES = {len(classes)}')

    with open(config_path, 'w') as f:
        f.write(content)

    print(f"✓ Config mis à jour: {len(classes)} classes")


if __name__ == "__main__":
    print("=== Setup Kaggle et téléchargement ===\n")
    setup_kaggle_credentials()
    download_dataset()
    print("\n=== Terminé ===")