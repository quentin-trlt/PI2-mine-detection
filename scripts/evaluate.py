"""
Script 04: Évaluation du modèle sur le test set
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.augmentation import create_dataset
from utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    print_classification_report,
    get_misclassified_samples
)


def load_data_from_directory(split_name):
    """Charge les données depuis un split"""
    split_dir = config.SPLITS_DIR / split_name

    file_paths = []
    labels = []

    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(config.CLASSES))}

    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]

            for img_path in class_dir.glob('*.jpg'):
                file_paths.append(str(img_path))
                labels.append(class_idx)

            for img_path in class_dir.glob('*.png'):
                file_paths.append(str(img_path))
                labels.append(class_idx)

    return np.array(file_paths), np.array(labels)


def evaluate_model(model_path, split='test'):
    """Évalue le modèle sur un split donné"""
    print("=" * 60)
    print(f"ÉVALUATION SUR {split.upper()}")
    print("=" * 60)

    # Charger modèle
    print(f"\nChargement du modèle: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Charger données
    print(f"\nChargement des données {split}...")
    file_paths, labels = load_data_from_directory(split)
    print(f"Nombre d'images: {len(file_paths)}")

    # Créer dataset
    dataset = create_dataset(file_paths, labels, training=False)

    # Évaluation
    print("\nÉvaluation en cours...")
    results = model.evaluate(dataset, verbose=1)

    print("\n" + "=" * 60)
    print("MÉTRIQUES GLOBALES")
    print("=" * 60)
    for metric_name, value in zip(model.metrics_names, results):
        print(f"{metric_name}: {value:.4f}")

    # Prédictions
    print("\nGénération des prédictions...")
    predictions = model.predict(dataset, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = labels

    # Rapport de classification
    print_classification_report(y_true, y_pred)

    # Métriques détaillées
    metrics = calculate_metrics(y_true, y_pred)

    print("\n" + "=" * 60)
    print("MÉTRIQUES PAR CLASSE")
    print("=" * 60)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  Precision: {class_metrics['precision']:.4f}")
        print(f"  Recall: {class_metrics['recall']:.4f}")
        print(f"  F1-Score: {class_metrics['f1']:.4f}")
        print(f"  Support: {class_metrics['support']}")

    # Matrice de confusion
    model_dir = Path(model_path).parent
    confusion_matrix_path = model_dir / f'confusion_matrix_{split}.png'
    plot_confusion_matrix(y_true, y_pred, save_path=confusion_matrix_path)

    # Échantillons mal classifiés
    print("\n" + "=" * 60)
    print("ÉCHANTILLONS MAL CLASSIFIÉS (Top 10)")
    print("=" * 60)
    misclassified = get_misclassified_samples(y_true, y_pred, file_paths, top_n=10)
    for i, sample in enumerate(misclassified, 1):
        print(f"\n{i}. {Path(sample['file']).name}")
        print(f"   Vraie classe: {sample['true_class']}")
        print(f"   Prédite: {sample['predicted_class']}")

    # Analyse des confiances
    print("\n" + "=" * 60)
    print("ANALYSE DES CONFIANCES")
    print("=" * 60)

    max_probs = np.max(predictions, axis=1)
    correct_mask = y_true == y_pred

    print(f"Confiance moyenne (correctes): {np.mean(max_probs[correct_mask]):.4f}")
    print(f"Confiance moyenne (incorrectes): {np.mean(max_probs[~correct_mask]):.4f}")
    print(f"Confiance min: {np.min(max_probs):.4f}")
    print(f"Confiance max: {np.max(max_probs):.4f}")

    # Sauvegarder les résultats
    results_file = model_dir / f'evaluation_{split}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Évaluation sur {split}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Métriques globales:\n")
        for metric_name, value in zip(model.metrics_names, results):
            f.write(f"{metric_name}: {value:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Métriques par classe:\n\n")
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {class_metrics['f1']:.4f}\n")
            f.write(f"  Support: {class_metrics['support']}\n\n")

    print(f"\n✓ Résultats sauvegardés: {results_file}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Évaluation du modèle')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle (.keras)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Split à évaluer'
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Modèle non trouvé: {model_path}")
        sys.exit(1)

    evaluate_model(str(model_path), args.split)

    print("\n=== Évaluation terminée ===")