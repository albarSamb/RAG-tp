import numpy as np
from sentence_transformers import SentenceTransformer


def charger_modele_embedding() -> SentenceTransformer:
    """
    Charge le modèle d'embedding multilingue.

    Le modèle est téléchargé automatiquement la première fois,
    puis mis en cache localement pour les lancements suivants.

    Returns:
        Le modèle SentenceTransformer prêt à encoder
    """
    print("Chargement du modèle d'embedding...")
    modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print("Modèle chargé.")
    return modele


def embedder_chunks(chunks: list[str], modele: SentenceTransformer) -> np.ndarray:
    """
    Transforme une liste de textes en vecteurs numériques.

    Chaque texte est converti en un vecteur de 768 dimensions.
    Le résultat est un tableau numpy de shape (n_chunks, 768).

    Args:
        chunks: liste de textes à encoder
        modele: le modèle SentenceTransformer (doit être le MÊME
                que celui utilisé pour encoder les questions plus tard)

    Returns:
        Tableau numpy de shape (n_chunks, 768), type float32
    """
    print(f"Encodage de {len(chunks)} chunks en vecteurs...")

    # batch_size=64 : encode 64 textes en parallèle pour aller plus vite
    vecteurs = modele.encode(
        chunks,
        show_progress_bar=True,
        batch_size=64,
    )

    # Conversion en float32 car FAISS exige ce type précis
    vecteurs = np.array(vecteurs, dtype=np.float32)

    # Vérification : la dimension doit être 768
    print(f"Vecteurs créés — shape : {vecteurs.shape}")
    assert vecteurs.shape[1] == 768, (
        f"Dimension inattendue : {vecteurs.shape[1]} (attendu 768)"
    )

    return vecteurs
