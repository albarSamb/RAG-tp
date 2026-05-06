import json
import os

import faiss
import numpy as np


def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """
    Crée un index FAISS à partir des vecteurs.

    IndexFlatL2 = recherche exacte par distance euclidienne.
    - "Flat" signifie que tous les vecteurs sont stockés tels quels
      (pas de compression, pas d'approximation)
    - "L2" signifie que la distance utilisée est la distance euclidienne
      (la "longueur" du segment entre deux points dans l'espace 768D)

    La dimension (768) DOIT correspondre à celle du modèle d'embedding.
    Si ce n'est pas le cas, FAISS plantera à l'ajout des vecteurs.

    Args:
        vecteurs: tableau numpy de shape (n, 768), type float32

    Returns:
        Index FAISS prêt pour la recherche
    """
    dimension = vecteurs.shape[1]  # 768 pour notre modèle
    index = faiss.IndexFlatL2(dimension)

    # Ajouter tous les vecteurs d'un coup dans l'index
    # Chaque vecteur reçoit un identifiant = son rang (0, 1, 2, ...)
    index.add(vecteurs)

    print(f"Index FAISS créé — {index.ntotal} vecteurs indexés (dimension {dimension})")
    return index


def sauvegarder_index(index: faiss.Index, chunks_avec_meta: list, chemin: str):
    """
    Sauvegarde l'index FAISS et les métadonnées sur disque.

    Deux fichiers sont créés :
    - {chemin}.index → l'index FAISS (vecteurs binaires)
    - {chemin}.json  → les chunks + métadonnées (texte lisible)

    IMPORTANT : les chunks dans le JSON doivent être dans le MÊME ORDRE
    que les vecteurs dans l'index FAISS. Le vecteur n°0 correspond au
    chunk n°0 du JSON, le vecteur n°1 au chunk n°1, etc.

    Args:
        index: l'index FAISS avec les vecteurs
        chunks_avec_meta: liste de dicts {contenu, metadata, chunk_id}
        chemin: chemin de base (sans extension)
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(chemin), exist_ok=True)

    # Sauvegarder l'index FAISS (format binaire propriétaire)
    faiss.write_index(index, chemin + ".index")
    print(f"Index FAISS sauvegardé → {chemin}.index")

    # Sauvegarder les métadonnées en JSON (lisible et portable)
    with open(chemin + ".json", "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)
    print(f"Métadonnées sauvegardées → {chemin}.json")


def charger_index(chemin: str):
    """
    Recharge l'index FAISS et les métadonnées depuis le disque.

    C'est cette fonction que rag.py appelle au démarrage.
    Elle évite de recalculer tous les embeddings à chaque lancement.

    Args:
        chemin: chemin de base (sans extension)

    Returns:
        (index FAISS, liste des chunks avec métadonnées)
    """
    index = faiss.read_index(chemin + ".index")

    with open(chemin + ".json", "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)

    print(f"Index chargé depuis le disque — {index.ntotal} vecteurs")

    # Vérification de cohérence : le nombre de vecteurs dans FAISS
    # doit correspondre au nombre de chunks dans le JSON
    assert index.ntotal == len(chunks_avec_meta), (
        f"Incohérence : {index.ntotal} vecteurs FAISS vs "
        f"{len(chunks_avec_meta)} chunks JSON"
    )

    return index, chunks_avec_meta
