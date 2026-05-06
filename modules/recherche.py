"""
Module recherche — Étape 5 du pipeline RAG

Responsabilité : prendre la question de l'utilisateur, la transformer
en vecteur, et trouver les chunks les plus proches dans l'index FAISS.

Comment ça marche ?
    1. On encode la question avec le MÊME modèle d'embedding que celui
       utilisé pour encoder les chunks à l'indexation
    2. On demande à FAISS : "quels sont les k vecteurs les plus proches
       de ce vecteur-question ?"
    3. FAISS retourne deux tableaux :
       - distances : la distance L2 entre la question et chaque résultat
       - indices   : le rang de chaque résultat (0, 1, 2...)
    4. On utilise les indices pour retrouver le texte et les métadonnées
       dans la liste chunks_avec_meta

Le score (distance L2) :
    - Plus la distance est PETITE, plus les textes sont proches (= pertinents)
    - Une distance de 0 = textes identiques
    - Ce n'est PAS un pourcentage de similarité, c'est une distance brute

Questions à se poser (TP) :
    - Le score L2 est une distance : petit = bon, grand = mauvais
    - On pourrait normaliser les vecteurs et utiliser IndexFlatIP pour
      avoir un vrai score de similarité cosinus (entre 0 et 1), mais
      L2 suffit pour classer les résultats par pertinence
    - IMPORTANT : tester la recherche AVANT de brancher le LLM.
      Si les résultats sont aberrants, le problème vient du chunking
      ou de l'embedding, pas du LLM
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def rechercher(
    question: str,
    modele: SentenceTransformer,
    index,
    chunks_avec_meta: list,
    k: int = 5,
) -> list[dict]:
    """
    Recherche les k chunks les plus pertinents pour une question.

    La question est encodée en vecteur avec le même modèle que les chunks,
    puis FAISS trouve les vecteurs les plus proches par distance L2.

    Args:
        question: la question de l'utilisateur en texte libre
        modele: le modèle d'embedding (DOIT être le même qu'à l'indexation)
        index: l'index FAISS chargé depuis le disque
        chunks_avec_meta: la liste des chunks avec métadonnées (même ordre
                          que les vecteurs dans l'index)
        k: nombre de résultats à retourner (défaut : 5)

    Returns:
        Liste de dicts triés par pertinence (du plus proche au plus loin) :
        {
            "contenu": "texte du chunk",
            "metadata": { titre, annee, note, ... },
            "score": 12.34  (distance L2, plus petit = plus pertinent)
        }
    """
    # 1. Encoder la question en vecteur (même modèle qu'à l'indexation)
    vecteur_question = modele.encode([question])
    vecteur_question = np.array(vecteur_question, dtype=np.float32)

    # 2. Recherche dans FAISS : trouver les k plus proches voisins
    #    distances = tableau des distances L2 pour chaque résultat
    #    indices   = tableau des rangs (positions) dans l'index
    distances, indices = index.search(vecteur_question, k)

    # 3. Construire la liste de résultats
    #    distances[0] et indices[0] car on n'a envoyé qu'une seule question
    resultats = []
    for dist, idx in zip(distances[0], indices[0]):
        # idx = -1 si FAISS n'a pas trouvé assez de résultats
        if idx == -1:
            continue

        chunk = chunks_avec_meta[idx]
        resultats.append({
            "contenu": chunk["contenu"],
            "metadata": chunk["metadata"],
            "score": float(dist),
        })

    return resultats
