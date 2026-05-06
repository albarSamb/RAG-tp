"""
Module chargement — Étape 1 du pipeline RAG

Responsabilité : charger le CSV TMDB et transformer chaque film
en un document structuré prêt à être embeddé.

Le format de sortie est une liste de dicts :
{
    "id": "film_0001",
    "contenu": "texte complet à embedder",
    "metadata": { "titre", "annee", "note", "genres", "langue", "source" }
}
"""

import json
import pandas as pd


def extraire_genres(genres_json: str) -> str:
    """
    Convertit la colonne genres du CSV (JSON imbriqué) en texte lisible.

    Entrée : '[{"id": 28, "name": "Action"}, {"id": 14, "name": "Fantasy"}]'
    Sortie : 'Action, Fantasy'

    json.loads() transforme la string JSON en liste de dicts Python,
    puis on récupère le champ "name" de chaque élément.
    """
    try:
        genres_list = json.loads(genres_json)
        return ", ".join(g["name"] for g in genres_list)
    except (json.JSONDecodeError, TypeError):
        return "Non spécifié"


def charger_films(chemin_csv: str) -> list[dict]:
    """
    Charge le CSV TMDB et construit la liste de documents.

    Chaque film est transformé en un TEXTE qui combine les colonnes
    importantes. C'est ce texte qui sera ensuite embeddé par le modèle.

    Choix des colonnes incluses dans le texte à embedder :
    - title     → pour les requêtes "un film comme Inception"
    - genres    → pour les requêtes "je cherche un thriller"
    - overview  → le contenu principal, la description du film
    - note/année/durée → pour les filtres "bien noté", "récent"

    Les films sans synopsis sont ignorés car l'embedding
    n'aurait rien de significatif à capturer.
    """
    df = pd.read_csv(chemin_csv)
    print(f"Nombre de films dans le CSV : {len(df)}")

    documents = []

    for idx, row in df.iterrows():
        # Extraire les genres du JSON imbriqué
        genres = extraire_genres(row["genres"])

        # Ignorer les films sans synopsis
        overview = row.get("overview", "")
        if pd.isna(overview) or not str(overview).strip():
            continue

        titre = str(row.get("title", "Sans titre"))
        annee = str(row.get("release_date", ""))[:4]  # "2014-05-01" → "2014"
        note = row.get("vote_average", 0)
        nb_votes = row.get("vote_count", 0)
        langue = row.get("original_language", "en")
        duree = row.get("runtime", 0)

        # Construction du texte à embedder
        # On combine toutes les infos en un paragraphe cohérent
        contenu = (
            f"{titre} ({annee}). "
            f"Genres : {genres}. "
            f"Note : {note}/10 ({int(nb_votes)} votes). "
            f"Durée : {int(duree)} min. "
            f"Synopsis : {overview}"
        )

        documents.append({
            "id": f"film_{idx:04d}",
            "contenu": contenu,
            # Les métadonnées ne sont PAS embeddées mais servent
            # à afficher les infos du film dans la réponse finale
            "metadata": {
                "titre": titre,
                "annee": annee,
                "note": float(note),
                "genres": genres,
                "langue": langue,
                "source": f"tmdb_5000_movies.csv ligne {idx}",
            }
        })

    print(f"Films retenus (avec synopsis) : {len(documents)}")
    return documents
