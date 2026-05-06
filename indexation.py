"""
indexation.py — Script principal de la PHASE 1 (indexation)

Ce script orchestre le pipeline complet :
  CSV → Nettoyage → Chunking → Embedding → Base FAISS

À exécuter UNE SEULE FOIS. La base est ensuite sauvegardée sur disque
et rechargée par rag.py sans réindexation.

Usage : python indexation.py
"""

from modules.chargement import charger_films


def main():
    chemin_csv = "data/tmdb_5000_movies.csv"

    # ── Étape 1 : Charger et nettoyer les données ──
    print("=" * 60)
    print("ÉTAPE 1 — Chargement des données")
    print("=" * 60)
    documents = charger_films(chemin_csv)

    # Aperçu du premier document pour vérifier
    print("\n--- Aperçu du premier document ---")
    print(f"ID : {documents[0]['id']}")
    print(f"Contenu : {documents[0]['contenu'][:200]}...")
    print(f"Metadata : {documents[0]['metadata']}")


if __name__ == "__main__":
    main()
