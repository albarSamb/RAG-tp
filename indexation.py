"""
indexation.py — Script principal de la PHASE 1 (indexation)

Ce script orchestre le pipeline complet :
  CSV → Nettoyage → Chunking → Embedding → Base FAISS

À exécuter UNE SEULE FOIS. La base est ensuite sauvegardée sur disque
et rechargée par rag.py sans réindexation.

Usage : python indexation.py
"""

from modules.chargement import charger_films
from modules.chunking import chunker


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

    # ── Étape 2 : Découper en chunks ──
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Chunking")
    print("=" * 60)

    chunks_avec_meta = []
    for doc in documents:
        # On découpe le contenu de chaque film en chunks
        chunks = chunker(doc["contenu"], taille_max=500, overlap=50)
        for i, chunk in enumerate(chunks):
            chunks_avec_meta.append({
                "contenu": chunk,
                "metadata": doc["metadata"],
                "chunk_id": f"{doc['id']}_chunk_{i}"
            })

    print(f"Total chunks créés : {len(chunks_avec_meta)}")

    # Statistique : combien de films ont été découpés en plusieurs chunks ?
    films_multi_chunks = sum(
        1 for doc in documents
        if len(chunker(doc["contenu"], taille_max=500, overlap=50)) > 1
    )
    print(f"Films découpés en plusieurs chunks : {films_multi_chunks}")

    # Aperçu d'un chunk
    print("\n--- Aperçu du premier chunk ---")
    print(f"Chunk ID : {chunks_avec_meta[0]['chunk_id']}")
    print(f"Contenu : {chunks_avec_meta[0]['contenu'][:200]}...")
    print(f"Metadata : {chunks_avec_meta[0]['metadata']}")


if __name__ == "__main__":
    main()
