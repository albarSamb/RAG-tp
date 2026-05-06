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
from modules.embedding import charger_modele_embedding, embedder_chunks
from modules.faiss_index import creer_index_faiss, sauvegarder_index


def main():
    chemin_csv = "data/tmdb_5000_movies.csv"
    chemin_index = "index_data/films"

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
        chunks = chunker(doc["contenu"], taille_max=500, overlap=50)
        for i, chunk in enumerate(chunks):
            chunks_avec_meta.append({
                "contenu": chunk,
                "metadata": doc["metadata"],
                "chunk_id": f"{doc['id']}_chunk_{i}"
            })

    print(f"Total chunks créés : {len(chunks_avec_meta)}")

    # ── Étape 3 : Créer les embeddings ──
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Embeddings")
    print("=" * 60)

    modele = charger_modele_embedding()
    textes = [c["contenu"] for c in chunks_avec_meta]
    vecteurs = embedder_chunks(textes, modele)

    # ── Étape 4 : Créer et sauvegarder l'index FAISS ──
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Index FAISS")
    print("=" * 60)

    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks_avec_meta, chemin_index)

    # ── Résumé ──
    print("\n" + "=" * 60)
    print("INDEXATION TERMINÉE")
    print(f"  → {len(documents)} films traités")
    print(f"  → {len(chunks_avec_meta)} chunks indexés")
    print(f"  → Index sauvegardé dans {chemin_index}.*")
    print("=" * 60)


if __name__ == "__main__":
    main()
