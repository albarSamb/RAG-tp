"""
rag.py — Script principal de la PHASE 2 (interrogation)

Ce script assemble toutes les briques du pipeline RAG :
  Question → Embedding → Recherche FAISS → LLM Groq → Réponse

Il charge l'index FAISS depuis le disque (créé par indexation.py)
et lance une boucle interactive où l'utilisateur pose ses questions.

Fonctionnalités :
  - Recherche vectorielle dans la base de 4799 films
  - Génération de réponses avec citations (titre, année, note)
  - Filtre par langue (français ou toutes langues)
  - Le LLM refuse de répondre si l'info n'est pas dans la base

Usage : python rag.py
Prérequis : avoir exécuté python indexation.py au moins une fois
"""

from modules.embedding import charger_modele_embedding
from modules.faiss_index import charger_index
from modules.recherche import rechercher
from modules.generation import generer_reponse


def filtrer_par_langue(resultats: list[dict], langue: str) -> list[dict]:
    """
    Filtre les résultats de recherche par langue originale du film.

    Contrainte spécifique du sujet A : l'utilisateur peut demander
    uniquement des films en VO française ou en toutes langues.

    Args:
        resultats: liste de résultats de la recherche vectorielle
        langue: code langue à filtrer ("fr" pour français, "all" pour tout)

    Returns:
        Liste filtrée (ou inchangée si langue == "all")
    """
    if langue == "all":
        return resultats

    return [r for r in resultats if r["metadata"]["langue"] == langue]


def afficher_sources(resultats: list[dict]):
    """
    Affiche les sources utilisées pour la réponse.

    Le TP exige que chaque réponse cite ses sources.
    On affiche le titre, l'année et le score de similarité
    de chaque chunk utilisé.
    """
    print("\n--- Sources ---")
    for i, res in enumerate(resultats, 1):
        meta = res["metadata"]
        print(
            f"  {i}. {meta['titre']} ({meta['annee']}) — "
            f"Note : {meta['note']}/10 — "
            f"Score L2 : {res['score']:.2f}"
        )


def main():
    chemin_index = "index_data/films"

    # ── Chargement (une seule fois au démarrage) ──
    print("Chargement de la base de connaissances...")
    index, chunks_avec_meta = charger_index(chemin_index)
    modele = charger_modele_embedding()

    # ── Configuration du filtre langue ──
    print("\nFiltre par langue :")
    print("  1. Toutes les langues")
    print("  2. Films en français uniquement")
    choix_langue = input("Votre choix (1/2) : ").strip()
    langue = "fr" if choix_langue == "2" else "all"

    if langue == "fr":
        print("→ Filtre activé : films en français uniquement\n")
    else:
        print("→ Pas de filtre : toutes les langues\n")

    # ── Boucle interactive ──
    print("=" * 60)
    print("Système RAG prêt. Tapez 'quit' pour quitter.")
    print("=" * 60)

    while True:
        print()
        question = input("Votre question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        # 1. Rechercher les chunks pertinents
        #    Quand le filtre langue est actif, on cherche large (k=100)
        #    car les films français ne représentent que ~70/4799 films.
        #    Avec k=10 on risque de n'avoir aucun film français dans les résultats.
        k_recherche = 100 if langue != "all" else 10
        resultats = rechercher(question, modele, index, chunks_avec_meta, k=k_recherche)

        # 2. Appliquer le filtre par langue
        resultats = filtrer_par_langue(resultats, langue)

        # 3. Garder les 5 meilleurs résultats pour le LLM
        #    Trop de chunks = contexte trop long et réponse diluée
        resultats = resultats[:5]

        if not resultats:
            print("\nAucun film trouvé correspondant à votre demande.")
            continue

        # 4. Générer la réponse avec le LLM
        print("\nGénération de la réponse...\n")
        reponse = generer_reponse(question, resultats)
        print(reponse)

        # 5. Afficher les sources
        afficher_sources(resultats)


if __name__ == "__main__":
    main()
