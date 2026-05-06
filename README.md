# RAG — Recommandation de Films (Sujet A)

Système RAG (Retrieval-Augmented Generation) de recommandation de films basé sur le dataset TMDB 5000.

## Comment lancer le projet

### Prérequis

- Python 3.10+
- Un compte Groq avec une clé API

### Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Configuration

Créer un fichier `.env` à la racine :

```env
GROQ_API_KEY=votre_clé_ici
```

### Lancement

```bash
# Phase 1 — Indexation (une seule fois)
python indexation.py

# Phase 2 — Questions-réponses
python rag.py
```

## Architecture du projet

```
RAG-tp/
  data/
    tmdb_5000_movies.csv       <- dataset TMDB (5000 films)
  modules/
    chargement.py              <- chargement et nettoyage du CSV
    chunking.py                <- decoupage des textes en chunks
    embedding.py               <- transformation texte -> vecteurs
    faiss_index.py             <- creation et persistance de l'index FAISS
    recherche.py               <- recherche vectorielle (top-k)
    generation.py              <- generation de reponse avec Groq
  faiss_index/                 <- fichiers d'index generes
  indexation.py                <- script principal phase 1
  rag.py                       <- script principal phase 2
```

## Choix techniques

### Modele d'embedding : `paraphrase-multilingual-mpnet-base-v2`

Choisi car les synopsis sont en anglais mais les questions de l'utilisateur seront en francais. Ce modele multilingue permet de faire le pont entre les deux langues dans l'espace vectoriel.

### Index FAISS : `IndexFlatL2`

Recherche exacte par distance euclidienne. Avec seulement ~5000 vecteurs, la recherche exacte est quasi instantanee. Un index approximatif (IVF, HNSW) ne serait utile qu'a partir de centaines de milliers de vecteurs.

### Modele LLM : Groq (llama3-70b-8192)

Groq offre une inference tres rapide. Le modele 70B est plus capable pour formuler des recommandations argumentees que le 8B.

---

## Questions de reflexion - Sujet A

### Q1. Comment convertir chaque ligne CSV en texte a embedder ?

On construit un paragraphe par film qui combine : titre, annee, genres, note, duree et synopsis. Le synopsis est le contenu le plus riche semantiquement, mais le titre et les genres sont essentiels pour les requetes du type "un film comme Inception" ou "je cherche un thriller". La note et l'annee permettent de capter les requetes avec des criteres de filtre ("bien note", "recent").

Les colonnes `budget`, `revenue`, `homepage` et `production_companies` ne sont pas incluses dans le texte car elles n'apportent rien a la recherche semantique d'un utilisateur.

### Q2. Comment extraire la colonne `genres` au format JSON imbrique ?

On utilise `json.loads()` pour parser la string JSON en liste de dicts Python, puis on extrait le champ `"name"` de chaque element. Un try/except gere les cas ou le JSON est malforme ou absent. Resultat : `"Action, Adventure, Fantasy"`.

### Q3. Strategie pour ne pas relancer l'indexation a chaque test ?

L'index FAISS et les metadonnees sont sauvegardes sur disque (`faiss_index/films.index` et `faiss_index/films.json`). Le script `rag.py` recharge ces fichiers directement. On ne relance `indexation.py` que si les donnees sources changent.

### Q4. Comment guider le LLM pour des recommandations pertinentes ?

Le prompt systeme definit le LLM comme un expert cinema qui doit :
- Baser ses recommandations uniquement sur les films presents dans le contexte fourni
- Argumenter chaque recommandation (pourquoi ce film correspond a la demande)
- Citer le titre, l'annee et la note de chaque film recommande
- Ne jamais inventer un film qui n'est pas dans le contexte

### Q5. Que faire si l'utilisateur demande un film tres recent (2024) ?

Le systeme doit le signaler honnetement : la base de donnees contient des films jusqu'a ~2017. Si aucun resultat pertinent n'est trouve, le LLM doit indiquer qu'il n'a pas cette information plutot que d'inventer. Le score de similarite peut aussi servir d'indicateur de confiance.
