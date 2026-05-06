# RAG — Recommandation de Films (Sujet A)

Systeme RAG (Retrieval-Augmented Generation) de recommandation de films base sur le dataset TMDB 5000.

## Comment lancer le projet

### Prerequis

- Python 3.10+ (teste avec Python 3.13.2)
- Un compte Groq avec une cle API
- Le dataset TMDB (`tmdb_5000_movies.csv`) telecharge depuis Kaggle

### Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Configuration

Creer un fichier `.env` a la racine :

```env
GROQ_API_KEY=votre_cle_ici
```

Placer le fichier `tmdb_5000_movies.csv` dans le dossier `data/`.

### Lancement

```bash
# Phase 1 — Indexation (une seule fois, ~22 minutes)
python indexation.py

# Phase 2 — Questions-reponses (interactif)
python rag.py
```

### Tests

```bash
# Verifier que les dependances et l'API fonctionnent
python test_installation.py

# Tester la recherche vectorielle + la generation LLM
python test_recherche.py
```

---

## Architecture du projet

```
RAG-tp/
  data/
    tmdb_5000_movies.csv       <- dataset TMDB (5000 films)
  modules/
    chargement.py              <- Etape 1 : chargement et nettoyage du CSV
    chunking.py                <- Etape 2 : decoupage des textes en chunks
    embedding.py               <- Etape 3 : transformation texte -> vecteurs
    faiss_index.py             <- Etape 4 : creation et persistance de l'index FAISS
    recherche.py               <- Etape 5 : recherche vectorielle (top-k)
    generation.py              <- Etape 6 : generation de reponse avec Groq
  index_data/                  <- fichiers d'index generes (non commites)
  indexation.py                <- script principal phase 1 (etapes 1-4)
  rag.py                       <- script principal phase 2 (etapes 5-7)
  test_installation.py         <- verification des dependances
  test_recherche.py            <- test recherche + generation
```

### Pipeline

```
PHASE 1 : INDEXATION (indexation.py — une seule fois)

  tmdb_5000_movies.csv
        |
        v
  [chargement.py] Chargement CSV + extraction genres JSON
        |                         4803 films -> 4799 retenus (sans synopsis exclus)
        v
  [chunking.py]   Decoupage en chunks (500 car. max, overlap 50)
        |                         4799 documents -> 6726 chunks
        v
  [embedding.py]  Encodage en vecteurs 768D (modele multilingue)
        |                         6726 chunks -> 6726 vecteurs
        v
  [faiss_index.py] Stockage dans l'index FAISS + sauvegarde disque
                          -> index_data/films.index + films.json


PHASE 2 : INTERROGATION (rag.py — a chaque question)

  Question utilisateur (en francais)
        |
        v
  [embedding.py]  Encodage de la question en vecteur 768D
        |
        v
  [recherche.py]  Recherche des 5 chunks les plus proches (FAISS)
        |                         + filtre par langue si active
        v
  [generation.py] Envoi question + chunks au LLM Groq
        |                         prompt systeme : expert cinema, citer sources
        v
  Reponse avec titres, notes et sources
```

---

## Choix techniques

### Modele d'embedding : `paraphrase-multilingual-mpnet-base-v2`

Choisi car les synopsis sont en anglais mais les questions de l'utilisateur sont en francais. Ce modele multilingue projette les deux langues dans le meme espace vectoriel : une question en francais ("thriller psychologique") sera proche d'un synopsis en anglais ("psychological thriller").

Alternative consideree : `all-mpnet-base-v2` (anglais uniquement) — ecarte car les questions sont en francais.

### Chunking : 500 caracteres, overlap 50

- **Taille 500 caracteres** : les synopsis de films font en moyenne 200-400 caracteres. La plupart tiennent en un seul chunk. Les plus longs sont decoupes proprement.
- **Overlap 50 caracteres** : les 50 derniers caracteres d'un chunk sont repetes au debut du suivant pour ne pas couper une idee en plein milieu.
- **Coupure intelligente** : le chunker cherche une frontiere naturelle (fin de phrase, paragraphe, espace) plutot que de couper au milieu d'un mot.
- Resultat : 4799 films -> 6726 chunks (1233 films decoupes en plusieurs chunks).

### Index FAISS : `IndexFlatL2`

- **Recherche exacte** par distance euclidienne (L2).
- Avec ~6700 vecteurs, la recherche est quasi instantanee (<10ms). Un index approximatif (IVF, HNSW) ne serait utile qu'a partir de centaines de milliers de vecteurs.
- Alternative : `IndexFlatIP` (produit scalaire) avec normalisation des vecteurs pour une vraie similarite cosinus. Non retenu car L2 suffit pour classer les resultats par pertinence.

### Modele LLM : Groq (`llama-3.3-70b-versatile`)

- Groq offre une inference tres rapide grace a ses puces LPU.
- Le modele `llama-3.3-70b-versatile` est plus capable que les modeles 8B pour formuler des recommandations argumentees et respecter les contraintes du prompt.
- Temperature 0.7 : un peu de creativite pour les recommandations tout en restant coherent.

### Filtre par langue

Le TP demande un filtre par langue originale du film. A ne pas confondre avec le modele multilingue :
- **Modele multilingue** = permet de poser des questions en francais sur des synopsis anglais (langue de la recherche)
- **Filtre par langue** = ne garder que les films dont la langue originale de production est le francais (langue du film)

Quand le filtre est actif, on cherche k=100 resultats dans FAISS (au lieu de 10) car les films francais ne representent que ~70/4799 films dans la base.

### Persistance

L'index FAISS et les metadonnees sont sauvegardes sur disque :
- `index_data/films.index` : les vecteurs (format binaire FAISS)
- `index_data/films.json` : les chunks + metadonnees (JSON lisible)

`rag.py` recharge ces fichiers directement. On ne relance `indexation.py` que si les donnees sources changent.

---

## Questions de reflexion — Sujet A

### Q1. Comment convertir chaque ligne CSV en texte a embedder ?

On construit un paragraphe par film qui combine : titre, annee, genres, note, duree et synopsis. Le synopsis est le contenu le plus riche semantiquement, mais le titre et les genres sont essentiels pour les requetes du type "un film comme Inception" ou "je cherche un thriller". La note et l'annee permettent de capter les requetes avec des criteres de filtre ("bien note", "recent").

Les colonnes `budget`, `revenue`, `homepage` et `production_companies` ne sont pas incluses car elles n'apportent rien a la recherche semantique d'un utilisateur.

### Q2. Comment extraire la colonne `genres` au format JSON imbrique ?

On utilise `json.loads()` pour parser la string JSON en liste de dicts Python, puis on extrait le champ `"name"` de chaque element. Un `try/except` gere les cas ou le JSON est malforme ou absent. Resultat : `"Action, Adventure, Fantasy"`.

### Q3. Strategie pour ne pas relancer l'indexation a chaque test ?

L'index FAISS et les metadonnees sont sauvegardes sur disque (`index_data/films.index` et `index_data/films.json`). Le script `rag.py` recharge ces fichiers au demarrage. On ne relance `indexation.py` que si les donnees sources changent. L'indexation prend ~22 minutes (encodage de 6726 chunks).

### Q4. Comment guider le LLM pour des recommandations pertinentes ?

Le prompt systeme definit le LLM comme un expert cinema qui doit :
- Baser ses recommandations uniquement sur les films presents dans le contexte fourni
- Argumenter chaque recommandation (pourquoi ce film correspond a la demande)
- Citer le titre, l'annee et la note de chaque film recommande
- Ne jamais inventer un film qui n'est pas dans le contexte
- Repondre en francais meme si les synopsis sont en anglais

### Q5. Que faire si l'utilisateur demande un film tres recent (2024) ?

Le systeme le signale honnetement : la base de donnees contient des films jusqu'a ~2017. Le prompt systeme instruit le LLM d'indiquer que sa base ne couvre pas cette periode plutot que d'inventer. Le score de similarite sert aussi d'indicateur : un score L2 eleve signifie que les resultats sont peu pertinents.

---

## Questions a se poser — Partie commune (etapes 3 a 7)

### Etape 3 — Embeddings

- **Choix du modele** : contenu en anglais, questions en francais -> modele multilingue obligatoire (`paraphrase-multilingual-mpnet-base-v2`, dimension 768).
- **Performance** : `batch_size=64` encode 64 textes en parallele au lieu de 1 par 1. `show_progress_bar=True` affiche la progression.
- **Verification** : on verifie que la dimension des vecteurs est bien 768 avec un `assert`.

### Etape 4 — Index FAISS

- **Ordre des chunks** : les chunks dans le JSON DOIVENT etre dans le meme ordre que les vecteurs dans FAISS. Le vecteur n.0 correspond au chunk n.0. Si l'ordre est different, les resultats de recherche pointeraient vers les mauvais films.
- **IndexFlatL2 vs IndexFlatIP** : L2 = distance euclidienne (petit = proche). IP = produit scalaire (grand = proche). Pour un score de similarite cosinus, il faudrait normaliser les vecteurs et utiliser IP. On garde L2 pour la simplicite.
- **Verification** : apres sauvegarde+rechargement, on verifie que `index.ntotal == len(chunks)`.

### Etape 5 — Recherche

- **Score L2** : c'est une distance, pas un pourcentage. Plus le score est PETIT, plus le resultat est pertinent. Un score de 0 = textes identiques.
- **Test avant LLM** : on teste la recherche sur 5 questions differentes (`test_recherche.py`) pour verifier que les resultats sont pertinents AVANT de brancher le LLM. Si les resultats sont aberrants, le probleme vient du chunking ou de l'embedding.

### Etape 6 — Generation

- **Format du contexte** : les chunks sont numerotes et accompagnes de leurs metadonnees (titre, annee, note, genres) pour que le LLM puisse les citer.
- **Taille du contexte** : on limite a 5 chunks pour ne pas depasser la fenetre de 8192 tokens du modele.
- **Forcer le LLM** : le prompt systeme dit explicitement "ne recommande QUE des films presents dans le contexte" et "ne jamais inventer".
