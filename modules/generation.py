"""
Module generation — Étape 6 du pipeline RAG

Responsabilité : prendre la question de l'utilisateur et les chunks
pertinents trouvés par la recherche vectorielle, puis demander au LLM
de générer une réponse argumentée.

Comment ça marche ?
    1. On construit un prompt système qui définit le rôle du LLM
       (expert cinéma) et ses contraintes (ne pas inventer, citer les
       sources, etc.)
    2. On assemble les chunks pertinents dans un "contexte" que le LLM
       reçoit avec la question
    3. Le LLM génère une réponse basée UNIQUEMENT sur ce contexte

Le prompt système est crucial :
    C'est lui qui contrôle le comportement du LLM. Sans instructions
    claires, le LLM pourrait inventer des films, ignorer les sources,
    ou répondre hors sujet. Le prompt doit :
    - Définir le rôle (expert cinéma)
    - Interdire l'invention (ne répondre que sur le contexte)
    - Exiger la citation des sources (titre, année, note)
    - Définir le format de la réponse

Questions à se poser (TP) :
    - Les chunks sont numérotés dans le prompt pour que le LLM puisse
      les référencer dans sa réponse
    - On inclut les métadonnées (titre, note, année) dans le contexte
      pour que le LLM puisse les citer
    - On limite le nombre de chunks envoyés pour ne pas dépasser la
      taille maximale du contexte du modèle (8192 tokens)
    - Le modèle llama-3.3-70b-versatile est plus capable que le 8b
      pour formuler des recommandations argumentées
"""

import os

from dotenv import load_dotenv
from groq import Groq


def construire_prompt_systeme() -> str:
    """
    Retourne le prompt système qui définit le comportement du LLM.

    Ce prompt est envoyé UNE FOIS au début de la conversation.
    Il cadre le LLM pour qu'il se comporte en expert cinéma
    et respecte les contraintes du TP.
    """
    return """Tu es un expert en cinéma et recommandation de films.

RÈGLES STRICTES :
- Tu ne recommandes QUE des films présents dans le contexte fourni ci-dessous.
- Tu ne dois JAMAIS inventer un film, un titre, une note ou un synopsis.
- Si aucun film du contexte ne correspond à la demande, dis-le honnêtement.
- La base de données contient des films jusqu'à environ 2017. Si l'utilisateur
  demande un film très récent, signale que ta base ne couvre pas cette période.

FORMAT DE RÉPONSE :
- Pour chaque film recommandé, cite : le titre, l'année et la note sur 10.
- Explique en 1-2 phrases pourquoi ce film correspond à la demande.
- À la fin, indique les sources (numéros des chunks utilisés).

LANGUE :
- Réponds toujours en français, même si les synopsis sont en anglais."""


def generer_reponse(question: str, chunks_pertinents: list[dict]) -> str:
    """
    Génère une réponse en utilisant les chunks comme contexte.

    Le LLM reçoit :
    - Le prompt système (rôle + contraintes)
    - Le contexte (les chunks pertinents numérotés avec leurs métadonnées)
    - La question de l'utilisateur

    Args:
        question: la question de l'utilisateur
        chunks_pertinents: les résultats de la recherche vectorielle,
                           chaque dict contient "contenu", "metadata", "score"

    Returns:
        La réponse générée par le LLM
    """
    # Charger la clé API depuis le fichier .env
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY manquant dans .env")

    client = Groq(api_key=api_key)

    # Assembler le contexte à partir des chunks
    # On numérote chaque chunk et on inclut les métadonnées
    # pour que le LLM puisse citer titre, année et note
    contexte = assembler_contexte(chunks_pertinents)

    # Construire le message utilisateur avec le contexte + la question
    message_utilisateur = (
        f"CONTEXTE (films de la base de données) :\n"
        f"{contexte}\n\n"
        f"QUESTION DE L'UTILISATEUR :\n{question}"
    )

    # Appel à l'API Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            {"role": "user", "content": message_utilisateur},
        ],
        temperature=0.7,  # Un peu de créativité pour les recommandations
        max_tokens=1024,
    )

    return response.choices[0].message.content


def assembler_contexte(chunks_pertinents: list[dict]) -> str:
    """
    Formate les chunks en un texte structuré pour le LLM.

    Chaque chunk est numéroté et accompagné de ses métadonnées.
    Le LLM peut ainsi écrire "le film n°2" ou citer le titre
    et la note directement depuis les métadonnées.

    Args:
        chunks_pertinents: liste de dicts avec contenu, metadata, score

    Returns:
        Texte formaté prêt à être injecté dans le prompt
    """
    parties = []
    for i, chunk in enumerate(chunks_pertinents, 1):
        meta = chunk["metadata"]
        partie = (
            f"--- Film n°{i} ---\n"
            f"Titre : {meta['titre']} ({meta['annee']})\n"
            f"Genres : {meta['genres']}\n"
            f"Note : {meta['note']}/10\n"
            f"Langue originale : {meta['langue']}\n"
            f"Texte : {chunk['contenu']}\n"
        )
        parties.append(partie)

    return "\n".join(parties)
