"""
Module chunking — Étape 2 du pipeline RAG

Responsabilité : découper un texte long en morceaux (chunks) de taille
contrôlée, avec un chevauchement (overlap) pour ne pas perdre le contexte
entre deux chunks consécutifs.

Pourquoi découper ?
    Le modèle d'embedding fonctionne mieux sur des textes courts et
    cohérents. Un texte trop long dilue l'information et rend le vecteur
    moins précis pour la recherche.

Pourquoi un chevauchement (overlap) ?
    Si on coupe brutalement à 500 caractères, on risque de couper une
    phrase ou une idée en plein milieu. L'overlap répète les derniers
    caractères du chunk N au début du chunk N+1, pour que le contexte
    ne soit jamais perdu.

Particularité du sujet Films :
    Les synopsis de films font en général 200-400 caractères. La plupart
    des documents tiendront en un seul chunk. Le chunker est quand même
    nécessaire car certains synopsis sont plus longs, et c'est une brique
    essentielle du pipeline RAG.
"""


def chunker(texte: str, taille_max: int = 500, overlap: int = 50) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Stratégie de coupure :
    1. Si le texte tient en un seul chunk → on le retourne tel quel
    2. Sinon, on avance par fenêtre de taille_max caractères
    3. À chaque fenêtre, on cherche une frontière naturelle (point, retour
       à la ligne) pour ne pas couper au milieu d'un mot ou d'une phrase
    4. Le chunk suivant commence "overlap" caractères avant la fin du
       précédent pour créer le chevauchement

    Args:
        texte: le texte à découper
        taille_max: nombre maximum de caractères par chunk
        overlap: nombre de caractères répétés entre deux chunks consécutifs

    Returns:
        Liste de chunks (strings)
    """
    # Cas simple : le texte tient en un seul chunk
    if len(texte) <= taille_max:
        return [texte]

    chunks = []
    debut = 0

    while debut < len(texte):
        fin = debut + taille_max

        # Si on n'est pas à la fin du texte, on cherche une coupure propre
        if fin < len(texte):
            fin = trouver_coupure(texte, debut, fin)

        chunk = texte[debut:fin].strip()
        if chunk:
            chunks.append(chunk)

        # Le prochain chunk commence "overlap" caractères avant la fin
        # pour créer le chevauchement.
        # On s'assure qu'on avance toujours d'au moins 1 caractère
        # pour éviter une boucle infinie
        nouveau_debut = fin - overlap
        if nouveau_debut <= debut:
            nouveau_debut = debut + 1
        debut = nouveau_debut

    return chunks


def trouver_coupure(texte: str, debut: int, fin: int) -> int:
    """
    Cherche la meilleure position de coupure dans la zone [debut, fin].

    On privilégie dans l'ordre :
    1. Un double retour à la ligne (séparation de paragraphe)
    2. Un point suivi d'un espace (fin de phrase)
    3. Un retour à la ligne simple
    4. Un espace (entre deux mots)
    5. Si rien trouvé → on coupe à fin (au milieu d'un mot, en dernier recours)

    Args:
        texte: le texte complet
        debut: début de la zone de recherche
        fin: fin de la zone (position max de coupure)

    Returns:
        Position de coupure choisie
    """
    # Chercher les séparateurs du plus propre au moins propre
    separateurs = ["\n\n", ". ", "\n", " "]

    for sep in separateurs:
        # rfind cherche la DERNIÈRE occurrence avant la limite
        # → on coupe le plus tard possible pour maximiser la taille du chunk
        position = texte.rfind(sep, debut, fin)
        if position > debut:
            # +len(sep) pour inclure le séparateur dans le chunk courant
            return position + len(sep)

    # Aucun séparateur trouvé → on coupe à la limite max
    return fin
