import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import re

# Regroupement des 71 genres en 8 grandes familles (lisibilité des graphiques)
FAMILLES = [
    ("Musiques du monde", ["afro", "reggae", "latin", "salsa", "celtic", "reggaeton"]),
    ("Calme / Instrumental", ["lo-fi", "chill", "ambient", "piano", "néo-classique", "cello", "downtempo", "slowcore", "trip-hop", "dream"]),
    ("Orchestral / Cinéma", ["orchestral", "post-rock", "soundtrack"]),
    ("Électro / Dance", ["house", "techno", "electro", "edm", "trance", "garage", "drum", "dubstep", "synthwave", "french touch", "phonk", "dance"]),
    ("Jazz / Soul / Funk", ["jazz", "soul", "funk", "r&b", "bossa", "blues", "disco"]),
    ("Rap / Hip-Hop", ["rap", "hip-hop", "trap"]),
    ("Rock / Metal", ["rock", "metal", "punk"]),
    ("Pop / Folk / Chanson", ["pop", "folk", "chanson", "acoustic"]),
]


def genre_family(genre: str) -> str:
    """Associe un genre précis à une grande famille musicale."""
    g = str(genre).lower()
    for family, keys in FAMILLES:
        if any(k in g for k in keys):
            return family
    return "Autres"

class MusicSemanticEngine:
    """
    Moteur d'analyse sémantique pour la recommandation musicale (Thème 4).
    Utilise SBERT (embeddings contextuels) et la similarité cosinus multi-blocs.
    """
    def __init__(self, dataset_path="data/tracks_dataset.csv", model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.df = pd.read_csv(dataset_path)
        self.df["Famille"] = self.df["Genre"].apply(genre_family)
        self.model = SentenceTransformer(model_name)
        self._prepare_reference_embeddings()

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Nettoyage et normalisation légère du texte (préserve le contexte pour SBERT)."""
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _prepare_reference_embeddings(self):
        """Pré-calcule les embeddings sémantiques des 3 blocs de référence du dataset."""
        # Bloc 1 : Humeur & Description sémantique riche
        mood_texts = (self.df["Mood"] + ". " + self.df["Description"]).apply(self.preprocess_text).tolist()
        # Bloc 2 : Contexte d'écoute / Activité
        context_texts = (self.df["Context"] + ". " + self.df["Keywords"]).apply(self.preprocess_text).tolist()
        # Bloc 3 : Genre & Style musical
        genre_texts = (self.df["Genre"] + " - " + self.df["Keywords"]).apply(self.preprocess_text).tolist()

        self.mood_embeddings = self.model.encode(mood_texts, convert_to_tensor=True, normalize_embeddings=True)
        self.context_embeddings = self.model.encode(context_texts, convert_to_tensor=True, normalize_embeddings=True)
        self.genre_embeddings = self.model.encode(genre_texts, convert_to_tensor=True, normalize_embeddings=True)

        # Embedding global combiné (utile pour la projection 2D PCA)
        full_texts = (self.df["Genre"] + " | " + self.df["Mood"] + " | " + self.df["Description"]).tolist()
        self.full_embeddings_np = self.model.encode(full_texts, normalize_embeddings=True)

    def recommend(self, user_profile: dict, top_k: int = 3):
        """
        Calcule les scores de similarité sémantique par bloc, applique la formule de scoring
        pondérée et génère des recommandations explicables.
        """
        # Construction des requêtes sémantiques utilisateur par bloc
        free_mood = self.preprocess_text(user_profile.get("mood_text", ""))
        style_pref = self.preprocess_text(user_profile.get("style_text", ""))
        contexts = ", ".join(user_profile.get("contexts", []))
        genres = ", ".join(user_profile.get("genres", []))

        # Ajout des nuances issues des échelles de Likert dans le texte sémantique
        energy_level = user_profile.get("energy_likert", 3)
        vocals_pref = user_profile.get("vocals_likert", 3)

        energy_desc = {
            1: "très calme, lent, apaisant, doux",
            2: "calme, posé, tranquille",
            3: "énergie modérée, équilibrée",
            4: "dynamique, rythmé, entraînant",
            5: "très énergique, intense, puissant, rapide"
        }.get(energy_level, "")

        vocals_desc = {
            1: "strictement instrumental, sans paroles",
            2: "plutôt instrumental, voix discrète",
            3: "peu importe les paroles",
            4: "avec chant et mélodie vocale",
            5: "paroles importantes, chant expressif"
        }.get(vocals_pref, "")

        query_mood = f"{free_mood}. {style_pref}. Ambiance {energy_desc}, {vocals_desc}".strip()
        query_context = f"Contexte d'écoute : {contexts}. {free_mood}".strip()
        query_genre = f"Style musical : {genres}. {style_pref}".strip()

        # Encodage des requêtes utilisateur
        q_mood_emb = self.model.encode(query_mood, convert_to_tensor=True, normalize_embeddings=True)
        q_ctx_emb = self.model.encode(query_context, convert_to_tensor=True, normalize_embeddings=True)
        q_gen_emb = self.model.encode(query_genre, convert_to_tensor=True, normalize_embeddings=True)

        # Calcul des similarités cosinus (ramenées entre 0 et 1)
        sim_mood = np.clip(util.cos_sim(q_mood_emb, self.mood_embeddings)[0].cpu().numpy(), 0, 1)
        sim_context = np.clip(util.cos_sim(q_ctx_emb, self.context_embeddings)[0].cpu().numpy(), 0, 1)
        sim_genre = np.clip(util.cos_sim(q_gen_emb, self.genre_embeddings)[0].cpu().numpy(), 0, 1)

        # Pondération dynamique selon l'importance accordée par l'utilisateur
        w_mood = user_profile.get("w_mood", 0.50)
        w_context = user_profile.get("w_context", 0.30)
        w_genre = user_profile.get("w_genre", 0.20)
        w_sum = w_mood + w_context + w_genre

        global_score = (w_mood * sim_mood + w_context * sim_context + w_genre * sim_genre) / w_sum

        results_df = self.df.copy()
        # Passage en float64 avant l'arrondi (évite l'affichage 75.400002 du float32)
        results_df["Score_Mood"] = np.round(sim_mood.astype(float) * 100, 1)
        results_df["Score_Context"] = np.round(sim_context.astype(float) * 100, 1)
        results_df["Score_Genre"] = np.round(sim_genre.astype(float) * 100, 1)
        results_df["Score_Global"] = np.round(global_score.astype(float) * 100, 1)

        # Génération des explications transparentes basées sur les scores NLP
        explanations = []
        for _, row in results_df.iterrows():
            scores_dict = {
                "l'atmosphère et l'humeur recherchées": row["Score_Mood"],
                "le contexte d'écoute": row["Score_Context"],
                "l'univers musical et le genre": row["Score_Genre"]
            }
            best_criterion = max(scores_dict, key=scores_dict.get)
            best_val = scores_dict[best_criterion]
            exp = (
                f"Affinité globale de **{row['Score_Global']}%**, portée principalement par "
                f"**{best_criterion}** ({best_val}%). "
                f"Ce morceau *{row['Genre']}* offre une ambiance *{row['Mood'].lower()}* "
                f"adaptée à : *{row['Context'].lower()}*."
            )
            explanations.append(exp)

        results_df["Explanation"] = explanations
        results_df = results_df.sort_values(by="Score_Global", ascending=False).reset_index(drop=True)

        # Projection 2D (PCA) pour visualiser la position de l'utilisateur vs le catalogue
        q_full_emb = self.model.encode([f"{query_mood} | {query_context} | {query_genre}"], normalize_embeddings=True)
        all_vectors = np.vstack([self.full_embeddings_np, q_full_emb])
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(all_vectors)

        tracks_coords = coords[:-1]
        user_coord = coords[-1]

        return results_df, tracks_coords, user_coord
