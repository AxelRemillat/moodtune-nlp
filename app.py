import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from src.nlp_engine import MusicSemanticEngine

# Chemins absolus : l'appli marche même si on la lance depuis un autre dossier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mêmes couleurs de familles sur tous les graphiques
FAMILY_COLORS = {
    "Calme / Instrumental": "#7C3AED", "Électro / Dance": "#06B6D4", "Jazz / Soul / Funk": "#F59E0B",
    "Rock / Metal": "#EF4444", "Musiques du monde": "#10B981", "Pop / Folk / Chanson": "#EC4899",
    "Rap / Hip-Hop": "#374151", "Orchestral / Cinéma": "#3B82F6",
}

st.set_page_config(
    page_title="MoodTune NLP — Recommandation Musicale Sémantique",
    page_icon="🎧",
    layout="wide"
)

@st.cache_resource
def load_engine():
    return MusicSemanticEngine(dataset_path=os.path.join(BASE_DIR, "data", "tracks_dataset.csv"))

engine = load_engine()

st.title("🎧 MoodTune NLP : Recommandation Musicale Contextuelle")
st.markdown(
    "Système intelligent d'analyse sémantique (**SBERT + Similarité Cosinus**) associant vos émotions, "
    "votre contexte et vos préférences musicales à notre catalogue de référence."
)

# Barre latérale : Paramètres de pondération & Exploration du dataset
with st.sidebar:
    st.header("⚙️ Pondération du Scoring")
    st.caption("Ajustez les poids de la formule d'affinité sémantique :")
    w_mood = st.slider("Poids Humeur & Émotion ($W_{mood}$)", 0.1, 1.0, 0.5, 0.05)
    w_context = st.slider("Poids Contexte / Activité ($W_{context}$)", 0.1, 1.0, 0.3, 0.05)
    w_genre = st.slider("Poids Genre & Style ($W_{genre}$)", 0.1, 1.0, 0.2, 0.05)

    st.divider()
    st.subheader("📚 Jeu de données de référence")
    st.metric("Morceaux référencés", len(engine.df))
    st.metric("Genres distincts", engine.df["Genre"].nunique())
    st.metric("Familles musicales", engine.df["Famille"].nunique())
    if st.checkbox("Afficher un aperçu du dataset"):
        st.dataframe(engine.df[["TrackID", "Title", "Artist", "Genre", "Famille", "Mood"]].head(10), use_container_width=True)

# Onglets principaux
tab_quiz, tab_results, tab_methodology = st.tabs([
    "📝 1. Questionnaire Utilisateur",
    "🎯 2. Recommandations & Visualisations",
    "🧠 3. Architecture & Méthodologie NLP"
])

with tab_quiz:
    st.subheader("Décrivez votre état d'esprit et vos envies musicales")

    with st.form("music_questionnaire"):
        col1, col2 = st.columns(2)

        with col1:
            mood_text = st.text_area(
                "1. Décrivez librement l'ambiance, l'émotion ou l'atmosphère recherchée (Texte libre) *",
                value="Je recherche une musique instrumentale relaxante qui m'aide à me concentrer pendant mes études, avec une atmosphère calme et apaisante.",
                height=110,
                help="Plus votre description est riche, plus l'analyse sémantique SBERT sera précise."
            )

            style_text = st.text_input(
                "2. Avez-vous des préférences d'instruments, de rythme ou d'artistes similaires ? (Question guidée)",
                value="Du piano doux ou des sonorités lo-fi feutrées, sans paroles distrayantes."
            )

            contexts = st.multiselect(
                "3. Dans quel contexte allez-vous écouter cette musique ? (Cases à cocher)",
                options=[
                    "Travail / Étude / Concentration",
                    "Sport / Running / Musculation",
                    "Soirée / Fête / Danse",
                    "Détente / Méditation / Sommeil",
                    "Trajet / Voyage / Conduite",
                    "Lecture / Écriture / Créativité",
                    "Introspection / Pluie / Solitude"
                ],
                default=["Travail / Étude / Concentration", "Lecture / Écriture / Créativité"]
            )

        with col2:
            genres = st.multiselect(
                "4. Quels univers musicaux appréciez-vous particulièrement ? (Choix multiple)",
                options=sorted(engine.df["Genre"].unique().tolist()),
                default=["Lo-Fi Hip Hop", "Ambient", "Néo-Classique"]
            )

            energy_likert = st.select_slider(
                "5. Échelle d'énergie souhaitée (Likert 1 à 5)",
                options=[1, 2, 3, 4, 5],
                value=2,
                format_func=lambda x: {
                    1: "1 - Très calme / Méditatif",
                    2: "2 - Posé / Doux",
                    3: "3 - Équilibré",
                    4: "4 - Dynamique / Entraînant",
                    5: "5 - Explosif / Intense"
                }[x]
            )

            vocals_likert = st.select_slider(
                "6. Importance des paroles et du chant (Likert 1 à 5)",
                options=[1, 2, 3, 4, 5],
                value=1,
                format_func=lambda x: {
                    1: "1 - Strictement instrumental",
                    2: "2 - Plutôt instrumental",
                    3: "3 - Indifférent",
                    4: "4 - Chant apprécié",
                    5: "5 - Paroles essentielles"
                }[x]
            )

        submitted = st.form_submit_button("🚀 Analyser sémantiquement et recommander", use_container_width=True)

    if submitted:
        if not mood_text.strip():
            st.error("Veuillez saisir au moins une description dans le champ n°1 pour lancer l'analyse sémantique.")
        else:
            user_profile = {
                "timestamp": datetime.now().isoformat(),
                "mood_text": mood_text,
                "style_text": style_text,
                "contexts": contexts,
                "genres": genres,
                "energy_likert": energy_likert,
                "vocals_likert": vocals_likert,
                "w_mood": w_mood,
                "w_context": w_context,
                "w_genre": w_genre
            }

            # Sauvegarde structurée des réponses (exigence du cahier des charges)
            out_dir = os.path.join(BASE_DIR, "outputs")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "last_user_response.json"), "w", encoding="utf-8") as f:
                json.dump(user_profile, f, ensure_ascii=False, indent=2)

            results_df, tracks_coords, user_coord = engine.recommend(user_profile)
            st.session_state["results_df"] = results_df
            st.session_state["tracks_coords"] = tracks_coords
            st.session_state["user_coord"] = user_coord
            st.session_state["user_profile"] = user_profile
            st.success("✅ Analyse terminée ! Consultez l'onglet **2. Recommandations & Visualisations**.")

with tab_results:
    if "results_df" not in st.session_state:
        st.info("👈 Remplissez et validez d'abord le questionnaire dans l'onglet 1.")
    else:
        results_df = st.session_state["results_df"]
        user_profile = st.session_state["user_profile"]
        tracks_coords = st.session_state["tracks_coords"]
        user_coord = st.session_state["user_coord"]

        # Synthèse du profil utilisateur
        st.subheader("👤 Synthèse de votre profil sémantique")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meilleur Match", f"{results_df.iloc[0]['Score_Global']}%", results_df.iloc[0]["Title"])
        c2.metric("Niveau d'énergie", f"{user_profile['energy_likert']} / 5")
        c3.metric("Préférence vocale", f"{user_profile['vocals_likert']} / 5")
        c4.metric("Contextes ciblés", len(user_profile["contexts"]))

        st.divider()

        # Top 3 Recommandations explicables
        st.subheader("🏆 Top 3 des morceaux recommandés")
        top3 = results_df.head(3)
        cols = st.columns(3)
        medals = ["🥇 N°1", "🥈 N°2", "🥉 N°3"]

        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                st.markdown(f"### {medals[idx]} — {row['Title']}")
                st.caption(f"Artiste : **{row['Artist']}** | Genre : **{row['Genre']}**")
                st.progress(int(row["Score_Global"]))
                st.markdown(row["Explanation"])
                with st.expander("Détails sémantiques du morceau"):
                    st.write(f"**Description :** {row['Description']}")
                    st.write(f"- Score Humeur : `{row['Score_Mood']}%`")
                    st.write(f"- Score Contexte : `{row['Score_Context']}%`")
                    st.write(f"- Score Genre : `{row['Score_Genre']}%`")

        st.divider()

        # Visualisations interactives
        st.subheader("📊 Visualisations de l'analyse sémantique")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # 1. Diagramme Radar du Top 3 sur les 3 dimensions sémantiques
            fig_radar = go.Figure()
            categories = ["Humeur & Émotion", "Contexte d'écoute", "Genre & Style"]
            for idx, (_, row) in enumerate(top3.iterrows()):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row["Score_Mood"], row["Score_Context"], row["Score_Genre"]],
                    theta=categories,
                    fill="toself",
                    name=f"#{idx+1} {row['Title']}"
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Profil d'affinité par bloc sémantique (Top 3)",
                height=420
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with viz_col2:
            # 2. Diagramme en barres du Top 10
            top10 = results_df.head(10).sort_values(by="Score_Global", ascending=True)
            fig_bar = px.bar(
                top10,
                x="Score_Global",
                y="Title",
                color="Famille",
                orientation="h",
                hover_data=["Genre"],
                color_discrete_map=FAMILY_COLORS,
                text="Score_Global",
                title="Classement sémantique du Top 10 (%)",
                labels={"Score_Global": "Score d'affinité (%)", "Title": "Morceau"}
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}")
            fig_bar.update_layout(height=420)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 3. Carte sémantique 2D (Projection PCA des embeddings SBERT)
        st.subheader("🗺️ Carte sémantique 2D (Projection PCA des embeddings)")
        plot_df = engine.df.copy()
        plot_df["PCA_1"] = tracks_coords[:, 0]
        plot_df["PCA_2"] = tracks_coords[:, 1]
        plot_df["Score"] = results_df.set_index("TrackID").loc[plot_df["TrackID"], "Score_Global"].values

        fig_pca = px.scatter(
            plot_df,
            x="PCA_1",
            y="PCA_2",
            color="Famille",
            size="Score",
            hover_data=["Title", "Artist", "Genre", "Mood", "Score"],
            color_discrete_map=FAMILY_COLORS,
            title="Position de votre requête dans l'espace vectoriel des morceaux"
        )
        fig_pca.add_trace(go.Scatter(
            x=[user_coord[0]],
            y=[user_coord[1]],
            mode="markers+text",
            marker=dict(size=18, color="red", symbol="star"),
            text=["VOTRE REQUÊTE"],
            textposition="top center",
            name="Profil Utilisateur"
        ))
        st.plotly_chart(fig_pca, use_container_width=True)

        # Tableau complet
        with st.expander("📋 Voir le classement complet des 75 morceaux"):
            st.dataframe(
                results_df[["TrackID", "Title", "Artist", "Genre", "Score_Global", "Score_Mood", "Score_Context", "Score_Genre"]],
                use_container_width=True
            )

with tab_methodology:
    st.subheader("Justification technique de l'architecture NLP")
    st.markdown(r"""
    ### 1. Pourquoi SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) ?
    * Contrairement à une recherche par mots-clés (TF-IDF / BM25), **Sentence-BERT** encode des phrases entières en vecteurs denses de 384 dimensions capturant le sens contextuel (synonymes, émotions implicites, nuances en français et en anglais).
    * Exemple : une requête mentionnant *"réviser au calme la nuit"* sera rapprochée sémantiquement de *"concentration profonde"* et *"atmosphère nocturne apaisante"* même sans mots identiques.

    ### 2. Formule de Scoring Sémantique Multi-Blocs
    Chaque morceau est évalué sur trois dimensions distinctes via la **similarité cosinus** :

    $$\text{Score}_{\text{global}} = \frac{W_{\text{mood}} \cdot S_{\text{mood}} + W_{\text{context}} \cdot S_{\text{context}} + W_{\text{genre}} \cdot S_{\text{genre}}}{W_{\text{mood}} + W_{\text{context}} + W_{\text{genre}}}$$

    ### 3. Pipeline en 5 étapes
    1. **Questionnaire** : texte libre, question guidée, cases à cocher, échelles de Likert → sauvegarde JSON.
    2. **Prétraitement léger** : nettoyage des espaces uniquement (pas de stopwords ni de lemmatisation, SBERT a besoin de la phrase complète).
    3. **Likert → texte** : « énergie 2/5 » devient « calme, posé, tranquille » pour être compris par le modèle.
    4. **Embeddings + cosinus** : 3 requêtes (humeur, contexte, genre) comparées aux 3 blocs de chaque morceau.
    5. **Score pondéré + Top 3 expliqué** : le bloc qui contribue le plus est affiché comme justification.

    ### 4. Limites connues
    * Catalogue de 75 morceaux rédigé par l'équipe (descriptions subjectives).
    * Pas d'audio analysé : seul le texte décrivant les morceaux est utilisé.
    * Les scores cosinus restent dans une plage resserrée (≈ 20 à 80 %), on compare donc surtout le **classement**.
    """)
