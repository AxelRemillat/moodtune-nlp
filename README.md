# 🎧 MoodTune NLP

**Recommandation musicale sémantique :** décrivez ce que vous ressentez avec vos propres mots, l'application trouve les morceaux qui correspondent.

Projet NLP du cours *Big Data & Data Science* (ESME, ING5 A3MSI, 2026-27, Sarah Malaeb). Le projet traite le thème 4 : recommandation de musique selon l'humeur.
Groupe 2 : Kelly Abendjock · Magaret Ngwanya · Axel Remillat · Lucas San Frutos

---

## ▶️ Tester l'application

- **En ligne :** https://moodtune-nlp-npznqurj7ftfhhs4a4bodf.streamlit.app/ (si l’appli est en veille, cliquer sur « Yes, get this app back up! » et attendre ~1 min)
- **En local (Windows) :** double-cliquer sur `LANCER_DEMO.bat`. Le premier lancement prend environ 5 min (installation et téléchargement du modèle). Les lancements suivants prennent environ 30 s et fonctionnent hors ligne.
- **En local (manuel) :**
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```

## 🧠 Fonctionnement

```
Questionnaire  ──►  Encodage SBERT  ──►  Cosinus + score pondéré  ──►  Top 3 expliqué + graphiques
(texte, cases,      (3 requêtes,          (humeur · contexte ·          (radar, Top 10,
 Likert → JSON)      384 dimensions)        genre)                        carte 2D PCA)
```

1. **Questionnaire mixte :** texte libre, question guidée, cases à cocher et deux échelles de Likert (énergie, paroles). Les réponses sont sauvegardées dans `outputs/last_user_response.json`.
2. **Prétraitement léger :** l'application retire seulement les espaces en trop. Les stopwords sont conservés, car « **sans** paroles » ≠ « paroles ».
3. **Likert → texte :** une énergie de 1/5 devient « très calme, lent, apaisant », pour que le modèle la comprenne.
4. **Embeddings SBERT :** le modèle `paraphrase-multilingual-MiniLM-L12-v2` est multilingue (FR et EN) et produit des vecteurs de 384 dimensions.
5. **Score global pondéré :**

   $$Score = \frac{W_{mood} S_{mood} + W_{context} S_{context} + W_{genre} S_{genre}}{W_{mood} + W_{context} + W_{genre}}$$

   Poids par défaut : 0,5 / 0,3 / 0,2. On peut les régler dans la barre latérale.
6. **Explicabilité :** chaque recommandation indique le bloc qui a le plus compté dans son score.

## 📁 Structure

```
moodtune-nlp/
├── app.py                              # Interface Streamlit (questionnaire, résultats, méthodologie)
├── src/
│   └── nlp_engine.py                   # Moteur NLP : embeddings, cosinus, score, PCA, familles de genres
├── data/
│   └── tracks_dataset.csv              # 75 morceaux décrits en texte (8 colonnes)
├── MoodTune_NLP_Documentation.ipynb    # Documentation pas à pas + comparaison TF-IDF vs SBERT
├── docs/                               # Support de soutenance et script oral
├── .streamlit/config.toml              # Thème de l'application
├── requirements.txt
└── LANCER_DEMO.bat                     # Lancement en un double-clic sous Windows
```

## 📊 Jeu de données

| Colonne | Exemple (T001) |
|---|---|
| Title / Artist | Midnight Study Session / Nujabes Tribute |
| Genre | Lo-Fi Hip Hop |
| Mood | Calme, apaisant, mélancolique doux, nostalgique |
| Context | Travail, étude, lecture tardive, concentration profonde |
| Description, Keywords | Texte libre décrivant l'ambiance |

Le catalogue compte 75 morceaux répartis en 5 ambiances de 15 (concentration, sport, soirée, mélancolie, bonne humeur) et 71 genres, regroupés en 8 familles pour les graphiques.

## 🔬 Pourquoi SBERT plutôt que TF-IDF ? (test réel, voir le notebook)

| Requête | TF-IDF | SBERT |
|---|---|---|
| « je dois bosser tard le soir sans être dérangé » | 13 % → indie pop sans rapport | 51 % → *Midnight Study Session* (lo-fi) |
| « I need hype music to lift heavy at the gym » | 0 % (aucun mot commun) | 55 % → *Break the Chains* (metal) |

## ⚠️ Limites et pistes

- Le catalogue est rédigé à la main, donc les descriptions restent subjectives. Il n'y a pas d'analyse audio.
- Les scores cosinus restent resserrés (environ 20 à 80 %) : c'est surtout le **classement** qui compte.
- Pistes : brancher l'API Spotify, ajouter un retour « j'aime / j'aime pas » pour ajuster les poids, utiliser un index FAISS pour les grands catalogues.

## 🛠️ Stack

Python · Sentence-Transformers · PyTorch (CPU) · scikit-learn · Pandas / NumPy · Streamlit · Plotly
