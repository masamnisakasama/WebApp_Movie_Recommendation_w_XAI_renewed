# app.py
# CPU/Threadの設定　安定した処理を実行するために設置
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import streamlit as st
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from lime_utils import BERTSimilarityExplainer

# -----------------------------
# TMDbのAPI設定
# -----------------------------
TMDB_API_KEY = st.secrets["tmd_api"]["api_key"]
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# -----------------------------
# ロード1回だけで高速化されて欲しいが
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embed_model()

# -----------------------------
# TMDb API 呼び出しキャッシュで高速化
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_movies(query: str, max_results: int = 10):
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "ja-JP",
        "include_adult": False,
        "page": 1,
    }
    res = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=30)
    res.raise_for_status()
    return res.json().get("results", [])[:max_results]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_details(movie_id: int):
    params = {"api_key": TMDB_API_KEY, "language": "ja-JP"}
    res = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params=params, timeout=30)
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_similar_movies(movie_id: int, limit: int = 20):
    params = {"api_key": TMDB_API_KEY, "language": "ja-JP", "page": 1}
    res = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/similar", params=params, timeout=30)
    res.raise_for_status()
    return res.json().get("results", [])[:limit]

# -----------------------------
# 埋め込みと類似度これもキャッシュで高速化
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_embeddings(descriptions):
    # Streamlitのハッシュ安定化のためtuple化
    if isinstance(descriptions, list):
        descriptions = tuple(descriptions)
    return model.encode(list(descriptions), convert_to_numpy=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def rank_similar_movies(selected_id: int, candidate_movies: list, candidate_embs: np.ndarray, top_k: int = 5):
    selected_desc = fetch_movie_details(selected_id).get("overview", "") or ""
    selected_emb = compute_embeddings([selected_desc])[0]

    scores = [cosine_sim(selected_emb, emb) for emb in candidate_embs]
    sorted_idx = np.argsort(scores)[::-1]

    top_movies  = [candidate_movies[i] for i in sorted_idx[:top_k]]
    top_scores  = [scores[i] for i in sorted_idx[:top_k]]
    rest_movies = [candidate_movies[i] for i in sorted_idx[top_k:]]
    rest_scores = [scores[i] for i in sorted_idx[top_k:]]

    return (top_movies, top_scores), (rest_movies, rest_scores)

def display_movie(movie: dict, similarity_score: float | None = None):
    title  = movie.get("title", "タイトル不明")
    year   = (movie.get("release_date") or "????")[:4]
    score  = movie.get("vote_average", "N/A")
    genres = ", ".join([g["name"] for g in movie.get("genres", [])]) if "genres" in movie else "不明"
    overview = movie.get("overview", "説明なし")
    poster = movie.get("poster_path")

    with st.expander(f"🎬 {title} ({year})", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w300{poster}")
        with col2:
            st.markdown(f"**ジャンル**: {genres}")
            st.markdown(f"**TMDbスコア**: {score}")
            if similarity_score is not None:
                st.markdown(f"**BERT類似度**: `{similarity_score:.3f}`")
            st.write(overview)

def batch_fetch_details(ids, max_workers: int = 8):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fetch_movie_details, ids))

# -----------------------------
# ここからUIゾーン
# -----------------------------
st.set_page_config(page_title="映画レコメンダー", layout="wide")
st.markdown("# 🎞️ 類似映画レコメンダー")
st.markdown("選んだ映画に似ている映画を人工知能がレコメンドします！ 🎥")

query = st.text_input("🔍 映画名を入力してください", value="")

if query:
    results = search_movies(query)
    if not results:
        st.warning("該当する映画が見つかりませんでした。")
    else:
        options = {f"{m['title']} ({(m.get('release_date') or '')[:4]})": m["id"] for m in results}
        selected_title = st.selectbox("🎯 検索結果から映画を選んでください", list(options.keys()))
        selected_id = options[selected_title]

        # 選択映画の基本情報
        selected_info = fetch_movie_details(selected_id)
        genres = ", ".join([g["name"] for g in selected_info.get("genres", [])])
        score  = selected_info.get("vote_average", "N/A")
        poster = selected_info.get("poster_path")

        st.markdown("## 🎬 選択された映画の情報")
        col1, col2 = st.columns([1, 2])
        with col1:
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w300{poster}")
        with col2:
            st.markdown(f"**タイトル**: {selected_info.get('title', '')}")
            st.markdown(f"**ジャンル**: {genres or '不明'}")
            st.markdown(f"**TMDbスコア**: {score}")
            st.write(selected_info.get("overview", "説明なし"))

        # 類似映画の取得
        similar = fetch_similar_movies(selected_id, limit=20)
        if not similar:
            st.info("類似映画が見つかりませんでした。")
        else:
            # まとめてエンコード（キャッシュして早くする！）
            cand_descs = [m.get("overview", "") or "" for m in similar]
            candidate_embs = compute_embeddings(cand_descs)

            # 上位とその他に分割
            (top5, top5_scores), (others, other_scores) = rank_similar_movies(selected_id, similar, candidate_embs, top_k=5)

            # 詳細は並列取得（HTTP待ちを短縮して高速化！）
            top_ids = [m["id"] for m in top5]
            other_ids = [m["id"] for m in others]
            top_details = batch_fetch_details(top_ids)
            other_details = batch_fetch_details(other_ids)

            # 上位5件の表示 + LIME解釈
            st.markdown("類似トップ5")
            for detail, score_ in zip(top_details, top5_scores):
                display_movie(detail, score_)
                with st.expander("🧠 LIMEでこの映画の類似度の理由を解説します（時間がかかることがあります）"):
                    st.markdown("""
                    **LIMEとは？**  
                    LIMEは「Local Interpretable Model-agnostic Explanations」の略で、  
                    複雑なモデルの予測結果を説明するために、入力テキストのどの単語が結果に影響しているかを教えてくれる技術です。  
                    これにより、なぜこの映画が似ていると判断されたかがわかります。  
                    """)
                    if st.button("LIME解説を表示", key=f"lime_{detail.get('id')}"):
                        explainer = BERTSimilarityExplainer(
                            selected_info.get("overview", ""),
                            token_unit="clause"   #sentences/clause/charの順に粒度が細かくなるよ
                            )
                        with st.spinner("LIMEで解釈中...しばらくお待ちください"):
                            target_desc = detail.get("overview", "") or ""  
                            explanation = explainer.explain(target_desc, num_features=10, num_samples=100)

                        # 正負で色分け　緑/赤
                        for word, weight in explanation.as_list(label=1):
                            color = "green" if weight > 0 else "red"
                            st.markdown(f"- <span style='color:{color}'>**{word}**</span>: `{weight:.3f}`", unsafe_allow_html=True)

            # その他の類似映画
            st.markdown("---")
            st.markdown("## 🎥 その他の類似映画")
            for detail, score_ in zip(other_details, other_scores):
                display_movie(detail, score_)
