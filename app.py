"""CineScore — Streamlit version.

Run locally:  streamlit run app.py
Uses the same trained Random Forest and the same preprocessing as server.py,
so predictions match the Render API exactly.
"""
import streamlit as st

st.set_page_config(page_title="CineScore", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading model and TMDB data…")
def load():
    # server.py loads the model, genre encoder, feature list and rebuilds the
    # actor/director average lookups at import time (no HTTP server starts).
    import server
    return server


S = load()

st.markdown(
    """
    <style>
      .score {font-size:4rem;font-weight:700;color:#f4c95d;line-height:1;margin:0}
      .stars {font-size:1.8rem;color:#f4c95d;letter-spacing:4px}
      .muted {color:#8e96a5;font-size:.9rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 CineScore")
st.caption(
    f"Movie rating predictor · Random Forest ({S.MODEL.n_estimators} trees) · "
    f"trained on {len(S.DATA):,} TMDB movies"
)

directors = sorted(d for d in S.DIRECTOR_AVG.index if isinstance(d, str) and d != "Unknown")
actors = sorted(a for a in S.ACTOR_AVG.index if isinstance(a, str))

with st.form("predict"):
    c1, c2 = st.columns(2)
    with c1:
        budget_m = st.number_input("Budget (USD, millions)", 0.0, 500.0, 50.0, 1.0)
        runtime = st.number_input("Runtime (minutes)", 40, 300, 120, 5)
        is_english = st.toggle("English language", value=True)
    with c2:
        release_year = st.number_input("Release year", 1910, 2035, 2026, 1)
        popularity = st.slider("Expected popularity (TMDB scale)", 0.0, 200.0, 20.0, 0.5)

    genres = st.multiselect("Genres", list(S.MLB.classes_), default=["Drama"])
    director = st.selectbox(
        "Director", directors, index=None, placeholder="Type to search…",
        accept_new_options=True,
    )
    cast = st.multiselect(
        "Top cast (up to 3)", actors, max_selections=3, placeholder="Type to search…",
        accept_new_options=True,
    )
    submitted = st.form_submit_button("Predict rating", type="primary", use_container_width=True)

if submitted:
    try:
        r = S.predict({
            "budget": budget_m * 1_000_000,
            "popularity": popularity,
            "runtime": runtime,
            "release_year": release_year,
            "genres": genres,
            "cast": cast,
            "director": director or "Unknown",
            "is_english": is_english,
        })
    except Exception as exc:  # never crash the page on bad input
        st.error(f"Prediction failed: {exc}")
    else:
        st.divider()
        st.markdown(f'<p class="score">{r["prediction"]:.1f}<span class="muted"> / 10</span></p>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="stars">{r["stars_text"]}</div>', unsafe_allow_html=True)

        unknown = [a for a in cast if a not in S.ACTOR_AVG.index]
        if director and director not in S.DIRECTOR_AVG.index:
            unknown.append(director)
        if unknown:
            st.info("Not in the dataset, so the global average was used: " + ", ".join(unknown))

with st.expander("How it works"):
    st.markdown(
        """
- **Features (9):** log budget, log popularity, runtime, release year, cast score,
  director score, number of genres, cast size, English flag.
- **Cast / director score:** average TMDB rating of their past movies (target encoding);
  unknown names fall back to the global average.
- **Model:** `RandomForestRegressor`, output clamped to 0–10 and shown as stars.
- The same `predict()` also powers the REST API in `server.py`.
        """
    )
