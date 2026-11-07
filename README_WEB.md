# CineScore Web App

This is the web layer for the existing CineScore movie-rating model.

## Run

```bash
python server.py
```

Then open:

`http://127.0.0.1:8000`

## API

- `GET /api/health` — confirms the model and dataset are loaded.
- `GET /api/meta` — returns model metadata and genre vocabulary.
- `POST /api/predict` — runs the existing trained model.

Example request body:

```json
{
  "budget": 50000000,
  "popularity": 20,
  "runtime": 120,
  "release_year": 2024,
  "genres": ["Drama", "Action"],
  "cast": "Tom Cruise, Brad Pitt, Will Smith",
  "director": "Christopher Nolan",
  "is_english": true
}
```

## Important

`movie_rating_model.pkl`, `genre_encoder.pkl`, and `features.pkl` are used as supplied. The old `actor_avg.pkl` and `director_avg.pkl` are not rewritten because they are incompatible with the current pandas pickle loader; the API recreates the same lookup tables from the bundled TMDB CSV files at startup.

## Streamlit version

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. `app.py` reuses `predict()` from `server.py`, so the Streamlit app and the REST API give identical predictions.

Deploy free on [Streamlit Community Cloud](https://share.streamlit.io): New app → repo `Arpit-1210/cinescore`, branch `main`, file `app.py`, Python 3.11.
