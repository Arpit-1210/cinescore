from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ast
import json
import math
import mimetypes
import threading

import os

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

# -----------------------------
# Load the EXISTING trained model
# -----------------------------
MODEL = joblib.load(BASE / "movie_rating_model.pkl")
MLB = joblib.load(BASE / "genre_encoder.pkl")
FEATURES = joblib.load(BASE / "features.pkl")

# The original actor/director pickles were created with an older pandas
# serialization format. Rebuild the same lookup tables from the bundled
# source CSV instead of modifying/retraining the model.
def parse_cast(value):
    try:
        if isinstance(value, str):
            return [c["name"] for c in ast.literal_eval(value)][:3]
    except Exception:
        pass
    return []


def get_director(value):
    try:
        crew = ast.literal_eval(value) if isinstance(value, str) else value
        directors = [c["name"] for c in crew if c.get("job") == "Director"]
        return directors[0] if directors else "Unknown"
    except Exception:
        return "Unknown"


def build_lookups():
    movies = pd.read_csv(BASE / "tmdb_5000_movies.csv")
    credits = pd.read_csv(BASE / "tmdb_5000_credits.csv")
    df = movies.merge(credits, left_on="id", right_on="movie_id")
    df["genres"] = df["genres"].apply(
        lambda x: [g["name"] for g in ast.literal_eval(x)] if isinstance(x, str) else []
    )
    df["cast"] = df["cast"].apply(parse_cast)
    df["director"] = df["crew"].apply(get_director)

    exploded = df.explode("cast")
    actor_avg = exploded.groupby("cast")["vote_average"].mean()
    director_avg = df.groupby("director")["vote_average"].mean()
    fallback = float(df["vote_average"].mean())

    return df, actor_avg, director_avg, fallback


DATA, ACTOR_AVG, DIRECTOR_AVG, GLOBAL_AVG = build_lookups()


def safe_number(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def predict(payload):
    budget = max(0.0, safe_number(payload.get("budget"), 0))
    popularity = max(0.0, safe_number(payload.get("popularity"), 0))
    runtime = max(0.0, safe_number(payload.get("runtime"), 0))
    release_year = safe_number(payload.get("release_year"), 2024)
    genres = payload.get("genres") or []
    cast = payload.get("cast") or []
    director = str(payload.get("director") or "Unknown").strip()
    is_english = bool(payload.get("is_english", True))

    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]
    if isinstance(cast, str):
        cast = [c.strip() for c in cast.split(",") if c.strip()]

    genres = [g for g in genres if g in set(MLB.classes_)]
    cast = cast[:3]

    cast_scores = [float(ACTOR_AVG.get(actor, GLOBAL_AVG)) for actor in cast]
    cast_score = float(np.mean(cast_scores)) if cast_scores else GLOBAL_AVG
    director_score = float(DIRECTOR_AVG.get(director, GLOBAL_AVG))

    row = {
        "budget_log": np.log1p(budget),
        "popularity_log": np.log1p(popularity),
        "runtime": runtime,
        "release_year": release_year,
        "cast_score": cast_score,
        "director_score": director_score,
        "num_genres": len(genres),
        "cast_size": len(cast),
        "is_english": int(is_english),
    }

    genre_vec = MLB.transform([genres])[0]
    for i, genre in enumerate(MLB.classes_):
        row[genre] = int(genre_vec[i])

    # Keep the exact feature order saved with the model project.
    X = pd.DataFrame([row])
    for feature in FEATURES:
        if feature not in X.columns:
            X[feature] = 0
    X = X[FEATURES]

    raw_prediction = float(MODEL.predict(X)[0])
    prediction = float(np.clip(raw_prediction, 0, 10))
    stars = int(np.clip(round(prediction / 2), 0, 5))

    return {
        "prediction": round(prediction, 2),
        "stars": stars,
        "stars_text": "★" * stars + "☆" * (5 - stars),
        "raw_prediction": round(raw_prediction, 4),
        "inputs": {
            "budget": budget,
            "popularity": popularity,
            "runtime": runtime,
            "release_year": release_year,
            "genres": genres,
            "cast": cast,
            "director": director,
            "is_english": is_english,
        },
    }


def json_response(handler, status, payload):
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/api/health", "/health"):
            return json_response(self, 200, {
                "status": "ok",
                "model_loaded": True,
                "model_type": type(MODEL).__name__,
                "dataset_rows": int(len(DATA)),
            })

        if self.path == "/api/meta":
            return json_response(self, 200, {
                "genres": list(MLB.classes_),
                "dataset_rows": int(len(DATA)),
                "global_rating": round(GLOBAL_AVG, 2),
                "model_type": type(MODEL).__name__,
                "features": FEATURES,
            })

        if self.path in ("/", "/index.html"):
            return self.serve_file(BASE / "index.html")

        path = BASE / self.path.lstrip("/")
        if path.exists() and path.is_file() and BASE in path.parents:
            return self.serve_file(path)
        return json_response(self, 404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return json_response(self, 404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                return json_response(self, 413, {"error": "Request too large"})
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = predict(payload)
            return json_response(self, 200, result)
        except Exception as exc:
            return json_response(self, 400, {"error": str(exc)})

    def serve_file(self, path):
        try:
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type == "text/html":
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"CineScore running at http://{HOST}:{PORT}")
    print(f"Model: {type(MODEL).__name__} | Dataset: {len(DATA):,} movies")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping CineScore...")
        server.shutdown()
