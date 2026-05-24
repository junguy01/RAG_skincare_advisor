import json
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer

# Source data
PRODUCTS_CSV = Path("products_cleaned.csv")
INGREDIENTS_JSON = Path("paulas_choice_ingredients.json")

INSTA_DIR = Path("instagram_downloads/xskincare_CvUO2zWK_N-")
ROUTINE_IMAGE_PATHS = [
    INSTA_DIR / "2023-07-30_09-15-37_UTC_1.jpg",
    INSTA_DIR / "2023-07-30_09-15-37_UTC_2.jpg",
    INSTA_DIR / "2023-07-30_09-15-37_UTC_3.jpg",
    INSTA_DIR / "2023-07-30_09-15-37_UTC_4.jpg",
    INSTA_DIR / "2023-07-30_09-15-37_UTC_5.jpg",
    INSTA_DIR / "2023-07-30_09-15-37_UTC_6.jpg",
    INSTA_DIR / "2023-07-30_09-15-37_UTC_7.jpg",
]


EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize(text):
    return clean_text(text).lower()


def shorten(text, max_len=220):
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " ..."

# csv loader
def load_products(csv_path: Path):
    df = pd.read_csv(csv_path)

    expected = {
        "href", "name", "brand", "price",
        "product_details", "instruction", "ingredients", "warnings"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende CSV-Spalten: {missing}")

    docs = []
    for idx, row in df.iterrows():
        text = f"""
        Produktname: {clean_text(row['name'])}
        Marke: {clean_text(row['brand'])}
        Preis: {clean_text(row['price'])}
        Beschreibung: {clean_text(row['product_details'])}
        Anwendung: {clean_text(row['instruction'])}
        Inhaltsstoffe: {clean_text(row['ingredients'])}
        Warnhinweise: {clean_text(row['warnings'])}
        Link: {clean_text(row['href'])}
        """.strip()

        docs.append({
            "source": "product",
            "row_id": int(idx),
            "text": text,
            "meta": {
                "href": clean_text(row["href"]),
                "name": clean_text(row["name"]),
                "brand": clean_text(row["brand"]),
                "price": clean_text(row["price"]),
                "product_details": clean_text(row["product_details"]),
                "instruction": clean_text(row["instruction"]),
                "ingredients": clean_text(row["ingredients"]),
                "warnings": clean_text(row["warnings"]),
            }
        })

    return df, docs


# json loader
def load_ingredient_knowledge(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        topic = clean_text(item.get("topic", ""))
        ingredient = clean_text(item.get("ingredient", ""))
        text = clean_text(item.get("text", ""))
        source_url = clean_text(item.get("source_url", ""))

        docs.append({
            "source": "ingredient_knowledge",
            "row_id": i,
            "text": f"Thema: {topic}\nWirkstoff: {ingredient}\nErklärung: {text}\nQuelle: {source_url}",
            "meta": {
                "topic": topic,
                "ingredient": ingredient,
                "text": text,
                "source_url": source_url
            }
        })

    return docs


# OCR for jpg
def ocr_image(image_path: Path, lang: str = "deu+eng"):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return clean_text(text)


def load_routine_images(image_paths):
    docs = []
    missing_files = []
    failed_files = []

    for i, image_path in enumerate(image_paths):
        if not image_path.exists():
            missing_files.append(str(image_path))
            continue

        try:
            ocr_text = ocr_image(image_path)

            if not ocr_text:
                failed_files.append(f"{image_path.name} (kein OCR-Text erkannt)")
                continue

            docs.append({
                "source": "routine_image",
                "row_id": i,
                "text": f"Routinebild-Datei: {image_path.name}\nOCR-Text: {ocr_text}",
                "meta": {
                    "file_name": image_path.name,
                    "ocr_text": ocr_text
                }
            })
        except Exception as e:
            failed_files.append(f"{image_path.name} ({e})")

    return docs, missing_files, failed_files


# index
def build_index(docs, model):
    if not docs:
        return None

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def semantic_search(query, docs, index, model, top_k=5):
    if not docs or index is None:
        return []

    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query_emb, min(top_k, len(docs)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = docs[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# skin profile
def extract_skin_profile(user_text: str):
    t = normalize(user_text)

    concern_map = {
        "unreinheiten": ["pickel", "akne", "unreinheiten"],
        "verstopfte poren": ["verstopfte poren", "poren", "mitesser"],
        "trockenheit": ["trocken", "dehydriert", "spannt", "feuchtigkeitsarm"],
        "rötungen": ["rötung", "rötungen", "empfindlich", "sensibel", "gereizt"],
        "pigmentflecken": ["pigment", "post-akne", "flecken", "ungleichmäßiger hautton"],
        "falten": ["falten", "linien", "anti aging", "anti-aging"]
    }

    skin_type_map = {
        "trockene Haut": ["trockene haut", "trocken"],
        "fettige Haut": ["fettige haut", "ölig", "glänzt"],
        "Mischhaut": ["mischhaut"],
        "sensible Haut": ["sensible haut", "empfindliche haut", "sensibel", "empfindlich"]
    }

    concerns = []
    skin_types = []

    for label, variants in concern_map.items():
        if any(v in t for v in variants):
            concerns.append(label)

    for label, variants in skin_type_map.items():
        if any(v in t for v in variants):
            skin_types.append(label)

    return {
        "concerns": sorted(set(concerns)),
        "skin_types": sorted(set(skin_types))
    }


# product logic
def detect_product_type(text: str):
    t = normalize(text)

    mapping = {
        "cleanser": ["reinigung", "reinigungs", "waschgel", "cleanser", "mizellen", "reinigungsöl", "balm"],
        "serum": ["serum", "ampoule", "ampulle"],
        "moisturizer": ["creme", "gel", "fluid", "feuchtigkeit", "nachtpflege", "tagespflege"],
        "sunscreen": ["lsf", "spf", "sonnenschutz", "uv", "sun"],
        "mask": ["maske", "tuchmaske", "augenpads", "hydrogel"]
    }

    for product_type, keywords in mapping.items():
        if any(k in t for k in keywords):
            return product_type
    return "other"

# ingredients
def ingredient_flags(ingredients_text: str):
    t = normalize(ingredients_text)
    return {
        "has_niacinamide": "niacinamide" in t,
        "has_hyaluron": "hyaluron" in t or "sodium hyaluronate" in t,
        "has_panthenol": "panthenol" in t,
        "has_ceramide": "ceramide" in t,
        "has_peptides": "peptide" in t,
        "has_allantoin": "allantoin" in t,
        "has_ectoin": "ectoin" in t,
        "has_glycerin": "glycerin" in t,
        "has_urea": "urea" in t,
        "has_bha": "salicylic acid" in t or "salicyl" in t,
        "has_aha": "glycolic acid" in t or "lactic acid" in t or "aha" in t,
        "has_vitamin_c": "ascorb" in t or "vitamin c" in t,
        "has_retinoid": "retinol" in t or "retinal" in t or "retinyl" in t,
        "has_fragrance": "parfum" in t or "fragrance" in t,
        "has_alcohol_denat": "alcohol denat" in t
    }


def score_product(doc, profile):
    meta = doc["meta"]
    flags = ingredient_flags(meta["ingredients"])
    product_type = detect_product_type(doc["text"])
    score = doc.get("score", 0.0)

    concerns = profile["concerns"]
    skin_types = profile["skin_types"]

    if "verstopfte poren" in concerns or "unreinheiten" in concerns:
        if flags["has_bha"] or flags["has_niacinamide"]:
            score += 0.22

    if "trockenheit" in concerns:
        if flags["has_hyaluron"] or flags["has_glycerin"] or flags["has_urea"] or flags["has_panthenol"]:
            score += 0.22

    if "rötungen" in concerns:
        if flags["has_panthenol"] or flags["has_allantoin"] or flags["has_ceramide"] or flags["has_ectoin"]:
            score += 0.24

    if "falten" in concerns:
        if flags["has_retinoid"] or flags["has_peptides"]:
            score += 0.20

    if "sensible Haut" in skin_types:
        if flags["has_fragrance"]:
            score -= 0.12
        if flags["has_alcohol_denat"]:
            score -= 0.15
        if flags["has_retinoid"] or flags["has_aha"] or flags["has_bha"]:
            score -= 0.08

    return score, product_type


def rerank_products(product_hits, profile, max_items=6):
    rescored = []
    for doc in product_hits:
        final_score, product_type = score_product(doc, profile)
        item = doc.copy()
        item["final_score"] = final_score
        item["product_type"] = product_type
        rescored.append(item)

    rescored.sort(key=lambda x: x["final_score"], reverse=True)
    return rescored[:max_items]


# answer
def build_answer(profile, ingredient_hits, routine_hits, product_hits):
    concerns = ", ".join(profile["concerns"]) if profile["concerns"] else "keine klar erkannten Anliegen"
    skin_types = ", ".join(profile["skin_types"]) if profile["skin_types"] else "kein klar erkannter Hauttyp"

    lines = []
    lines.append(f"### Erkanntes Profil")
    lines.append(f"- Hauttyp: {skin_types}")
    lines.append(f"- Anliegen: {concerns}")
    lines.append("")

    if routine_hits:
        lines.append("### Hinweise aus den eingelesenen Routinebildern")
        for hit in routine_hits:
            preview = shorten(hit['meta']['ocr_text'], 500)
            lines.append(f"- **{hit['meta']['file_name']}**: {preview}")
        lines.append("")

    if ingredient_hits:
        lines.append("### Passende Wirkstoffe aus dem JSON")
        for hit in ingredient_hits[:3]:
            m = hit["meta"]
            lines.append(f"- **{m['ingredient']}** für **{m['topic']}**: {m['text']}")
        lines.append("")

    if product_hits:
        lines.append("### Passende Produkte")
        for item in product_hits:
            m = item["meta"]
            lines.append(f"- **[{item['product_type']}] {m['brand']} - {m['name']}** | {m['price']}")
            lines.append(f"  - Anwendung: {shorten(m['instruction'], 220)}")
            if m["warnings"]:
                lines.append(f"  - Warnhinweise: {shorten(m['warnings'], 180)}")
        lines.append("")

    lines.append("Hinweis: Das ist kosmetische Orientierung und keine medizinische Diagnose.")
    return "\n".join(lines)


@st.cache_resource
def prepare_system():
    model = SentenceTransformer(EMBED_MODEL)

    df, product_docs = load_products(PRODUCTS_CSV)
    ingredient_docs = load_ingredient_knowledge(INGREDIENTS_JSON)
    routine_docs, missing_files, failed_files = load_routine_images(ROUTINE_IMAGE_PATHS)

    product_index = build_index(product_docs, model)
    ingredient_index = build_index(ingredient_docs, model)
    routine_index = build_index(routine_docs, model)

    return {
        "df": df,
        "model": model,
        "product_docs": product_docs,
        "ingredient_docs": ingredient_docs,
        "routine_docs": routine_docs,
        "missing_files": missing_files,
        "failed_files": failed_files,
        "product_index": product_index,
        "ingredient_index": ingredient_index,
        "routine_index": routine_index
    }


# UI
st.set_page_config(page_title="Skincare RAG Chatbot", layout="wide")
st.title("Skincare RAG Chatbot")
st.caption("RAG über CSV + JSON + OCR aus Routinebildern")

system = prepare_system()

if system["missing_files"]:
    st.warning("Nicht gefundene Bilder:\n" + "\n".join(system["missing_files"]))

if system["failed_files"]:
    st.warning("OCR-Probleme bei:\n" + "\n".join(system["failed_files"]))

user_text = st.text_area(
    "Beschreibe deine Haut oder wonach du suchst",
    placeholder="Beispiel: Ich habe sensible Mischhaut mit Rötungen und suche eine einfache Abendroutine und passende Produkte."
)

if st.button("Routine & Produkte vorschlagen") and user_text.strip():
    profile = extract_skin_profile(user_text)

    ingredient_hits = semantic_search(
        user_text,
        system["ingredient_docs"],
        system["ingredient_index"],
        system["model"],
        top_k=4
    )

    routine_hits = semantic_search(
        user_text,
        system["routine_docs"],
        system["routine_index"],
        system["model"],
        top_k=3
    )

    product_hits = semantic_search(
        user_text,
        system["product_docs"],
        system["product_index"],
        system["model"],
        top_k=20
    )

    reranked_products = rerank_products(product_hits, profile, max_items=6)
    answer = build_answer(profile, ingredient_hits, routine_hits, reranked_products)

    st.subheader("Antwort")
    st.markdown(answer)

    with st.expander("OCR aus Bildern"):
        for doc in system["routine_docs"]:
            st.write(f"**{doc['meta']['file_name']}**")
            st.write(doc["meta"]["ocr_text"][:3000])

    with st.expander("Top Produkte"):
        table_data = []
        for p in reranked_products:
            table_data.append({
                "brand": p["meta"]["brand"],
                "name": p["meta"]["name"],
                "price": p["meta"]["price"],
                "product_type": p["product_type"],
                "score": round(p["final_score"], 3),
                "href": p["meta"]["href"]
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)