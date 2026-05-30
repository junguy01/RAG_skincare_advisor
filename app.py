import json
import re
from pathlib import Path

import faiss
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer


# =========================
# Config
# =========================
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

# LM Studio defaults
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "local-model"


# =========================
# Helpers
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize(text):
    return clean_text(text).lower()


def make_doc(source_type, row_id, text, meta=None, title=""):
    return {
        "source_type": source_type,
        "row_id": int(row_id),
        "title": clean_text(title),
        "text": clean_text(text),
        "meta": meta or {},
    }


def shorten(text, max_len=220):
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " ..."


# =========================
# Data loading
# =========================
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
        meta = {
            "href": clean_text(row["href"]),
            "name": clean_text(row["name"]),
            "brand": clean_text(row["brand"]),
            "price": clean_text(row["price"]),
            "product_details": clean_text(row["product_details"]),
            "instruction": clean_text(row["instruction"]),
            "ingredients": clean_text(row["ingredients"]),
            "warnings": clean_text(row["warnings"]),
        }

        text = f"""
        Produktname: {meta['name']}
        Marke: {meta['brand']}
        Preis: {meta['price']}
        Beschreibung: {meta['product_details']}
        Anwendung: {meta['instruction']}
        Inhaltsstoffe: {meta['ingredients']}
        Warnhinweise: {meta['warnings']}
        Link: {meta['href']}
        """.strip()

        docs.append(
            make_doc(
                source_type="product",
                row_id=idx,
                title=f"{meta['brand']} {meta['name']}",
                text=text,
                meta=meta,
            )
        )

    return df, docs


def load_ingredient_knowledge(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        topic = clean_text(item.get("topic", ""))
        ingredient = clean_text(item.get("ingredient", ""))
        text = clean_text(item.get("text", ""))
        source_url = clean_text(item.get("source_url", ""))

        docs.append(
            make_doc(
                source_type="ingredient_knowledge",
                row_id=i,
                title=f"{ingredient} - {topic}",
                text=f"Thema: {topic}\nWirkstoff: {ingredient}\nErklärung: {text}\nQuelle: {source_url}",
                meta={
                    "topic": topic,
                    "ingredient": ingredient,
                    "text": text,
                    "source_url": source_url
                },
            )
        )

    return docs


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

            docs.append(
                make_doc(
                    source_type="routine_image",
                    row_id=i,
                    title=image_path.name,
                    text=f"Routinebild-Datei: {image_path.name}\nOCR-Text: {ocr_text}",
                    meta={
                        "file_name": image_path.name,
                        "ocr_text": ocr_text
                    },
                )
            )
        except Exception as e:
            failed_files.append(f"{image_path.name} ({e})")

    return docs, missing_files, failed_files


# =========================
# Retrieval
# =========================
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

    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(query_emb, min(top_k, len(docs)))

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = docs[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# =========================
# Skin profile
# =========================
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


# =========================
# Product scoring
# =========================
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
    flags = ingredient_flags(meta.get("ingredients", ""))
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


def rerank_products(product_hits, profile, max_items=8):
    rescored = []
    for doc in product_hits:
        final_score, product_type = score_product(doc, profile)
        item = doc.copy()
        item["final_score"] = final_score
        item["product_type"] = product_type
        rescored.append(item)

    rescored.sort(key=lambda x: x["final_score"], reverse=True)
    return rescored[:max_items]


# =========================
# Routine building
# =========================
def choose_best_product_by_type(products, wanted_type, used_ids=None):
    used_ids = used_ids or set()
    for p in products:
        unique_id = (p["source_type"], p["row_id"])
        if p.get("product_type") == wanted_type and unique_id not in used_ids:
            return p
    return None


def infer_weekly_plan(profile, routine_hits, reranked_products):
    used = set()

    cleanser = choose_best_product_by_type(reranked_products, "cleanser", used)
    if cleanser:
        used.add((cleanser["source_type"], cleanser["row_id"]))

    serum = choose_best_product_by_type(reranked_products, "serum", used)
    if serum:
        used.add((serum["source_type"], serum["row_id"]))

    moisturizer = choose_best_product_by_type(reranked_products, "moisturizer", used)
    if moisturizer:
        used.add((moisturizer["source_type"], moisturizer["row_id"]))

    sunscreen = choose_best_product_by_type(reranked_products, "sunscreen", used)
    if sunscreen:
        used.add((sunscreen["source_type"], sunscreen["row_id"]))

    mask = choose_best_product_by_type(reranked_products, "mask", used)

    concerns = set(profile.get("concerns", []))
    skin_types = set(profile.get("skin_types", []))

    evening_serum_days = []
    if serum:
        if "sensible Haut" in skin_types or "rötungen" in concerns:
            evening_serum_days = ["Mo", "Mi", "Fr"]
        else:
            evening_serum_days = ["Mo", "Di", "Do", "Sa"]

    mask_days = ["So"] if mask else []

    base_morning = []
    if cleanser:
        base_morning.append({"step": "Reinigung", "product": cleanser})
    if serum and ("trockenheit" in concerns or "pigmentflecken" in concerns):
        base_morning.append({"step": "Serum", "product": serum})
    if moisturizer:
        base_morning.append({"step": "Feuchtigkeitspflege", "product": moisturizer})
    if sunscreen:
        base_morning.append({"step": "Sonnenschutz", "product": sunscreen})

    base_evening = []
    if cleanser:
        base_evening.append({"step": "Reinigung", "product": cleanser})
    if moisturizer:
        base_evening_tail = [{"step": "Feuchtigkeitspflege", "product": moisturizer}]
    else:
        base_evening_tail = []

    weekly_plan = []
    day_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    for day in day_names:
        morning_steps = list(base_morning)
        evening_steps = list(base_evening)

        if serum and day in evening_serum_days:
            evening_steps.append({"step": "Serum", "product": serum})

        if mask and day in mask_days:
            evening_steps.append({"step": "Maske", "product": mask})

        evening_steps.extend(base_evening_tail)

        weekly_plan.append({
            "day": day,
            "morning": morning_steps,
            "evening": evening_steps,
        })

    routine_context = [
        {
            "file_name": r["meta"].get("file_name", ""),
            "ocr_text": r["meta"].get("ocr_text", "")
        }
        for r in routine_hits
    ]

    return {
        "weekly_plan": weekly_plan,
        "routine_context": routine_context
    }


# =========================
# Context formatting
# =========================
def format_context_blocks(ingredient_hits, routine_hits, reranked_products):
    ingredient_context = []
    for hit in ingredient_hits[:5]:
        m = hit["meta"]
        ingredient_context.append({
            "ingredient": m.get("ingredient", ""),
            "topic": m.get("topic", ""),
            "text": m.get("text", ""),
            "source_url": m.get("source_url", "")
        })

    routine_context = []
    for hit in routine_hits[:4]:
        routine_context.append({
            "file_name": hit["meta"].get("file_name", ""),
            "ocr_text": hit["meta"].get("ocr_text", "")
        })

    product_context = []
    for item in reranked_products[:8]:
        m = item["meta"]
        product_context.append({
            "brand": m.get("brand", ""),
            "name": m.get("name", ""),
            "price": m.get("price", ""),
            "instruction": m.get("instruction", ""),
            "ingredients": m.get("ingredients", ""),
            "warnings": m.get("warnings", ""),
            "href": m.get("href", ""),
            "product_type": item.get("product_type", "other"),
            "score": round(item.get("final_score", 0.0), 4),
        })

    return {
        "ingredients": ingredient_context,
        "routines": routine_context,
        "products": product_context
    }


# =========================
# LM Studio generation
# =========================
def get_lmstudio_client(base_url):
    return OpenAI(
        base_url=base_url,
        api_key=LMSTUDIO_API_KEY
    )


def generate_rag_answer_lmstudio(user_text, profile, inferred_routine, context, model_name, base_url):
    client = get_lmstudio_client(base_url)

    system_prompt = """
Du bist ein deutschsprachiger Skincare-RAG-Assistent.

Nutze ausschließlich den bereitgestellten Kontext.
Wenn etwas im Kontext nicht ausreichend belegt ist, sage das offen.
Keine medizinischen Diagnosen.

Erzeuge die Antwort IMMER in genau dieser Reihenfolge:
1. Empfohlene Routine
2. Erklärte Wirkstoffe
3. Produktempfehlungen
4. Hinweis

Wichtige Regeln:
- Beginne mit einer konkreten Wochenroutine.
- Die Routine soll nach Tagen gegliedert sein, jeweils Morgen und Abend.
- Die OCR-Routinebilder sollen als Orientierung dienen, aber an das Hautprofil angepasst werden.
- Erkläre danach die relevantesten Wirkstoffe knapp und verständlich.
- Empfiehl danach passende Produkte aus dem Produktkontext.
- Bei Produktempfehlungen möglichst Produkttyp, Marke, Name, Preis und kurzer Nutzungszweck.
- Antworte in sauberem Markdown.
    """.strip()

    user_prompt = f"""
Nutzeranfrage:
{user_text}

Erkanntes Hautprofil:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Vorstrukturierte Wochenroutine:
{json.dumps(inferred_routine, ensure_ascii=False, indent=2)}

Abgerufener RAG-Kontext:
{json.dumps(context, ensure_ascii=False, indent=2)}

Erzeuge jetzt die finale Antwort.
    """.strip()

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return completion.choices[0].message.content


# =========================
# Prepare system
# =========================
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


# =========================
# UI
# =========================
st.set_page_config(page_title="Skincare RAG Chatbot", layout="wide")
st.title("Skincare RAG Chatbot")
st.caption("RAG über CSV + JSON + OCR + LM Studio")

system = prepare_system()

if system["missing_files"]:
    st.warning("Nicht gefundene Bilder:\n" + "\n".join(system["missing_files"]))

if system["failed_files"]:
    st.warning("OCR-Probleme bei:\n" + "\n".join(system["failed_files"]))

with st.sidebar:
    st.header("LM Studio")
    lmstudio_base_url = st.text_input("Base URL", value=LMSTUDIO_BASE_URL)
    lmstudio_model = st.text_input("Model Name", value=LMSTUDIO_MODEL)
    st.caption("Beispiel Base URL: http://localhost:1234/v1")

user_text = st.text_area(
    "Beschreibe deine Haut oder wonach du suchst",
    placeholder="Beispiel: Ich habe sensible Mischhaut mit Rötungen und suche eine einfache Abendroutine und passende Produkte.",
    height=140
)

if st.button("Routine & Produkte vorschlagen") and user_text.strip():
    with st.spinner("Suche relevantes Wissen und generiere Antwort mit LM Studio ..."):
        profile = extract_skin_profile(user_text)

        ingredient_hits = semantic_search(
            user_text,
            system["ingredient_docs"],
            system["ingredient_index"],
            system["model"],
            top_k=5
        )

        routine_hits = semantic_search(
            user_text,
            system["routine_docs"],
            system["routine_index"],
            system["model"],
            top_k=4
        )

        product_hits = semantic_search(
            user_text,
            system["product_docs"],
            system["product_index"],
            system["model"],
            top_k=20
        )

        reranked_products = rerank_products(product_hits, profile, max_items=8)
        inferred_routine = infer_weekly_plan(profile, routine_hits, reranked_products)
        rag_context = format_context_blocks(ingredient_hits, routine_hits, reranked_products)

        try:
            answer = generate_rag_answer_lmstudio(
                user_text=user_text,
                profile=profile,
                inferred_routine=inferred_routine,
                context=rag_context,
                model_name=lmstudio_model,
                base_url=lmstudio_base_url
            )

            st.subheader("Antwort")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Fehler bei der LM-Studio-RAG-Generierung: {e}")

    with st.expander("Erkanntes Hautprofil", expanded=False):
        st.json(profile)

    with st.expander("Verwendeter Routine-Kontext", expanded=False):
        for hit in routine_hits:
            st.write(f"**{hit['meta'].get('file_name', '')}**")
            st.write(hit["meta"].get("ocr_text", ""))

    with st.expander("Top-Produkte nach Re-Ranking", expanded=False):
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

    with st.expander("Wirkstoff-Kontext", expanded=False):
        for hit in ingredient_hits:
            st.write(f"**{hit['meta'].get('ingredient', '')}** — {hit['meta'].get('topic', '')}")
            st.write(hit["meta"].get("text", ""))

    with st.expander("Vorstrukturierte Wochenroutine", expanded=False):
        st.json(inferred_routine)