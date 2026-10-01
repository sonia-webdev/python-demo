import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ner_embeddings.pipeline import ROOT, load_models, predict_text


EXAMPLES = [
    ("People, company, and location", "Sophia met Daniel at Microsoft in Seattle on Friday."),
    ("Company and date", "Amazon opened an office in Dublin in 2020."),
    ("Multi-word location", "David visited New York in June."),
    ("Organizations and place", "The team from Meta met in Rome."),
    ("Person and organization", "Emma met James at Stanford University."),
    ("Date and location", "The launch is scheduled for December 2024 in Lisbon."),
    ("Long organization name", "Lucas joined the World Health Organization in 2022."),
    ("Write your own", "Emma joined Google in London in 2024."),
]
ENTITY_COLORS = {
    "PER": "#f8e8ee",
    "ORG": "#e7effb",
    "LOC": "#e4f4f1",
    "DATE": "#fcf1d9",
}


st.set_page_config(page_title="EntityLens · NER", page_icon="🔎", layout="wide")
st.title("🔎 EntityLens")
st.subheader("Named Entity Recognition with Word Embeddings")
st.write(
    "Explore people, organizations, locations, and dates in text. "
    "Compare a sparse-feature baseline with corpus-trained word embeddings."
)


@st.cache_resource
def cached_models() -> tuple:
    return load_models()


@st.cache_data
def load_report() -> dict:
    report_path = ROOT / "reports" / "results.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


with st.spinner("Loading the NER models…"):
    models = cached_models()


def entity_frame(predictions: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Entity": item["token"], "Type": item["tag"][2:]}
            for item in predictions
            if item["tag"] != "O"
        ],
        columns=["Entity", "Type"],
    )


def show_predictions(text: str) -> None:
    result = predict_text(text, models)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Sparse baseline")
        entities = entity_frame(result["baseline"])
        if entities.empty:
            st.info("No entities found.")
        else:
            st.dataframe(entities, hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Word embeddings")
        entities = entity_frame(result["word_embeddings"])
        if entities.empty:
            st.info("No entities found.")
        else:
            st.dataframe(entities, hide_index=True, use_container_width=True)

    with st.expander("Token-level predictions"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Token": baseline["token"],
                        "Baseline": baseline["tag"],
                        "Word embeddings": embedded["tag"],
                    }
                    for baseline, embedded in zip(result["baseline"], result["word_embeddings"])
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


sample_names = [name for name, _ in EXAMPLES]
selected_sample = st.selectbox("Choose a sample text", sample_names)
default_text = dict(EXAMPLES)[selected_sample]
text = st.text_area(
    "Text to analyze",
    value=default_text,
    max_chars=5_000,
    height=120,
    help="Enter a word, sentence, or short passage (up to 5,000 characters).",
)

analyze_column, batch_column = st.columns(2)
with analyze_column:
    analyze_clicked = st.button("Analyze text", type="primary", use_container_width=True)
with batch_column:
    analyze_all_clicked = st.button("Analyze all 8 examples", use_container_width=True)

if analyze_clicked:
    try:
        show_predictions(text)
    except ValueError as error:
        st.error(str(error))

if analyze_all_clicked:
    st.markdown("### All sample predictions")
    try:
        rows = []
        for sample_name, sample_text in EXAMPLES:
            result = predict_text(sample_text, models)
            rows.append({
                "Example": sample_name,
                "Text": sample_text,
                "Baseline entities": ", ".join(
                    f"{item['token']} ({item['tag'][2:]})"
                    for item in result["baseline"] if item["tag"] != "O"
                ) or "None",
                "Word-embedding entities": ", ".join(
                    f"{item['token']} ({item['tag'][2:]})"
                    for item in result["word_embeddings"] if item["tag"] != "O"
                ) or "None",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    except ValueError as error:
        st.error(str(error))

st.divider()
st.markdown("### Held-out evaluation results")
st.caption(
    "Exact entity-span metrics on the bundled, small demonstration dataset; "
    "these scores are not a reliable estimate of real-world performance."
)

try:
    report = load_report()
    model_names = [("Sparse baseline", report["baseline"]), ("Word embeddings", report["word_embeddings"])]
    metric_rows = []
    for label in ["PER", "ORG", "LOC", "DATE", "micro_avg"]:
        display_name = "Micro average" if label == "micro_avg" else label
        for model_name, scores in model_names:
            metric = scores[label]
            metric_rows.append({
                "Entity": display_name,
                "Model": model_name,
                "Precision": metric["precision"],
                "Recall": metric["recall"],
                "F1": metric["f1"],
                "Support": metric["support"],
            })
    st.dataframe(
        pd.DataFrame(metric_rows).style.format(
            {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
except (OSError, json.JSONDecodeError, KeyError) as error:
    st.error(f"Could not load evaluation results: {error}")

st.caption("Predictions run using this app's local models; submitted text is not stored.")
