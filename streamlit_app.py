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

if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = ""


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
    left, middle, right = st.columns(3)
    with left:
        st.markdown("#### Sparse baseline")
        entities = entity_frame(result["baseline"])
        if entities.empty:
            st.info("No entities found.")
        else:
            st.dataframe(entities, hide_index=True, use_container_width=True)
    with middle:
        st.markdown("#### Word embeddings")
        st.caption("Original embedding-model prediction")
        entities = entity_frame(result["word_embeddings"])
        if entities.empty:
            st.info("No entities found.")
        else:
            st.dataframe(entities, hide_index=True, use_container_width=True)

    with right:
        st.markdown("#### Context-assisted")
        st.caption("Adds date and name-context rules for unfamiliar words")
        entities = entity_frame(result["context_assisted"])
        if entities.empty:
            st.info("No entities found.")
        else:
            st.dataframe(entities, hide_index=True, use_container_width=True)

    with st.expander("Token-level predictions"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Model": model_name, "Entity": entity["token"], "Type": entity["tag"][2:]}
                    for model_name, predictions in (
                        ("Sparse baseline", result["baseline"]),
                        ("Word embeddings", result["word_embeddings"]),
                        ("Context-assisted", result["context_assisted"]),
                    )
                    for entity in predictions
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


sample_names = [name for name, _ in EXAMPLES]
selected_sample = st.selectbox("Optional: choose a sample to load", ["Write your own text", *sample_names])
if st.button("Load selected sample", disabled=selected_sample == "Write your own text"):
    st.session_state.analysis_text = dict(EXAMPLES)[selected_sample]
text = st.text_area(
    "Text to analyze",
    key="analysis_text",
    max_chars=5_000,
    height=120,
    placeholder="Type or paste any word, sentence, or short passage here…",
    help="Free typing is enabled. Enter a word, sentence, or short passage (up to 5,000 characters).",
)
st.caption("You can type any text here; sample texts are optional. Use Analyze text to extract entities.")

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
                "Context-assisted entities": ", ".join(
                    f"{item['token']} ({item['tag'][2:]})"
                    for item in result["context_assisted"] if item["tag"] != "O"
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
