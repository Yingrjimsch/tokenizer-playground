import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from typing import Dict, List

from tokenizer_framework import (
    TokenizerAnalyzer,
    BaseTokenizer,
    TiktokenTokenizer,
    HuggingFaceTokenizer,
    SimpleTokenizer,
    TextSampleCollector,
    create_sample_texts,
    AutoTokenizer,
)

st.set_page_config(page_title="Tokenizer Comparison", layout="wide")

# --------- Caching helpers ---------
# Factories allow lazy creation of heavy tokenizers
TokenizerFactory = Dict[str, callable]

def _hf_enabled() -> bool:
    # Enable HF only if transformers is installed and not explicitly disabled
    if AutoTokenizer is None:
        return False
    return os.environ.get("APP_ENABLE_HF", "0").lower() in ("1", "true", "yes")

@st.cache_resource(show_spinner=False)
def get_available_tokenizers() -> Dict[str, callable]:
    factories: Dict[str, callable] = {}

    # Tiktoken (OpenAI) — small footprint
    for enc in ["cl100k_base", "o200k_base"]:
        def make_factory(encoding_name: str):
            return lambda: TiktokenTokenizer(encoding_name)
        try:
            # Validate availability once by trying to build and encode trivial input
            _ = TiktokenTokenizer(enc)
            factories[f"tiktoken_{enc}"] = make_factory(enc)
        except Exception:
            pass

    # Simple tokenizer
    factories["simple"] = lambda: SimpleTokenizer()

    # Optional: curated small HF models only when enabled
    if _hf_enabled():
        small_models = [
            "gpt2",
            "bert-base-uncased",
            "roberta-base",
        ]
        for model in small_models:
            name = f"hf_{model.split('/')[-1]}"
            factories[name] = (lambda m=model: lambda: HuggingFaceTokenizer(m))()

    return factories

@st.cache_resource(show_spinner=False)
def load_hf_tokenizer(model_name: str) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(model_name)

@st.cache_data(show_spinner=False)
def get_default_texts() -> Dict[str, str]:
    return create_sample_texts()

@st.cache_data(show_spinner=False)
def get_wikipedia_languages(topic: str) -> Dict[str, str]:
    """Get available Wikipedia languages for a topic."""
    collector = TextSampleCollector()
    return collector.get_wikimedia_languages(topic)

# --------- UI Components ---------

def sidebar_controls():
    st.sidebar.title("Tokenizer Comparison")

    # Tokenizers selection
    st.sidebar.subheader("Tokenizers")
    available_factories = get_available_tokenizers()
    tokenizer_names = list(available_factories.keys())

    selected_tokenizers: List[str] = st.sidebar.multiselect(
        "Select tokenizers",
        options=tokenizer_names,
        default=[name for name in tokenizer_names if name.startswith("tiktoken_")] or tokenizer_names[:3],
    )

    # Add custom HF tokenizer
    with st.sidebar.expander("Add Hugging Face tokenizer"):
        if not _hf_enabled():
            st.caption("HF disabled. Set env APP_ENABLE_HF=1 to enable.")
        custom_model = st.sidebar.text_input("Model name (e.g., 'gpt2' or 'bert-base-uncased')", value="")
        if st.sidebar.button("Add tokenizer", use_container_width=True, disabled=not _hf_enabled()):
            if custom_model.strip():
                name = f"hf_{custom_model.strip().split('/')[-1]}"
                # Add a factory so it's only loaded when used
                available_factories[name] = (lambda m=custom_model.strip(): lambda: load_hf_tokenizer(m))()
                st.success(f"Added {custom_model}")

    # Text sources
    st.sidebar.subheader("Text Sources")
    source_mode = st.sidebar.radio(
        "Choose source",
        options=[
            "Sample texts",
            "Wikipedia topic",
            "HP Philosopher's Stone (Ch. 1)",
            "Custom text(s)",
            "PDF URL(s)",
        ],
        index=0,
    )

    texts: Dict[str, str] = {}
    
    if source_mode == "Sample texts":
        samples = get_default_texts()
        languages_to_use = st.sidebar.multiselect(
            "Select languages",
            options=list(samples.keys()),
            default=["english", "german", "chinese"],
        )
        for lang in languages_to_use:
            texts[lang] = samples[lang]
            
    elif source_mode == "Wikipedia topic":
        topic = st.sidebar.text_input("Wikipedia topic (e.g., 'Python_(programming_language)')", value="Python_(programming_language)")
        if topic.strip():
            with st.sidebar.expander("Available languages"):
                languages = get_wikipedia_languages(topic)
                if languages:
                    st.write("Available languages:")
                    
                    # Define default languages (5 most common + German)
                    default_languages = ['en', 'zh', 'hi', 'es', 'ar', 'de']  # English, Chinese, Hindi, Spanish, Arabic, German
                    
                    # Create checkboxes for language selection
                    selected_languages = {}
                    for lang, title in languages.items():
                        # Default to True for common languages, False for others
                        default_value = lang in default_languages
                        if st.checkbox(f"{lang}: {title}", value=default_value, key=f"lang_{lang}"):
                            selected_languages[lang] = title
                    
                    st.write(f"Selected: {len(selected_languages)} languages")
                    
                    if st.sidebar.button("Load selected languages", use_container_width=True):
                        if selected_languages:
                            print(selected_languages)
                            collector = TextSampleCollector()
                            # Load only selected languages
                            for lang_code, title in selected_languages.items():
                                try:
                                    # 1) Fetch the HTML for the current language/topic
                                    html_url = f"https://api.wikimedia.org/core/v1/wikipedia/{lang_code}/page/{title.replace(' ', '_')}/html"
                                    print(html_url)
                                    resp = requests.get(html_url, timeout=10)
                                    resp.raise_for_status()

                                    # 2) Extract plain text
                                    text = collector._extract_text_from_html(resp.text)
                                    # print(text)
                                    if len(text.strip()) > 100:
                                        key = f"{lang_code}_{topic}"
                                        collector.samples[key] = text
                                        st.success(f"Added Wikipedia: {lang_code} – {title}")

                                except requests.HTTPError as http_err:
                                    if http_err.response.status_code == 404:
                                        st.warning(f"[SKIP] Page not found: {lang_code}/{topic}")
                                    else:
                                        st.error(f"[HTTP ERROR] {lang_code}/{topic}: {http_err}")
                                except Exception as e:
                                    st.error(f"[ERROR] Unexpected error fetching {lang_code}/{topic}: {e}")
                                else:
                                    # 3) Wait to avoid overwhelming the API
                                    time.sleep(0.1)
                            # Store the loaded texts in session state
                            st.session_state['loaded_texts'] = collector.samples
                            st.success(f"Loaded {len(collector.samples)} Wikipedia articles")
                        else:
                            st.warning("Please select at least one language")
                else:
                    st.warning("No Wikipedia articles found for this topic")
        
        # Use loaded texts from session state
        if 'loaded_texts' in st.session_state:
            texts.update(st.session_state['loaded_texts'])
                    
    elif source_mode == "HP Philosopher's Stone (Ch. 1)":
        # Prefer local translations if present; otherwise allow remote fetch
        translations_dir = os.path.join(os.getcwd(), "hpps_translations")
        use_remote = False
        try:
            files = [f for f in os.listdir(translations_dir) if f.endswith('.txt')]
        except Exception:
            files = []

        if files:
            lang_codes = sorted([os.path.splitext(f)[0] for f in files])
            default_langs = [c for c in ["de", "fr", "es", "zh", "ru"] if c in lang_codes] or lang_codes[:3]
            selected_langs = st.sidebar.multiselect(
                "Select HPPS chapter 1 languages (local)",
                options=lang_codes,
                default=default_langs,
            )
            for code in selected_langs:
                path = os.path.join(translations_dir, f"{code}.txt")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        texts[f"hpps_{code}"] = f.read()
                except Exception as e:
                    st.warning(f"Failed to load {code}: {e}")
            use_remote = st.sidebar.checkbox("Use remote files instead", value=False)
        else:
            st.sidebar.info("No local HPPS translations found. Use remote download.")
            use_remote = True

        if use_remote:
            st.sidebar.divider()
            st.sidebar.caption("Remote HPPS translations (GitHub raw URLs)")
            default_base = os.environ.get("HPPS_REMOTE_BASE", "")
            base_url = st.sidebar.text_input("Remote base URL", value=default_base, placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/hpps_translations")
            default_codes = ["de", "fr", "es", "zh", "ru"]
            codes_text = st.sidebar.text_input("Language codes (comma-separated)", value=",".join(default_codes))
            candidate_codes = [c.strip() for c in codes_text.split(",") if c.strip()]
            selected_codes = st.sidebar.multiselect("Select languages (remote)", options=candidate_codes, default=candidate_codes)

            @st.cache_data(show_spinner=False)
            def fetch_remote_hpps(base: str, code: str) -> str:
                url = f"{base.rstrip('/')}/{code}.txt"
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                return resp.text

            if base_url and selected_codes:
                if st.sidebar.button("Download selected languages", use_container_width=True):
                    ok = 0
                    for code in selected_codes:
                        try:
                            content = fetch_remote_hpps(base_url, code)
                            texts[f"hpps_{code}"] = content
                            ok += 1
                            time.sleep(0.05)
                        except Exception as e:
                            st.warning(f"Failed to fetch {code}: {e}")
                    if ok:
                        st.success(f"Loaded {ok} remote translations")

    elif source_mode == "Custom text(s)":
        # Allow adding multiple custom texts
        if 'custom_texts' not in st.session_state:
            st.session_state['custom_texts'] = []  # list of dicts {name, text}

        name = st.sidebar.text_input("Text label", value=f"custom_{len(st.session_state['custom_texts'])+1}")
        custom_text = st.sidebar.text_area("Enter text", height=180)
        if st.sidebar.button("Add text", use_container_width=True):
            if custom_text.strip():
                st.session_state['custom_texts'].append({"name": name.strip() or f"custom_{len(st.session_state['custom_texts'])+1}", "text": custom_text})
                st.success(f"Added custom text: {name}")
            else:
                st.warning("Text is empty; nothing added.")

        # Show current items and include them
        if st.session_state['custom_texts']:
            st.sidebar.caption(f"Custom texts: {len(st.session_state['custom_texts'])}")
            for item in st.session_state['custom_texts']:
                texts[item['name']] = item['text']

    elif source_mode == "PDF URL(s)":
        # Allow adding multiple PDFs by URL
        if 'custom_pdfs' not in st.session_state:
            st.session_state['custom_pdfs'] = {}  # name -> text

        pdf_name = st.sidebar.text_input("PDF name/label", value=f"document_{len(st.session_state['custom_pdfs'])+1}")
        pdf_url = st.sidebar.text_input("PDF URL", value="")
        if st.sidebar.button("Add PDF", use_container_width=True):
            if pdf_url.strip():
                collector = TextSampleCollector()
                with st.spinner("Loading PDF..."):
                    collector.add_pdf_from_url(pdf_name, pdf_url)
                    if pdf_name in collector.samples:
                        st.session_state['custom_pdfs'][pdf_name] = collector.samples[pdf_name]
                        st.success(f"Added PDF: {pdf_name}")
                    else:
                        st.warning("No text extracted from PDF.")
            else:
                st.warning("Please provide a PDF URL.")

        if st.session_state['custom_pdfs']:
            st.sidebar.caption(f"PDFs added: {len(st.session_state['custom_pdfs'])}")
            for name, text in st.session_state['custom_pdfs'].items():
                texts[name] = text

    # Show loaded texts count
    if texts:
        st.sidebar.success(f"Ready to analyze {len(texts)} text samples")

    run_button = st.sidebar.button("Run analysis", type="primary")
    return selected_tokenizers, available_factories, texts, run_button

# --------- Analysis and plotting ---------

def run_comparison(selected_tokenizers: List[str], factories: Dict[str, callable], texts: Dict[str, str]) -> pd.DataFrame:
    analyzer = TokenizerAnalyzer()

    # Add tokenizers lazily via factories
    for name in selected_tokenizers:
        try:
            instance = factories[name]()
            analyzer.add_tokenizer(instance)
        except Exception as e:
            st.warning(f"Skipping tokenizer {name}: {e}")

    # Add texts
    for name, text in texts.items():
        analyzer.add_text_sample(name, text)

    # Progress bar
    progress = st.progress(0.0, text="Analyzing...")

    total_steps = max(1, len(texts) * len(selected_tokenizers))
    step = 0

    for text_name, text in analyzer.text_samples.items():
        for tokenizer_name, tokenizer in analyzer.tokenizers.items():
            analyzer.results.setdefault(text_name, {})
            metrics = analyzer.analyze_tokenizer_on_text(tokenizer, text)
            analyzer.results[text_name][tokenizer_name] = metrics
            step += 1
            progress.progress(step / total_steps, text=f"{tokenizer_name} on {text_name} ({step}/{total_steps})")

    progress.empty()
    return analyzer.get_comparison_dataframe(), analyzer


def plot_overview(df: pd.DataFrame):
    if df.empty:
        st.info("No data to display.")
        return

    st.subheader("Overview: Tokens and Efficiency")

    # Layout
    col1, col2 = st.columns(2)

    # Bar: total tokens per language per tokenizer
    with col1:
        st.caption("Tokens per language by tokenizer")
        pivot_tokens = df.pivot(index='text_name', columns='tokenizer_name', values='total_tokens')
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        pivot_tokens.plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Total tokens')
        ax1.set_xlabel('Language')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(loc='best')
        st.pyplot(fig1)

    # Heatmap: efficiency score
    with col2:
        st.caption("Efficiency score heatmap (higher is better)")
        pivot_eff = df.pivot(index='text_name', columns='tokenizer_name', values='efficiency_score')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot_eff, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f', ax=ax2)
        ax2.set_xlabel('Tokenizer')
        ax2.set_ylabel('Language')
        st.pyplot(fig2)

    # Timing
    st.caption("Tokenization time (seconds)")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    pivot_time = df.pivot(index='text_name', columns='tokenizer_name', values='tokenization_time')
    pivot_time.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Seconds')
    ax3.set_xlabel('Language')
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)


def detail_section(analyzer: TokenizerAnalyzer):
    if not analyzer.results:
        return

    st.subheader("Details by Language and Tokenizer")
    languages = list(analyzer.results.keys())
    
    # Store selected language and tokenizer in session state to persist selections
    if 'selected_language' not in st.session_state:
        st.session_state['selected_language'] = languages[0] if languages else None
    if 'selected_tokenizer' not in st.session_state:
        st.session_state['selected_tokenizer'] = None
    
    # Update selected language
    lang = st.selectbox("Select language", options=languages, index=languages.index(st.session_state['selected_language']) if st.session_state['selected_language'] in languages else 0, key="lang_select")
    st.session_state['selected_language'] = lang
    
    # Update available tokenizers for selected language
    tokenizers_for_lang = list(analyzer.results[lang].keys())
    
    # Update selected tokenizer
    if st.session_state['selected_tokenizer'] not in tokenizers_for_lang:
        st.session_state['selected_tokenizer'] = tokenizers_for_lang[0] if tokenizers_for_lang else None
    
    tok = st.selectbox("Select tokenizer", options=tokenizers_for_lang, index=tokenizers_for_lang.index(st.session_state['selected_tokenizer']) if st.session_state['selected_tokenizer'] in tokenizers_for_lang else 0, key="tok_select")
    st.session_state['selected_tokenizer'] = tok

    metrics = analyzer.results[lang][tok]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total tokens", f"{metrics.total_tokens}")
    k2.metric("Unique tokens", f"{metrics.unique_tokens}")
    k3.metric("Compression (chars/token)", f"{metrics.compression_ratio:.2f}")
    k4.metric("Time (s)", f"{metrics.tokenization_time:.4f}")

    # Most/least/longest
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Most used tokens**")
        most_df = pd.DataFrame(metrics.most_common_tokens, columns=["token", "count"]) if metrics.most_common_tokens else pd.DataFrame(columns=["token", "count"])
        st.dataframe(most_df, use_container_width=True, height=240)
    with colB:
        st.markdown("**Least used tokens**")
        least_df = pd.DataFrame(metrics.least_common_tokens, columns=["token", "count"]) if metrics.least_common_tokens else pd.DataFrame(columns=["token", "count"])
        st.dataframe(least_df, use_container_width=True, height=240)
    with colC:
        st.markdown("**Token length distribution**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(metrics.token_lengths, bins=30, color='steelblue', alpha=0.8)
        ax.axvline(np.mean(metrics.token_lengths) if metrics.token_lengths else 0, color='red', linestyle='--', label='mean')
        ax.set_xlabel('Token length')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

    st.markdown("**Longest and shortest tokens**")
    long_token, long_len = metrics.longest_token
    short_token, short_len = metrics.shortest_token
    c1, c2 = st.columns(2)
    with c1:
        st.code(f"Longest token ({long_len}): {long_token}")
    with c2:
        st.code(f"Shortest token ({short_len}): {short_token}")


def advanced_analysis(df: pd.DataFrame):
    """Advanced analysis section with statistical insights."""
    if df.empty:
        return
        
    st.subheader("Advanced Analysis")
    
    # Statistical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best performing tokenizers**")
        best_by_efficiency = df.loc[df.groupby('text_name')['efficiency_score'].idxmax()]
        st.dataframe(best_by_efficiency[['text_name', 'tokenizer_name', 'efficiency_score']].round(4))
    
    with col2:
        st.markdown("**Fastest tokenizers**")
        fastest = df.loc[df.groupby('text_name')['tokenization_time'].idxmin()]
        st.dataframe(fastest[['text_name', 'tokenizer_name', 'tokenization_time']].round(4))
    
    # Correlation analysis
    st.markdown("**Correlation Analysis**")
    corr_matrix = df[['total_tokens', 'compression_ratio', 'efficiency_score', 'tokenization_time']].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)


# --------- Main app ---------

def main():
    st.title("Tokenizer Comparison Playground")
    st.write(
        "Compare SOTA tokenizers across languages. Add Wikipedia topics, Harry Potter Ch. 1 translations, PDFs, and more!"
    )

    selected_tokenizers, factories, texts, run_button = sidebar_controls()

    if run_button:
        if not selected_tokenizers:
            st.warning("Please select at least one tokenizer.")
            return
        if not texts:
            st.warning("Please provide at least one text sample.")
            return

        with st.spinner("Running analysis..."):
            df, analyzer = run_comparison(selected_tokenizers, factories, texts)
            
            # Store results in session state
            st.session_state['analysis_df'] = df
            st.session_state['analyzer'] = analyzer

        st.success("Analysis complete!")

        # Results table
        st.subheader("Results")
        st.dataframe(df.round(4), use_container_width=True)

        # Downloads
        colx, coly = st.columns(2)
        with colx:
            st.download_button(
                label="Download results CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="tokenizer_comparison_results.csv",
                mime="text/csv",
            )
        with coly:
            st.download_button(
                label="Download results JSON",
                data=df.to_json(orient="records").encode("utf-8"),
                file_name="tokenizer_comparison_results.json",
                mime="application/json",
            )

        # Plots
        plot_overview(df)

        # Details
        detail_section(analyzer)
        
        # Advanced analysis
        advanced_analysis(df)
        
    # Show results even if not running new analysis (for UI navigation)
    elif 'analysis_df' in st.session_state and 'analyzer' in st.session_state:
        df = st.session_state['analysis_df']
        analyzer = st.session_state['analyzer']
        
        st.subheader("Previous Analysis Results")
        st.dataframe(df.round(4), use_container_width=True)
        
        # Plots
        plot_overview(df)
        
        # Details
        detail_section(analyzer)
        
        # Advanced analysis
        advanced_analysis(df)
    else:
        st.info("Configure tokenizers and texts in the sidebar, then click 'Run analysis'.")
        
        # Show available features
        with st.expander("Available Features"):
            st.markdown("""
            **Text Sources:**
            - **Sample texts**: Pre-defined multi-language samples
            - **Wikipedia topic**: Fetch articles in all available languages for any topic
            - **HP Philosopher's Stone (Ch. 1)**: Local translations in multiple languages
            - **Custom text(s)**: Add multiple custom text inputs
            - **PDF URL(s)**: Add multiple PDFs by URL
            
            **SOTA Tokenizers:**
            - **Tiktoken**: OpenAI's cl100k_base (GPT-4) and o200k_base (GPT-4o)
            - **Llama 3**: Meta's latest model
            - **Llama 2**: Meta's previous generation
            - **Qwen 3**: Alibaba's latest model
            - **DeepSeek**: Specialized for code
            - **Gemini/Gemma**: Google's models
            - **BERT/RoBERTa**: Classic transformer models
            - **GPT-2**: OpenAI's earlier model
            """)


if __name__ == "__main__":
    main()
