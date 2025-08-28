"""
Tokenizer Comparison Framework
A comprehensive framework for comparing different tokenizers across multiple languages.
"""

import os
import time
import requests
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Tokenizer imports
# Make optional imports lazy to allow the app to run without all extras
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
import PyPDF2
import io
from urllib.parse import urlparse, parse_qs

@dataclass
class TokenizerMetrics:
    """Container for tokenizer analysis metrics."""
    total_tokens: int
    unique_tokens: int
    vocabulary_size: int
    tokenization_time: float
    token_frequencies: Counter
    token_lengths: List[int]
    most_common_tokens: List[Tuple[str, int]]
    least_common_tokens: List[Tuple[str, int]]
    longest_token: Tuple[str, int]
    shortest_token: Tuple[str, int]
    compression_ratio: float  # characters / tokens
    efficiency_score: float  # custom metric combining various factors

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the given text and return list of tokens."""
        pass
    
    def get_vocabulary_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return -1  # Default for unknown vocab size

class TiktokenTokenizer(BaseTokenizer):
    """Wrapper for tiktoken tokenizers."""
    
    def __init__(self, encoding_name: str):
        super().__init__(f"tiktoken_{encoding_name}")
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed")
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using tiktoken."""
        tokens = self.encoding.encode(text)
        return [self.encoding.decode([token]) for token in tokens]
    
    def get_vocabulary_size(self) -> int:
        """Return vocabulary size."""
        return self.encoding.n_vocab

class HuggingFaceTokenizer(BaseTokenizer):
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(self, model_name: str):
        super().__init__(f"hf_{model_name.split('/')[-1]}")
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using HuggingFace tokenizer."""
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.convert_ids_to_tokens(tokens)
    
    def get_vocabulary_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

class SimpleTokenizer(BaseTokenizer):
    """Simple tokenizer for comparison (splits on whitespace)."""
    
    def __init__(self, name: str = "simple"):
        super().__init__(name)
    
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace-based tokenization."""
        return text.split()

class TokenizerAnalyzer:
    """Main class for analyzing and comparing tokenizers."""
    
    def __init__(self):
        self.tokenizers: Dict[str, BaseTokenizer] = {}
        self.text_samples: Dict[str, str] = {}
        self.results: Dict[str, Dict[str, TokenizerMetrics]] = {}
    
    def add_tokenizer(self, tokenizer: BaseTokenizer):
        """Add a tokenizer to the analyzer."""
        self.tokenizers[tokenizer.name] = tokenizer
    
    def add_text_sample(self, name: str, text: str):
        """Add a text sample for analysis."""
        self.text_samples[name] = text
    
    def analyze_tokenizer_on_text(self, tokenizer: BaseTokenizer, text: str) -> TokenizerMetrics:
        """Analyze a single tokenizer on a single text."""
        start_time = time.time()
        tokens = tokenizer.tokenize(text)
        tokenization_time = time.time() - start_time
        
        # Basic metrics
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        vocabulary_size = tokenizer.get_vocabulary_size()
        
        # Frequency analysis
        token_frequencies = Counter(tokens)
        most_common = token_frequencies.most_common(10)
        least_common = token_frequencies.most_common()[:-11:-1]  # Last 10
        
        # Token length analysis
        token_lengths = [len(token) for token in tokens]
        longest_token = max(tokens, key=len) if tokens else ("", 0)
        shortest_token = min(tokens, key=len) if tokens else ("", 0)
        
        # Compression ratio (characters per token)
        compression_ratio = len(text) / total_tokens if total_tokens > 0 else 0
        
        # Efficiency score (custom metric)
        efficiency_score = self._calculate_efficiency_score(
            total_tokens, unique_tokens, vocabulary_size, 
            tokenization_time, compression_ratio
        )
        
        return TokenizerMetrics(
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            vocabulary_size=vocabulary_size,
            tokenization_time=tokenization_time,
            token_frequencies=token_frequencies,
            token_lengths=token_lengths,
            most_common_tokens=most_common,
            least_common_tokens=least_common,
            longest_token=(longest_token, len(longest_token)),
            shortest_token=(shortest_token, len(shortest_token)),
            compression_ratio=compression_ratio,
            efficiency_score=efficiency_score
        )
    
    def _calculate_efficiency_score(self, total_tokens: int, unique_tokens: int, 
                                  vocab_size: int, tokenization_time: float, 
                                  compression_ratio: float) -> float:
        """Calculate a custom efficiency score."""
        # Higher score = better efficiency
        # Factors: compression ratio, speed, vocabulary utilization
        
        # Normalize factors (assuming reasonable ranges)
        compression_score = min(compression_ratio / 10, 1.0)  # Cap at 10 chars/token
        speed_score = max(0, 1 - tokenization_time / 10)  # Cap at 10 seconds
        vocab_utilization = unique_tokens / max(vocab_size, 1) if vocab_size > 0 else 0
        
        # Weighted combination
        efficiency = (0.7 * compression_score + 
                     0.1 * speed_score + 
                     0.2 * vocab_utilization)
        
        return efficiency
    
    def run_analysis(self) -> Dict[str, Dict[str, TokenizerMetrics]]:
        """Run analysis on all tokenizers and text samples."""
        results = {}
        
        for text_name, text in self.text_samples.items():
            results[text_name] = {}
            for tokenizer_name, tokenizer in self.tokenizers.items():
                print(f"Analyzing {tokenizer_name} on {text_name}...")
                metrics = self.analyze_tokenizer_on_text(tokenizer, text)
                results[text_name][tokenizer_name] = metrics
        
        self.results = results
        return results
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for easy analysis."""
        data = []
        
        for text_name, tokenizer_results in self.results.items():
            for tokenizer_name, metrics in tokenizer_results.items():
                data.append({
                    'text_name': text_name,
                    'tokenizer_name': tokenizer_name,
                    'total_tokens': metrics.total_tokens,
                    'unique_tokens': metrics.unique_tokens,
                    'vocabulary_size': metrics.vocabulary_size,
                    'tokenization_time': metrics.tokenization_time,
                    'compression_ratio': metrics.compression_ratio,
                    'efficiency_score': metrics.efficiency_score,
                    'avg_token_length': np.mean(metrics.token_lengths) if metrics.token_lengths else 0,
                    'max_token_length': max(metrics.token_lengths) if metrics.token_lengths else 0,
                    'min_token_length': min(metrics.token_lengths) if metrics.token_lengths else 0
                })
        
        return pd.DataFrame(data)

class TokenizerVisualizer:
    """Class for creating visualizations of tokenizer comparisons."""
    
    def __init__(self, analyzer: TokenizerAnalyzer):
        self.analyzer = analyzer
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup matplotlib style for better plots."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_token_count_comparison(self, figsize=(12, 8)):
        """Plot token count comparison across tokenizers and languages."""
        df = self.analyzer.get_comparison_dataframe()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Total tokens by tokenizer
        tokenizer_totals = df.groupby('tokenizer_name')['total_tokens'].sum()
        ax1.bar(tokenizer_totals.index, tokenizer_totals.values, color='skyblue')
        ax1.set_title('Total Tokens by Tokenizer')
        ax1.set_ylabel('Total Tokens')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Tokens per language/text
        pivot_data = df.pivot(index='text_name', columns='tokenizer_name', values='total_tokens')
        pivot_data.plot(kind='bar', ax=ax2)
        ax2.set_title('Tokens per Language/Text by Tokenizer')
        ax2.set_ylabel('Total Tokens')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_efficiency_metrics(self, figsize=(15, 10)):
        """Plot efficiency metrics comparison."""
        df = self.analyzer.get_comparison_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Compression ratio
        pivot_compression = df.pivot(index='text_name', columns='tokenizer_name', values='compression_ratio')
        pivot_compression.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Compression Ratio (Chars/Token)')
        axes[0].set_ylabel('Compression Ratio')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Efficiency score
        pivot_efficiency = df.pivot(index='text_name', columns='tokenizer_name', values='efficiency_score')
        pivot_efficiency.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Efficiency Score')
        axes[1].set_ylabel('Efficiency Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Tokenization time
        pivot_time = df.pivot(index='text_name', columns='tokenizer_name', values='tokenization_time')
        pivot_time.plot(kind='bar', ax=axes[2])
        axes[2].set_title('Tokenization Time')
        axes[2].set_ylabel('Time (seconds)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Vocabulary utilization
        df['vocab_utilization'] = df['unique_tokens'] / df['vocabulary_size'].replace(0, 1)
        pivot_vocab = df.pivot(index='text_name', columns='tokenizer_name', values='vocab_utilization')
        pivot_vocab.plot(kind='bar', ax=axes[3])
        axes[3].set_title('Vocabulary Utilization')
        axes[3].set_ylabel('Utilization Ratio')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_token_frequency_analysis(self, text_name: str, top_n: int = 20, figsize=(15, 10)):
        """Plot detailed token frequency analysis for a specific text."""
        if text_name not in self.analyzer.results:
            print(f"Text '{text_name}' not found in results.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (tokenizer_name, metrics) in enumerate(self.analyzer.results[text_name].items()):
            if i >= 4:  # Limit to 4 tokenizers
                break
            
            # Top tokens frequency
            top_tokens = metrics.most_common_tokens[:top_n]
            if top_tokens:
                tokens, counts = zip(*top_tokens)
                axes[i].bar(range(len(tokens)), counts, color=f'C{i}')
                axes[i].set_title(f'{tokenizer_name} - Top {top_n} Tokens')
                axes[i].set_ylabel('Frequency')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add token labels (truncated for readability)
                token_labels = [token[:10] + '...' if len(token) > 10 else token for token in tokens]
                axes[i].set_xticks(range(len(tokens)))
                axes[i].set_xticklabels(token_labels)
        
        plt.tight_layout()
        plt.show()
    
    def plot_token_length_distribution(self, text_name: str, figsize=(15, 10)):
        """Plot token length distribution for different tokenizers."""
        if text_name not in self.analyzer.results:
            print(f"Text '{text_name}' not found in results.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (tokenizer_name, metrics) in enumerate(self.analyzer.results[text_name].items()):
            if i >= 4:  # Limit to 4 tokenizers
                break
            
            axes[i].hist(metrics.token_lengths, bins=30, alpha=0.7, color=f'C{i}')
            axes[i].set_title(f'{tokenizer_name} - Token Length Distribution')
            axes[i].set_xlabel('Token Length')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(np.mean(metrics.token_lengths), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(metrics.token_lengths):.2f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

class TextSampleCollector:
    """Class for collecting text samples from various sources."""
    
    def __init__(self):
        self.samples = {}
        self.session = requests.Session()
        
    def get_wikimedia_languages(self, topic: str) -> Dict[str, str]:
        """
        Uses the Wikimedia REST API to fetch all available language versions of a Wikipedia article.
        """
        url = (
            f"https://api.wikimedia.org/core/v1/wikipedia/"
            f"en/page/{topic}/links/language"
        )
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Build a mapping of language code to translated title
        return {entry['code']: entry['title'] for entry in data}

    def add_wikimedia_article(self, lang_code: str, title: str):
        """
        Fetches the full page HTML via Wikimedia REST API and extracts text if meaningful.
        """
        html_url = (
            f"https://{lang_code}.wikipedia.org/w/rest.php/v1/page/"
            f"{title.replace(' ', '_')}/html"
        )
        resp = self.session.get(html_url, timeout=10)
        resp.raise_for_status()
        text = self._extract_text_from_html(resp.text)
        if len(text.strip()) > 100:
            key = f"wikipedia_{lang_code}_{title}"
            self.samples[key] = text
            print(f"[ADDED] {lang_code} — {title}")

    def process_topic_all_languages(self, source_lang: str, topic: str):
        """
        Fetches all translation versions via Wikimedia API and retrieves their content.
        """
        languages = self.get_wikimedia_languages(source_lang, topic)
        for code, title in languages.items():
            try:
                self.add_wikimedia_article(code, title)
                time.sleep(0.1)  # stay within safe request perf
            except requests.HTTPError as he:
                if he.response.status_code == 404:
                    print(f"[SKIP] '{topic}' not found in {code}")
                else:
                    print(f"[ERROR] {code}/{title}: {he}")
            except Exception as e:
                print(f"[ERROR] {code}/{title}: {e}")

    def _extract_text_from_html(self, html: str) -> str:
        import re
        return re.sub(r"<[^>]+>", "", html)


    # def add_wikipedia_sample(self, language_code: str, topic: str = "Python_(programming_language)"):
    #     """Add a Wikipedia article sample."""
    #     try:
    #         url = f"https://{language_code}.wikipedia.org/api/rest_v1/page/summary/{topic}"
    #         response = requests.get(url, timeout=10)
    #         response.raise_for_status()
    #         data = response.json()
            
    #         # Get the full article content
    #         content_url = f"https://{language_code}.wikipedia.org/api/rest_v1/page/html/{topic}"
    #         content_response = requests.get(content_url, timeout=10)
    #         content_response.raise_for_status()
            
    #         # Extract text content (simplified)
    #         text = self._extract_text_from_html(content_response.text)
    #         self.samples[f"wikipedia_{language_code}_{topic}"] = text
    #         print(f"Added Wikipedia sample: {language_code} - {topic}")
            
    #     except Exception as e:
    #         print(f"Error fetching Wikipedia sample for {language_code}: {e}")
    
    # def get_wikipedia_languages_for_topic(self, topic: str) -> Dict[str, str]:
    #     """Get available languages for a Wikipedia topic using Wikimedia API."""
    #     languages = {}
        
    #     # Use Wikimedia API to get all available languages
    #     try:
    #         # First, get the page info from English Wikipedia
    #         url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    #         response = requests.get(url, timeout=10)
    #         if response.status_code == 200:
    #             data = response.json()
    #             if 'content_urls' in data and 'desktop' in data['content_urls']:
    #                 # Get the page ID
    #                 page_url = data['content_urls']['desktop']['page']
    #                 page_id = page_url.split('/')[-1]
                    
    #                 # Now get all language links using Wikimedia API
    #                 wikimedia_url = f"https://en.wikipedia.org/api/rest_v1/page/links/{topic}"
    #                 links_response = requests.get(wikimedia_url, timeout=15)
                    
    #                 if links_response.status_code == 200:
    #                     links_data = links_response.json()
    #                     if 'links' in links_data:
    #                         for link in links_data['links']:
    #                             if 'lang' in link and 'title' in link:
    #                                 lang_code = link['lang']
    #                                 title = link['title']
    #                                 languages[lang_code] = title
                    
    #                 # If no links found, try alternative method
    #                 if not languages:
    #                     # Try to get language links from the page content
    #                     content_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic}"
    #                     content_response = requests.get(content_url, timeout=10)
    #                     if content_response.status_code == 200:
    #                         # Extract language links from HTML (simplified)
    #                         html_content = content_response.text
    #                         # Look for language links in the HTML
    #                         import re
    #                         lang_links = re.findall(r'href="https://([a-z]{2,3})\.wikipedia\.org/wiki/[^"]*"', html_content)
    #                         for lang in set(lang_links):
    #                             try:
    #                                 # Verify the page exists in this language
    #                                 verify_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{topic}"
    #                                 verify_response = requests.get(verify_url, timeout=5)
    #                                 if verify_response.status_code == 200:
    #                                     verify_data = verify_response.json()
    #                                     if 'title' in verify_data:
    #                                         languages[lang] = verify_data['title']
    #                             except:
    #                                 continue
                                    
    #     except Exception as e:
    #         print(f"Error fetching language links: {e}")
        
    #     # If still no languages found, fall back to common languages
    #     if not languages:
    #         common_langs = [
    #             'en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'ja', 'zh', 'ko', 'ar', 'hi',
    #             'nl', 'pl', 'sv', 'da', 'no', 'fi', 'tr', 'he', 'th', 'vi', 'id', 'ms',
    #             'ca', 'eu', 'gl', 'cy', 'ga', 'mt', 'sq', 'mk', 'bg', 'hr', 'sr', 'sl',
    #             'sk', 'cs', 'hu', 'ro', 'et', 'lv', 'lt', 'uk', 'be', 'ka', 'hy', 'az',
    #             'kk', 'ky', 'uz', 'tg', 'fa', 'ps', 'ur', 'bn', 'si', 'my', 'km', 'lo',
    #             'ne', 'ta', 'te', 'kn', 'ml', 'gu', 'pa', 'or', 'as', 'mr', 'sa', 'dv'
    #         ]
            
    #         for lang in common_langs:
    #             try:
    #                 url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{topic}"
    #                 response = requests.get(url, timeout=5)
    #                 if response.status_code == 200:
    #                     data = response.json()
    #                     if 'title' in data and 'extract' in data:
    #                         languages[lang] = data['title']
    #             except:
    #                 continue
        
    #     return languages
    # def get_wikipedia_languages_for_topic(self, topic: str) -> Dict[str, str]:
    #     languages = {}
    #     url = f"https://en.wikipedia.org/w/rest.php/v1/page/{topic}/links/language"
    #     headers = {'User-Agent': 'YourAppName/1.0 (contact@domain.com)'}
    #     resp = requests.get(url, headers=headers, timeout=10)

    #     if resp.status_code == 200:
    #         data = resp.json()
    #         for lang in data:
    #             languages[lang['code']] = lang.get('title', '')
    #     return languages

    # def add_wikipedia_topic_all_languages(self, topic: str):
    #     """Add a Wikipedia topic in all available languages."""
    #     languages = self.get_wikipedia_languages_for_topic(topic)
        
    #     for lang_code, title in languages.items():
    #         try:
    #             # Get full article content
    #             content_url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/html/{topic}"
    #             content_response = requests.get(content_url, timeout=10)
    #             content_response.raise_for_status()
                
    #             # Extract text content
    #             text = self._extract_text_from_html(content_response.text)
    #             if len(text.strip()) > 100:  # Only add if substantial content
    #                 self.samples[f"wikipedia_{lang_code}_{topic}"] = text
    #                 print(f"Added Wikipedia: {lang_code} - {title}")
                    
    #         except Exception as e:
    #             print(f"Error fetching {lang_code} Wikipedia for {topic}: {e}")
    
    def add_pdf_from_url(self, name: str, url: str):
        """Add text from a PDF URL."""
        try:
            print(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with io.BytesIO(response.content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                if text.strip():
                    self.samples[name] = text
                    print(f"Added PDF sample: {name} ({len(text)} characters)")
                else:
                    print(f"Warning: No text extracted from PDF {name}")
                    
        except Exception as e:
            print(f"Error processing PDF {name} from {url}: {e}")
    
    
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text content from HTML (simplified)."""
        # Remove HTML tags (basic implementation)
        text = re.sub(r'<[^>]+>', '', html_content)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def add_sample_text(self, name: str, text: str):
        """Add a custom text sample."""
        self.samples[name] = text
    
    def get_sample(self, name: str) -> str:
        """Get a specific sample by name."""
        return self.samples.get(name, "")

# Predefined tokenizer configurations with SOTA models
def get_default_tokenizers() -> Dict[str, BaseTokenizer]:
    """Get a dictionary of commonly used tokenizers including SOTA models."""
    tokenizers = {}
    
    # Tiktoken tokenizers (OpenAI)
    try:
        tokenizers["tiktoken_cl100k"] = TiktokenTokenizer("cl100k_base")  # GPT-4, GPT-3.5
    except:
        print("Warning: cl100k_base not available")
    
    try:
        tokenizers["tiktoken_o200k"] = TiktokenTokenizer("o200k_base")  # GPT-4o
    except:
        print("Warning: o200k_base not available")
    
    # HuggingFace tokenizers - SOTA models
    try:
        tokenizers["llama3"] = HuggingFaceTokenizer("meta-llama/Meta-Llama-3-8B-Instruct")
    except:
        print("Warning: Llama 3 tokenizer not available (requires access)")
    
    try:
        tokenizers["llama2"] = HuggingFaceTokenizer("meta-llama/Llama-2-7b-chat-hf")
    except:
        print("Warning: Llama 2 tokenizer not available (requires access)")
    
    try:
        tokenizers["qwen3"] = HuggingFaceTokenizer("Qwen/Qwen3-0.6B")
    except:
        print("Warning: Qwen3 tokenizer not available")
    
    try:
        tokenizers["deepseek"] = HuggingFaceTokenizer("deepseek-ai/deepseek-coder-6.7b-instruct")
    except:
        print("Warning: DeepSeek tokenizer not available")
    
    try:
        tokenizers["gemini"] = HuggingFaceTokenizer("google/gemma-2b")
    except:
        print("Warning: Gemini/Gemma tokenizer not available")
    
    try:
        tokenizers["gpt2"] = HuggingFaceTokenizer("gpt2")
    except:
        print("Warning: GPT-2 tokenizer not available")
    
    try:
        tokenizers["bert"] = HuggingFaceTokenizer("bert-base-uncased")
    except:
        print("Warning: BERT tokenizer not available")
    
    try:
        tokenizers["roberta"] = HuggingFaceTokenizer("roberta-base")
    except:
        print("Warning: RoBERTa tokenizer not available")
    
    # Simple tokenizer for comparison
    tokenizers["simple"] = SimpleTokenizer("simple")
    
    return tokenizers

# Utility functions
def create_sample_texts() -> Dict[str, str]:
    """Create sample texts in different languages for testing."""
    samples = {
        "english": """
        Python is a high-level, interpreted programming language known for its simplicity and readability. 
        It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, 
        including procedural, object-oriented, and functional programming.
        """,
        "german": """
        Python ist eine interpretierte, objektorientierte Programmiersprache. Sie wurde Anfang der 1990er Jahre 
        von Guido van Rossum am Centrum Wiskunde & Informatica in Amsterdam entwickelt. Python unterstützt 
        mehrere Programmierparadigmen, einschließlich der prozeduralen, objektorientierten und funktionalen Programmierung.
        """,
        "french": """
        Python est un langage de programmation de haut niveau, interprété, connu pour sa simplicité et sa lisibilité. 
        Il a été créé par Guido van Rossum et publié pour la première fois en 1991. Python prend en charge plusieurs 
        paradigmes de programmation, y compris la programmation procédurale, orientée objet et fonctionnelle.
        """,
        "spanish": """
        Python es un lenguaje de programación de alto nivel interpretado, conocido por su simplicidad y legibilidad. 
        Fue creado por Guido van Rossum y lanzado por primera vez en 1991. Python admite múltiples paradigmas de programación, 
        incluyendo programación procedural, orientada a objetos y funcional.
        """,
        "chinese": """
        Python是一种高级解释型编程语言，以其简单性和可读性而闻名。它由Guido van Rossum创建，
        并于1991年首次发布。Python支持多种编程范式，包括过程式、面向对象和函数式编程。
        """,
        "japanese": """
        Pythonは、シンプルさと可読性で知られる高レベルインタープリタ型プログラミング言語です。
        グイド・ヴァンロッサムによって作成され、1991年に初めてリリースされました。Pythonは、
        手続き型、オブジェクト指向、関数型プログラミングを含む複数のプログラミングパラダイムをサポートしています。
        """
    }
    return samples
