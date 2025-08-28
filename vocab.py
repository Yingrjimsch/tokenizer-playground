# /// script
# dependencies = [
#     "tiktoken",
#     "transformers",
# ]
# ///

"""
vocab.py

Utility to:
- Load a tokenizer (same options as the Streamlit app).
- Extract and filter its vocabulary by language-specific characters.
- Simulate random token draws to try to form a target word using chunks of 3 characters per token.

Usage examples:
  python vocab.py --tokenizer tiktoken_cl100k_base --language english --word Meat --max-tries 100000
  python vocab.py --tokenizer hf_gpt2 --language german --word Straße --seed 42 --max-tries 500000

Notes:
- Hugging Face tokenizers may require network to download unless cached locally.
- Filtering removes whitespace-only, control, and special marker prefixes (e.g., 'Ġ', '▁', '##').
"""

from __future__ import annotations

import argparse
import random
import time
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

# Optional deps; import lazily and handle absence gracefully
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore


# ---------------------- Language character filters ----------------------

def _is_latin_letter(ch: str) -> bool:
    # Accept Latin ranges (Basic + Latin-1 Supplement) and treat letters only
    if not ch:
        return False
    cat = unicodedata.category(ch)
    if not cat.startswith("L"):
        return False
    code = ord(ch)
    return (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x017F) or (0x0180 <= code <= 0x024F)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
    )


def _is_hiragana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x309F


def _is_katakana(ch: str) -> bool:
    code = ord(ch)
    return (0x30A0 <= code <= 0x30FF) or (0x31F0 <= code <= 0x31FF)  # incl. small kana


def _is_long_vowel_mark(ch: str) -> bool:
    # Katakana-Hiragana prolonged sound mark
    return ord(ch) == 0x30FC


def allowed_char(lang: str, ch: str) -> bool:
    """Return True if character is allowed for the language."""
    if ch.isspace():
        return False

    lang = lang.lower()
    if lang in {"english", "german", "french", "spanish"}:
        return _is_latin_letter(ch)
    if lang == "chinese":
        return _is_cjk(ch)
    if lang == "japanese":
        return _is_cjk(ch) or _is_hiragana(ch) or _is_katakana(ch) or _is_long_vowel_mark(ch)
    # Default: letters only
    return unicodedata.category(ch).startswith("L")


def normalize_for_filtering(token: str) -> str:
    """Normalize a token string before filtering and comparison.

    - Remove SentencePiece/BBPE/WordPiece artifacts: leading '▁'/'Ġ', '##'.
    - Strip whitespace.
    - Keep original accents; lowercase for Latin scripts to be case-insensitive.
    """
    if token is None:
        return ""
    # Remove common tokenizer markers
    if token.startswith("▁") or token.startswith("Ġ"):
        token = token[1:]
    if token.startswith("##"):
        token = token[2:]
    token = token.strip()
    return token


def filtered_vocab_tokens(raw_tokens: Iterable[str], language: str) -> List[str]:
    out: List[str] = []
    lang = language.lower()
    for tok in raw_tokens:
        norm = normalize_for_filtering(tok)
        if not norm:
            continue
        # Remove any remaining internal spaces or control characters
        if any(c.isspace() or unicodedata.category(c).startswith("C") for c in norm):
            continue
        # Filter: must contain only allowed letters for the language
        if all(allowed_char(lang, c) for c in norm):
            out.append(norm)
    return out


# ---------------------- Tokenizer loading and vocab extraction ----------------------

def load_tokenizer(spec: str):
    """Load tokenizer by spec.

    Accepted forms:
    - tiktoken_cl100k_base, tiktoken_o200k_base
    - hf_gpt2, hf_bert-base-uncased, hf_roberta-base, etc.
    - raw forms: tiktoken:cl100k_base, hf:gpt2
    Returns: (kind, object)
    kind in {"tiktoken", "hf"}
    """
    s = spec.strip()
    if s.startswith("tiktoken:"):
        enc = s.split(":", 1)[1]
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed")
        return "tiktoken", tiktoken.get_encoding(enc)
    if s.startswith("hf:"):
        model = s.split(":", 1)[1]
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed")
        return "hf", AutoTokenizer.from_pretrained(model)
    # Streamlit-style keys
    if s.startswith("tiktoken_"):
        enc = s.split("tiktoken_", 1)[1]
        if tiktoken is None:
            raise RuntimeError("tiktoken is not installed")
        return "tiktoken", tiktoken.get_encoding(enc)
    if s.startswith("hf_"):
        suffix = s.split("hf_", 1)[1]
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed")
        # Map Streamlit keys to full model names when needed
        suffix_to_full = {
            "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
            "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
            "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
            "deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "gemma-2b": "google/gemma-2b",
            # Common ones are already direct names
            "gpt2": "gpt2",
            "bert-base-uncased": "bert-base-uncased",
            "roberta-base": "roberta-base",
        }
        model = suffix_to_full.get(suffix, suffix)
        return "hf", AutoTokenizer.from_pretrained(model)
    # Direct model name assumed HF
    if AutoTokenizer is not None:
        try:
            return "hf", AutoTokenizer.from_pretrained(s)
        except Exception:
            pass
    raise ValueError(f"Unrecognized tokenizer spec: {spec}")


def get_raw_vocab_tokens(kind: str, obj) -> List[str]:
    """Return raw (un-normalized) tokens from tokenizer."""
    if kind == "tiktoken":
        enc = obj
        tokens: List[str] = []
        for i in range(enc.n_vocab):
            try:
                # decode([id]) returns str; robust for most ids
                s = enc.decode([i])
            except Exception:
                # Fallback to bytes; ignore undecodable sequences
                try:
                    s = enc.decode_single_token_bytes(i).decode("utf-8", errors="ignore")
                except Exception:
                    s = ""
            tokens.append(s)
        return tokens
    elif kind == "hf":
        tok = obj
        # get_vocab returns dict[str, int]; we want strings
        try:
            vocab_dict = tok.get_vocab()
            return list(vocab_dict.keys())
        except Exception:
            # Fallback for some tokenizers
            if hasattr(tok, "vocab") and isinstance(tok.vocab, dict):
                return list(tok.vocab.keys())
            raise RuntimeError("Could not extract HF tokenizer vocabulary")
    else:
        raise ValueError(f"Unsupported tokenizer kind: {kind}")


# ---------------------- Simulation ----------------------

def chunk_word(word: str, chunk_size: int = 3) -> List[str]:
    return [word[i : i + chunk_size] for i in range(0, len(word), chunk_size)]


@dataclass
class SimulationResult:
    success: bool
    tries: int
    elapsed_sec: float
    per_try_probability: float
    expected_tries: Optional[float]
    chunks: List[str]
    missing_chunks: List[str]
    token_count: int


def simulate(word: str, tokens: Sequence[str], language: str, chunk_size: int, max_tries: int, seed: Optional[int] = None) -> SimulationResult:
    # Normalize target for comparison similar to tokens
    lang = language.lower()
    target = word.strip()
    if lang in {"english", "german", "french", "spanish"}:
        target = target.lower()

    chunks = chunk_word(target, chunk_size)
    token_set = set(t.lower() if lang in {"english", "german", "french", "spanish"} else t for t in tokens)

    missing = [c for c in chunks if c not in token_set]
    K = len(tokens)
    per_try_p = 0.0
    expected = None
    if K > 0 and not missing:
        per_try_p = (1.0 / K) ** len(chunks)
        expected = 1.0 / per_try_p if per_try_p > 0 else None

    if seed is not None:
        random.seed(seed)

    start = time.perf_counter()
    tries = 0
    success = False

    if missing:
        # Impossible to hit exactly via this chunking and tokens
        elapsed = time.perf_counter() - start
        return SimulationResult(
            success=False,
            tries=0,
            elapsed_sec=elapsed,
            per_try_probability=per_try_p,
            expected_tries=expected,
            chunks=chunks,
            missing_chunks=missing,
            token_count=K,
        )

    # Run capped simulation of independent uniform picks with replacement
    while tries < max_tries:
        tries += 1
        pick = [tokens[random.randrange(K)] for _ in range(len(chunks))]
        cand = "".join(pick)
        cand_cmp = cand.lower() if lang in {"english", "german", "french", "spanish"} else cand
        if cand_cmp == target:
            success = True
            break

    elapsed = time.perf_counter() - start
    return SimulationResult(
        success=success,
        tries=tries,
        elapsed_sec=elapsed,
        per_try_probability=per_try_p,
        expected_tries=expected,
        chunks=chunks,
        missing_chunks=missing,
        token_count=K,
    )


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="Filter tokenizer vocabulary by language and simulate random draws to match a word.")
    ap.add_argument("--tokenizer", required=True, help=(
        "Tokenizer spec, e.g. 'tiktoken_cl100k_base', 'tiktoken_o200k_base', 'hf_gpt2', 'hf_bert-base-uncased', "
        "or raw forms 'tiktoken:cl100k_base', 'hf:gpt2'"
    ))
    ap.add_argument("--language", required=True, choices=[
        "english", "german", "french", "spanish", "chinese", "japanese"
    ], help="Language to filter by")
    ap.add_argument("--word", required=True, help="Target word to form")
    ap.add_argument("--chunk-size", type=int, default=3, help="Characters per token (default: 3)")
    ap.add_argument("--max-tries", type=int, default=100000, help="Maximum tries for simulation")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")

    args = ap.parse_args()

    # Load tokenizer
    try:
        kind, obj = load_tokenizer(args.tokenizer)
    except Exception as e:
        raise SystemExit(f"Failed to load tokenizer '{args.tokenizer}': {e}")

    # Extract raw vocab
    try:
        raw_tokens = get_raw_vocab_tokens(kind, obj)
    except Exception as e:
        raise SystemExit(f"Failed to extract vocabulary: {e}")

    # Filter
    filtered = filtered_vocab_tokens(raw_tokens, args.language)
    # Case-insensitive for Latin scripts in simulation/comparison
    if args.language in {"english", "german", "french", "spanish"}:
        filtered = [t.lower() for t in filtered]

    # Report counts
    print(f"Tokenizer: {args.tokenizer} (kind={kind})")
    print(f"Language: {args.language}")
    print(f"Filtered tokens (letters in language only): {len(filtered)}")

    # Simulate
    result = simulate(args.word, filtered, args.language, args.chunk_size, args.max_tries, args.seed)

    # Overview
    print("--- Simulation Overview ---")
    print(f"Target word: {args.word}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunks: {result.chunks}")
    if result.missing_chunks:
        print(f"Missing chunks (not in filtered vocab): {result.missing_chunks}")
    print(f"Per-try success probability: {result.per_try_probability:.12g}")
    if result.expected_tries is not None:
        print(f"Expected tries (theoretical): {result.expected_tries:.3f}")
    else:
        print("Expected tries (theoretical): N/A")
    print(f"Simulation tries: {result.tries}")
    print(f"Time elapsed: {result.elapsed_sec:.6f} s")
    print(f"Success: {result.success}")


if __name__ == "__main__":
    main()
