# Tokenizer Comparison Playground

A comprehensive framework and interactive web application for comparing state-of-the-art tokenizers across multiple languages and text sources.

## 🎯 What This Project Does

This project allows you to analyze and compare different tokenizers to understand:

- **Tokenization efficiency** across languages
- **Compression ratios** (characters per token)
- **Performance metrics** (speed, vocabulary utilization)
- **Language-specific patterns** in tokenization
- **Cross-tokenizer comparisons** on the same content

## 🌟 Key Features

### **SOTA Tokenizers Supported:**
- **OpenAI**: Tiktoken (cl100k_base, o200k_base) - used by GPT-4, GPT-3.5
- **Meta**: Llama 3, Llama 2
- **Alibaba**: Qwen 3
- **DeepSeek**: Specialized for code
- **Google**: Gemini/Gemma models
- **Classic**: BERT, RoBERTa, GPT-2
- **Simple**: Whitespace-based tokenizer for comparison

### **Multi-Language Text Sources:**
- **Sample texts**: Pre-defined multi-language samples
- **Wikipedia articles**: Fetch any topic in all available languages (50+ languages)
- **Harry Potter — Philosopher's Stone (Ch. 1)**: Local translations from `hpps_translations/`
- **Custom texts**: Add multiple of your own text inputs
- **PDF URLs**: Add multiple PDFs by URL with automatic text extraction

### **Comprehensive Analysis:**
- **Token counts** and **unique tokens**
- **Compression ratios** (characters per token)
- **Efficiency scores** (custom metric)
- **Tokenization speed** and **timing**
- **Most/least used tokens**
- **Longest/shortest tokens**
- **Token length distributions**
- **Statistical correlations**

### **Rich Visualizations:**
- **Token count comparisons** across languages and tokenizers
- **Efficiency heatmaps** showing best performing combinations
- **Token frequency analysis** with detailed breakdowns
- **Token length distributions** for each tokenizer
- **Correlation matrices** between different metrics

## 🚀 Quick Start

### **Option 1: Local Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/tokenizer-playground.git
   cd tokenizer-playground
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### **Option 2: Static Hosting (GitHub Pages)**

The project includes GitHub Actions to automatically build a static version using stlite. Simply:

1. **Push to GitHub** - the workflow will automatically build and deploy
2. **Access your app** at `https://yourusername.github.io/tokenizer-playground`

## 📖 How to Use

### **1. Select Tokenizers**
- Choose from the available SOTA tokenizers in the sidebar
- Add custom Hugging Face tokenizers by name
- Default selection includes popular models

### **2. Choose Text Sources**

#### **Sample Texts:**
- Quick comparison with pre-defined multi-language samples
- Select specific languages to analyze

#### **Wikipedia Topics:**
- Enter any Wikipedia topic (e.g., "Python_(programming_language)")
- See all available languages (50+ languages detected)
- Select specific languages with checkboxes
- Smart defaults: 5 most common languages + German pre-selected

#### **Harry Potter — Philosopher's Stone (Ch. 1):**
- Load local Chapter 1 translations from `hpps_translations/`
- Select which language codes to include (e.g., `de`, `fr`, `es`, `zh`, `ru`)

#### **Custom Texts:**
- Enter one or more custom texts with labels
- Great for testing multiple snippets side-by-side

#### **PDF URLs:**
- Add one or more PDF URLs with labels
- Automatic text extraction

### **3. Run Analysis**
- Click "Run analysis" to process all combinations
- View comprehensive results table
- Explore interactive visualizations
- Download results as CSV or JSON

### **4. Explore Results**

#### **Overview Section:**
- Token counts per language and tokenizer
- Efficiency score heatmaps
- Tokenization timing comparisons

#### **Details Section:**
- Select specific language and tokenizer
- View most/least used tokens
- Analyze token length distributions
- See longest and shortest tokens

#### **Advanced Analysis:**
- Best performing tokenizers per language
- Fastest tokenizers comparison
- Statistical correlation analysis

## 🔧 Framework Usage

### **Programmatic Usage:**

```python
from tokenizer_framework import TokenizerAnalyzer, get_default_tokenizers, create_sample_texts

# Initialize analyzer
analyzer = TokenizerAnalyzer()

# Add tokenizers
tokenizers = get_default_tokenizers()
for name, tokenizer in tokenizers.items():
    analyzer.add_tokenizer(tokenizer)

# Add text samples
samples = create_sample_texts()
for name, text in samples.items():
    analyzer.add_text_sample(name, text)

# Run analysis
results = analyzer.run_analysis()

# Get results as DataFrame
df = analyzer.get_comparison_dataframe()
print(df)
```

### **Adding Custom Tokenizers:**

```python
from tokenizer_framework import BaseTokenizer

class CustomTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("custom_tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        # Your tokenization logic here
        return text.split()
    
    def get_vocabulary_size(self) -> int:
        return 10000  # Your vocab size

# Add to analyzer
analyzer.add_tokenizer(CustomTokenizer())
```

## 📊 Example Use Cases

### **1. Language Efficiency Analysis**
Compare how different tokenizers handle various languages:
- Which tokenizer is most efficient for Chinese?
- How do character-based vs word-based tokenizers differ?
- Which tokenizer compresses text best for each language?

### **2. Content-Specific Analysis**
Analyze tokenization patterns for different content types:
- Code vs natural language
- Technical vs casual text
- Long-form vs short-form content

### **3. Model Selection**
Help choose the right tokenizer for your use case:
- Speed vs compression trade-offs
- Language-specific performance
- Vocabulary utilization patterns

### **4. Research and Development**
- Compare new tokenizers against established ones
- Analyze tokenization patterns across languages
- Benchmark performance on real-world content

## 🛠️ Technical Details

### **Architecture:**
- **Modular design** with clear separation of concerns
- **Extensible framework** for adding new tokenizers
- **Caching system** for performance optimization
- **Session state management** for persistent UI state

### **Performance:**
- **Parallel processing** for multiple tokenizer comparisons
- **Efficient text extraction** from various sources
- **Optimized visualizations** for large datasets
- **Memory-efficient** handling of large texts

### **Dependencies:**
- **Streamlit**: Web interface
- **Transformers**: Hugging Face tokenizers
- **Tiktoken**: OpenAI tokenizers
- **Pandas/NumPy**: Data analysis
- **Matplotlib/Seaborn**: Visualizations
- **Requests**: HTTP requests
- **PyPDF2**: PDF text extraction

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Adding New Tokenizers:**
- Extend the `BaseTokenizer` class
- Implement `tokenize()` and `get_vocabulary_size()` methods
- Add to `get_default_tokenizers()` function

### **Adding New Text Sources:**
- Extend the `TextSampleCollector` class
- Implement new collection methods
- Update the Streamlit interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library
- **OpenAI** for tiktoken
- **Streamlit** for the web framework
- **Wikipedia** for multi-language content

## 📞 Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join the community discussions
- **Documentation**: Check the code comments for detailed explanations

---

**Happy tokenizer analyzing! 🎉**

## Containerization and Kubernetes

This repo includes a Dockerfile and Kubernetes manifests under `deployment/` to run the Streamlit app in a container and deploy it to a cluster.

- Build: `docker build -t transformer-playground:latest .`
- Run: `docker run --rm -p 8501:8501 transformer-playground:latest`
- Open: http://localhost:8501

Publish to a registry (example uses GHCR):

- Tag: `docker tag transformer-playground:latest ghcr.io/your-org/transformer-playground:latest`
- Push: `docker push ghcr.io/your-org/transformer-playground:latest`

Deploy to Kubernetes (kubectl + kustomize):

1. Update the image in `deployment/deployment.yaml` (`image: ghcr.io/your-org/transformer-playground:latest`).
2. Apply all manifests: `kubectl apply -k deployment/`
3. Port-forward (if not using Ingress): `kubectl -n transformer-playground port-forward svc/transformer-playground 8501:80`
4. Open: http://localhost:8501

Ingress (optional):

- Edit `deployment/ingress.yaml` host and TLS secret to match your domain.
- Ensure an Ingress controller (e.g., NGINX) is installed in your cluster.

Notes:

- The app may download tokenizer/model files on first use when selecting Hugging Face tokenizers; ensure outbound internet access in your cluster if you need this functionality.
- Resource requests/limits in `deployment/deployment.yaml` are conservative defaults; tune as needed.
