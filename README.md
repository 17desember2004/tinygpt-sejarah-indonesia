# TinyGPT dengan 3 Pendekatan Tokenisasi
## Corpus: Sejarah Indonesia

### Overview
Implementasi TinyGPT (Tiny Generative Pre-trained Transformer) dengan 3 pendekatan tokenisasi berbeda pada corpus Sejarah Indonesia.

### Corpus
- **Domain:** Sejarah Indonesia
- **Ukuran:** 2016 kata, 15.340 karakter
- **File:** [corpus_sejarah_indonesia.txt](corpus/corpus_sejarah_indonesia.txt)

### 3 Pendekatan Tokenisasi
1. **Character-Level** - Setiap karakter = 1 token
2. **Word-Level** - Setiap kata = 1 token
3. **SentencePiece BPE** - Subword tokenization

### Quick Start

```bash
pip install -r requirements.txt

python models/approach1_char.py
python models/approach2_word.py
python models/approach3_bpe.py
python models/COMPLETE_ANALYSIS.py
```

### Hasil Summary

| Pendekatan | Vocab | Final Loss | Reduction |
|-----------|-------|-----------|-----------|
| Character-Level | 68 | 1.8819 | 57.0% |
| Word-Level | 502 | 1.4876 | 80.3% ⭐ |
| SentencePiece BPE | 100 | 2.3078 | 56.0% |

### Kesimpulan
- **Best Loss:** Word-Level (1.4876)
- **Best Text Quality:** Word-Level (coherent & meaningful)
- **Best for Production:** SentencePiece BPE (balanced, no OOV)

### Analisis Detail
Lihat [analysis_report.md](analysis_report.md) untuk analisis lengkap Requirement NO.3 & NO.4

### File Structure
