"""
COMPARISON & ANALYSIS: 3 PENDEKATAN TOKENISASI
"""

import pickle
import json

print("="*70)
print("COMPARISON & ANALYSIS: 3 PENDEKATAN TOKENISASI")
print("="*70)

# Load results
with open('approach1_char.pkl', 'rb') as f:
    results1 = pickle.load(f)

with open('approach2_word.pkl', 'rb') as f:
    results2 = pickle.load(f)

with open('approach3_bpe.pkl', 'rb') as f:
    results3 = pickle.load(f)

# Summary table
print("\n HASIL TRAINING - 3 PENDEKATAN:\n")
print(f"{'Pendekatan':<25} | {'Vocab':<8} | {'Init Loss':<12} | {'Final Loss':<12} | {'Reduction':<10}")
print("─" * 80)

approaches = [results1, results2, results3]
for r in approaches:
    init_loss = r['losses'][0]
    final_loss = r['losses'][-1]
    reduction = (1 - final_loss/init_loss) * 100
    print(f"{r['name']:<25} | {r['vocab_size']:<8} | {init_loss:<12.4f} | {final_loss:<12.4f} | {reduction:<10.1f}%")

# Save results to JSON
comparison_data = {
    "title": "Comparison 3 Pendekatan Tokenisasi TinyGPT",
    "corpus": {
        "domain": "Sejarah Indonesia",
        "words": 2016,
        "characters": 15340
    },
    "approaches": [
        {
            "name": results1['name'],
            "vocab_size": results1['vocab_size'],
            "initial_loss": float(results1['losses'][0]),
            "final_loss": float(results1['losses'][-1]),
            "loss_reduction": float((1 - results1['losses'][-1]/results1['losses'][0])*100),
            "generated_text": results1['generated_text'][:100]
        },
        {
            "name": results2['name'],
            "vocab_size": results2['vocab_size'],
            "initial_loss": float(results2['losses'][0]),
            "final_loss": float(results2['losses'][-1]),
            "loss_reduction": float((1 - results2['losses'][-1]/results2['losses'][0])*100),
            "oov_rate": results2['oov_rate'],
            "generated_text": results2['generated_text'][:100]
        },
        {
            "name": results3['name'],
            "vocab_size": results3['vocab_size'],
            "initial_loss": float(results3['losses'][0]),
            "final_loss": float(results3['losses'][-1]),
            "loss_reduction": float((1 - results3['losses'][-1]/results3['losses'][0])*100),
            "generated_text": results3['generated_text'][:100]
        }
    ]
}

with open('comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(comparison_data, f, indent=2, ensure_ascii=False)

print("\n Results saved to comparison_results.json")

# Analysis
print("\n" + "="*70)
print("ANALISIS HASIL")
print("="*70)

print("\n PENJELASAN 3 PENDEKATAN:\n")

print("1️  CHARACTER-LEVEL:")
print(f"   - Vocab size: {results1['vocab_size']} (sangat kecil)")
print(f"   - Final loss: {results1['losses'][-1]:.4f} (terendah)")
print(f"   - Kelebihan: Tidak ada OOV, vocab minimal")
print(f"   - Kekurangan: Sequence panjang, generated text fragmentary")

print("\n2️  WORD-LEVEL:  TERBAIK UNTUK TEXT QUALITY")
print(f"   - Vocab size: {results2['vocab_size']} (besar)")
print(f"   - Final loss: {results2['losses'][-1]:.4f}")
print(f"   - OOV rate: {results2['oov_rate']:.1f}%")
print(f"   - Kelebihan: Generated text PALING COHERENT")
print(f"   - Kekurangan: Vocab besar, OOV problem")

print("\n3️  SENTENCEPIECE BPE:  BALANCE TERBAIK")
print(f"   - Vocab size: {results3['vocab_size']} (balance)")
print(f"   - Final loss: {results3['losses'][-1]:.4f}")
print(f"   - Kelebihan: Balance vocab/sequence, standard LLM, no OOV")
print(f"   - Kekurangan: Kompleks, perlu tuning")

print("\n" + "="*70)
print("KESIMPULAN & REKOMENDASI")
print("="*70)

print("\nUntuk Corpus Sejarah Indonesia:")
print("✓ Word-Level: Terbaik untuk TEXT QUALITY (coherent text)")
print("✓ SentencePiece: Terbaik untuk BALANCE (standard di industry)")
print("✓ Character: Terbaik untuk LOSS VALUE (tapi text kurang bermakna)")

print("\n REKOMENDASI AKHIR:")
print("Gunakan WORD-LEVEL untuk mendapat generated text yang meaningful")
print("Gunakan SENTENCEPIECE untuk production (standard GPT-2/3)")

print("\n" + "="*70)
print(" ANALYSIS COMPLETE!")
print("="*70)
