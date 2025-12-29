# SIMPLE CHUNK EXPERIMENT WITHOUT NUMPY
# Save as: chunk_experiment_no_numpy.py

print("=" * 60)
print("CHUNK SIZE EXPERIMENT (No NumPy Needed)")
print("=" * 60)

# Test data - no imports needed!
documents = [
    """Machine learning is AI that lets computers learn from data. 
    It has three types: supervised, unsupervised, and reinforcement.""",
    
    """Neural networks are like artificial brains. They have layers 
    of nodes that process information in complex ways.""",
    
    """Deep learning uses many neural network layers. It's good for 
    images, speech, and language tasks like translation."""
]

queries = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain deep learning"
]

def chunk_text(text, chunk_size):
    """Split text into chunks without NumPy"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def calculate_similarity(query, chunk):
    """Simple similarity without embeddings"""
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    
    # Count common words
    common = len(query_words.intersection(chunk_words))
    total = len(query_words.union(chunk_words))
    
    return common / total if total > 0 else 0

# Test different chunk sizes
chunk_sizes = [50, 100, 150, 200, 300]
results = {}

print("\n📊 TESTING DIFFERENT CHUNK SIZES")
print("-" * 40)

for size in chunk_sizes:
    print(f"\nChunk size: {size} characters")
    
    # Create all chunks
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, size))
    
    print(f"  Total chunks created: {len(all_chunks)}")
    
    # Test each query
    total_score = 0
    for query in queries:
        # Find best matching chunks
        scores = []
        for chunk in all_chunks:
            score = calculate_similarity(query, chunk)
            scores.append((score, chunk))
        
        # Get top 3 chunks
        scores.sort(reverse=True)
        top_chunks = scores[:3]
        
        # Calculate average score for this query
        avg_score = sum(s[0] for s in top_chunks) / len(top_chunks)
        total_score += avg_score
    
    # Average across all queries
    avg_total = total_score / len(queries)
    avg_chunk_words = sum(len(c.split()) for c in all_chunks) / len(all_chunks)
    
    results[size] = {
        'chunks': len(all_chunks),
        'score': avg_total,
        'avg_words': avg_chunk_words
    }
    
    print(f"  Average relevance score: {avg_total:.2%}")
    print(f"  Average words per chunk: {avg_chunk_words:.1f}")

# Show results
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print("\nSize | Chunks | Score | Words/Chunk")
print("-" * 40)

best_size = 0
best_score = 0

for size in chunk_sizes:
    r = results[size]
    print(f"{size:4} | {r['chunks']:6} | {r['score']:.2%} | {r['avg_words']:11.1f}")
    
    if r['score'] > best_score:
        best_score = r['score']
        best_size = size

print("\n" + "=" * 60)
print(f"🎯 BEST CHUNK SIZE: {best_size} characters")
print(f"   Score: {best_score:.2%}")
print("=" * 60)

# Recommendations
print("\n💡 RECOMMENDATIONS:")
print("• 50-100 chars: Good for exact facts")
print("• 100-200 chars: Best for most tasks")
print("• 200-300 chars: Good for complex topics")
print("\n✅ Start with 150 characters for general use!")