import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load hard eval results
with open('../experiments/results/hybrid_bm25_hard_full_with_quality.json', 'r') as f:
    data = json.load(f)

# Get all queries
queries = []
results_map = {}
for r in data['results']:
    if r.get('k') == 5:
        query = r['query']
        queries.append(query)
        results_map[query] = r

print(f"Found {len(queries)} queries")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(queries)

# Determine optimal number of clusters using elbow method
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertias = []
silhouettes = []
K_range = range(2, min(10, len(queries)))

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    inertias.append(kmeans.inertia_)
    if k > 2:
        silhouettes.append(silhouette_score(embeddings, kmeans.labels_))

# Use 5 clusters as default
n_clusters = 5
print(f"Using {n_clusters} clusters")

# Cluster
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# Analyze each cluster
cluster_queries = defaultdict(list)
cluster_relevance = defaultdict(list)
cluster_faithful = defaultdict(list)

for query, cluster in zip(queries, clusters):
    cluster_queries[cluster].append(query)
    result = results_map[query]
    cluster_relevance[cluster].append(result.get('relevant', False))
    cluster_faithful[cluster].append(result.get('faithful', False))

print("\n" + "=" * 60)
print("SEMANTIC CLUSTERING RESULTS")
print("=" * 60)

for cluster in sorted(cluster_queries.keys()):
    queries_in_cluster = cluster_queries[cluster]
    relevance_rate = sum(cluster_relevance[cluster]) / len(cluster_relevance[cluster]) * 100
    faithful_rate = sum(cluster_faithful[cluster]) / len(cluster_faithful[cluster]) * 100
    
    print(f"\n📁 CLUSTER {cluster + 1}: {len(queries_in_cluster)} queries")
    print(f"   Relevance rate: {relevance_rate:.1f}%")
    print(f"   Faithful rate: {faithful_rate:.1f}%")
    print(f"\n   Sample queries:")
    for q in queries_in_cluster[:5]:
        status = "✅" if results_map[q].get('relevant') else "❌"
        print(f"   {status} {q[:70]}...")

print("\n" + "=" * 60)
print("CLUSTER CENTROIDS (thematic center of each cluster)")
print("=" * 60)

# Find the closest query to each centroid
from sklearn.metrics.pairwise import cosine_similarity

for cluster in range(n_clusters):
    centroid = kmeans.cluster_centers_[cluster]
    cluster_indices = np.where(clusters == cluster)[0]
    
    similarities = cosine_similarity([centroid], embeddings[cluster_indices])[0]
    closest_idx = cluster_indices[np.argmax(similarities)]
    representative_query = queries[closest_idx]
    
    print(f"\n📁 CLUSTER {cluster + 1}:")
    print(f"   Representative: \"{representative_query}\"")
    
    # Show relevance pattern
    relevant_count = sum(cluster_relevance[cluster])
    total = len(cluster_relevance[cluster])
    print(f"   Relevance: {relevant_count}/{total} = {relevant_count/total*100:.1f}%")

# Save cluster assignments
output = []
for query, cluster in zip(queries, clusters):
    output.append({
        'query': query,
        'cluster': int(cluster),
        'relevant': results_map[query].get('relevant', False),
        'faithful': results_map[query].get('faithful', False),
        'success': results_map[query].get('success', False)
    })

with open('../experiments/query_clusters.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ Saved cluster assignments to experiments/query_clusters.json")
