# summary_utils.py (can go in src/)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_top_terms(texts, n_terms=10):
    """Return top TF-IDF terms from a list of texts."""
    if len(texts) == 0:
        return []
    
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1,2))
    X = tfidf.fit_transform(texts)
    scores = X.mean(axis=0).A1
    terms = tfidf.get_feature_names_out()
    top = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:n_terms]
    return [term for term, score in top]

def generate_cluster_summary(df, cluster_labels, n_terms=10, n_examples=3):
    """
    Generate a summary table for clusters.
    
    df: pandas DataFrame with at least ['cluster', 'doc', 'verified_purchase']
    cluster_labels: dict {cluster_id: human_label}
    n_terms: number of top TF-IDF terms to extract
    n_examples: number of review examples per cluster
    """
    summary_data = []

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        if cluster_df.empty:
            continue

        n_reviews = len(cluster_df)
        verified_ratio = cluster_df["verified_purchase"].mean() if "verified_purchase" in cluster_df.columns else None
        label = cluster_labels.get(cluster_id, "Unknown")
        terms = get_top_terms(cluster_df["doc"], n_terms)
        examples = cluster_df["doc"].head(n_examples).tolist()

        summary_data.append({
            "Cluster ID": cluster_id,
            "Label": label,
            "Top Terms": ", ".join(terms),
            "Num Reviews": n_reviews,
            "Verified Ratio": round(verified_ratio, 2) if verified_ratio is not None else "NA",
            "Examples": "\n---\n".join(examples)
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

def save_summary(df_summary, filepath="cluster_summary.csv"):
    """Save the cluster summary DataFrame to CSV."""
    df_summary.to_csv(filepath, index=False)
    print(f"Summary saved to {filepath}")
