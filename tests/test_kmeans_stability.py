import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TestKMeansStability(unittest.TestCase):
    def test_label_stability_across_seeds(self):
        # Generate synthetic data
        np.random.seed(42)
        n = 1000
        
        # High value: high freq, high aov, low recency
        hv = pd.DataFrame({
            'frequency': np.random.normal(50, 10, 250),
            'aov': np.random.normal(200, 30, 250),
            'recency': np.random.normal(10, 5, 250)
        })
        
        # Loyal: med freq, med aov, low recency
        loyal = pd.DataFrame({
            'frequency': np.random.normal(20, 5, 250),
            'aov': np.random.normal(100, 20, 250),
            'recency': np.random.normal(15, 5, 250)
        })
        
        # Occasional: low freq, low aov, med recency
        occ = pd.DataFrame({
            'frequency': np.random.normal(5, 2, 250),
            'aov': np.random.normal(50, 15, 250),
            'recency': np.random.normal(30, 10, 250)
        })
        
        # At Risk: low freq, low aov, high recency
        risk = pd.DataFrame({
            'frequency': np.random.normal(2, 1, 250),
            'aov': np.random.normal(40, 10, 250),
            'recency': np.random.normal(100, 20, 250)
        })
        
        df = pd.concat([hv, loyal, occ, risk], ignore_index=True)
        features = df[['frequency', 'aov', 'recency']]
        
        def run_kmeans_and_label(seed):
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=4, random_state=seed, n_init=10)
            clusters = kmeans.fit_predict(scaled_features)
            
            cluster_means = df.assign(cluster=clusters).groupby('cluster')[['frequency', 'aov', 'recency']].mean()
            
            from sklearn.preprocessing import MinMaxScaler
            cluster_scaler = MinMaxScaler()
            centers = cluster_means.copy()
            centers['inv_recency'] = -centers['recency']
            scaled_centers = cluster_scaler.fit_transform(centers[['frequency', 'aov', 'inv_recency']])
            
            cluster_scores = []
            for i in range(len(scaled_centers)):
                score = (scaled_centers[i][0] * 0.4) + (scaled_centers[i][1] * 0.4) + (scaled_centers[i][2] * 0.2)
                cluster_scores.append((i, score, centers.loc[i, 'recency']))
                
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            c0, c1 = cluster_scores[0][0], cluster_scores[1][0]
            
            remaining = [cluster_scores[2], cluster_scores[3]]
            remaining.sort(key=lambda x: x[2], reverse=True)
            at_risk = remaining[0][0]
            occasional = remaining[1][0]
            
            mapping = {c0: "High Value", c1: "Loyal", occasional: "Occasional", at_risk: "At Risk"}
            return pd.Series(clusters).map(mapping)

        # Run with two completely different random states
        labels_seed1 = run_kmeans_and_label(42)
        labels_seed2 = run_kmeans_and_label(999)
        
        # Assert they produce EXACTLY the same business labels (or >99% identical)
        match_pct = (labels_seed1 == labels_seed2).mean()
        self.assertGreater(match_pct, 0.99, "Business labels are not stable across random seeds!")

if __name__ == '__main__':
    unittest.main()
