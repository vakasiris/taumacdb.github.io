import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from pyclustertend.visual_assessment_of_tendency import compute_ivat_ordered_dissimilarity_matrix



def run_ap(df,distance_matrix,similarity_matrix, damping=0.9, max_iter=400):

    #affinity propagation on a precomputed similarity matrix

    print(f"\nRunning Affinity Propagation...")
    print(f"  Damping:  {damping}")
    print(f"  Max iter: {max_iter}")

    ap = AffinityPropagation(
        affinity='precomputed',
        damping=damping,
        max_iter=max_iter,
        random_state=42,
        verbose=True
    )

    ap.fit(similarity_matrix)

    n_clusters = len(np.unique(ap.labels_))
    converged = ap.n_iter_ < max_iter

    if not converged:
        print(f"  WARNING: Did not converge!")
        print(f"  Try increasing damping (currently {damping})")
        print(f"  or increasing max_iter (currently {max_iter})")



    print("Silhouette Coefficient: %0.3f"
          %
          metrics.silhouette_score(
    distance_matrix,
    ap.labels_,
    metric='precomputed'
)
    )




    exemplars = ap.cluster_centers_indices_
    labels = ap.labels_

    print("\nRunning UMAP...")

    reducer = umap.UMAP(
        metric='precomputed',  # use Gower distance directly
        n_components=2,
        random_state=42,
        n_neighbors=40,
        min_dist=0.3
    )

    embedding = reducer.fit_transform(distance_matrix)


    print(f"  UMAP complete. Shape: {embedding.shape}")


    #Running some score tests here
    db_score = davies_bouldin_score(embedding, labels)
    print(f"  Davies Bouldin Score: {db_score}")

    ch_score = calinski_harabasz_score(embedding, labels)
    print(f"  Calinski Harabasz Score: {ch_score}")

    d_prim=compute_ivat_ordered_dissimilarity_matrix(distance_matrix)
    plt.imshow(d_prim, cmap='gray')
    #plt.show()












    return {
        'labels': ap.labels_,
        'n_clusters': n_clusters,
        'converged': converged,
        'n_iterations': ap.n_iter_,
        'exemplars': ap.cluster_centers_indices_,
        'embedding': embedding
    }

