import numpy as np
import pandas as pd



def get_cluster_defining_features(cluster_data, col_types, top_n=10):
    feature_scores = []

    for col in cluster_data.columns:

        morphometric_cols = [
            'cell_length_min', 'cell_length_max', 'cell_length_avg',
            'cell_width_min', 'cell_width_max', 'cell_width_avg'
        ]

        if col in morphometric_cols:
            continue

        col_data = cluster_data[col].dropna()
        n_available = len(col_data)
        n_total = len(cluster_data)
        coverage = (n_available / n_total) * 100

        # Skip if no data at all
        if n_available == 0:
            continue


        if n_available < 2 or coverage < 30:
            continue

        is_hab_src = col.startswith('hab_') or col.startswith('src_') or col.startswith('gene_')
        ctype = col_types.get(col, 'continuous')

        if ctype == 'categorical':
            most_common = col_data.mode()[0]

            n_matching = int((cluster_data[col] == most_common).sum())
            n_not_matching = n_total - n_matching# zeros and nan
            consistency = (n_matching / n_total) * 100
            # Count against WHOLE cluster, not just available
            #n_matching = (cluster_data[col] == most_common).sum()

            #consistency = (n_matching / n_total) * 100

            if consistency >= 50:
                feature_scores.append({
                    'feature': col,
                    'type': 'categorical',
                    'value': most_common,
                    'consistency': round(consistency, 1),
                    #'n_available': n_available,
                    'n_missing': int(n_not_matching),
                    'n_total': n_total
                })

        elif ctype == 'binary':
            if is_hab_src:

                n_present = int((cluster_data[col] == 1).sum())
                n_missing = n_total - n_present  # zeros + NaN
                consistency = (n_present / n_total) * 100
                #n_present = (cluster_data[col] == 1).sum()
                #consistency = (n_present / n_total) * 100

                if consistency >= 30:
                    feature_scores.append({
                        'feature': col,
                        'type': 'binary_asymmetric',
                        'value': 1,
                        'consistency': round(consistency, 1),
                        'n_missing':  int(n_missing),
                        'n_total': n_total
                    })
            else:
                dominant = col_data.mode()[0]

                # Count against WHOLE cluster
                n_matching = (cluster_data[col] == dominant).sum()
                n_missing = n_total - n_matching
                consistency = (n_matching / n_total) * 100

                if consistency >= 50:
                    feature_scores.append({
                        'feature': col,
                        'type': 'binary_symmetric',
                        'value': int(dominant),
                        'consistency': round(consistency, 1),
                        'n_missing':  int(n_missing),
                        'n_total': n_total
                    })

        elif ctype == 'continuous':



            mean_val = col_data.mean()
            std_val = col_data.std()
            cv = std_val / mean_val if mean_val != 0 else float('inf')
            consistency = max(0, (1 - cv) * 100)
            n_missing = n_total - n_available

            feature_scores.append({
                'feature': col,
                'type': 'continuous',
                'mean': round(float(mean_val), 2),
                'std': round(float(std_val), 2),
                'consistency': round(consistency, 1),
                'n_missing':  int(n_missing),
                'n_total': n_total
            })

    # sort by consistency, return top N
    feature_scores.sort(key=lambda x: x['consistency'], reverse=True)
    return feature_scores[:top_n]


def get_cluster_morphometrics(cluster_data):
    morphometrics = {}

    # Group related columns together
    dimensions = {
        'cell_length': {
            'min_col': 'cell_length_min',
            'max_col': 'cell_length_max',
            'avg_col': 'cell_length_avg',
            'label': 'Cell Length'
        },
        'cell_width': {
            'min_col': 'cell_width_min',
            'max_col': 'cell_width_max',
            'avg_col': 'cell_width_avg',
            'label': 'Cell Width'
        }
    }

    for dim_name, cols in dimensions.items():

        min_col = cols['min_col']
        max_col = cols['max_col']
        avg_col = cols['avg_col']

        # Check which columns exist in data
        has_min = min_col in cluster_data.columns
        has_max = max_col in cluster_data.columns
        has_avg = avg_col in cluster_data.columns

        if not any([has_min, has_max, has_avg]):
            continue

        # Overall range:
        # smallest minimum across all strains
        # largest maximum across all strains
        overall_min = None
        overall_max = None
        avg_of_avgs = None
        n_available = 0
        n_total = len(cluster_data)

        if has_min:
            min_data = cluster_data[min_col].dropna()
            if len(min_data) > 0:
                overall_min = round(float(min_data.min()), 2)
                n_available = max(n_available, len(min_data))

        if has_max:
            max_data = cluster_data[max_col].dropna()
            if len(max_data) > 0:
                overall_max = round(float(max_data.max()), 2)
                n_available = max(n_available, len(max_data))

        if has_avg:
            avg_data = cluster_data[avg_col].dropna()
            if len(avg_data) > 0:
                avg_of_avgs = round(float(avg_data.mean()), 2)
                n_available = max(n_available, len(avg_data))

        if n_available == 0:
            continue

        morphometrics[dim_name] = {
            'label': cols['label'],
            'overall_min': overall_min,  # min(cell_length_min)
            'overall_max': overall_max,  # max(cell_length_max)

            'n_available': n_available,
            'n_total': n_total
        }

    return morphometrics

def get_missing_data_summary(cluster_data):


    summary = {}

    for col in cluster_data.columns:
        n_total     = len(cluster_data)
        n_missing   = int(cluster_data[col].isna().sum())
        n_available = n_total - n_missing

        if n_missing > 0:
            summary[col] = {
                'missing_count':   n_missing,
                'available_count': n_available,
                'missing_pct':     round((n_missing / n_total) * 100, 1)
            }

    return summary


def build_strain_features(strain_row, col_types):


    features = {}

    for col, val in strain_row.items():

        is_hab_src = col.startswith('hab_') or col.startswith('src_')
        is_nan     = pd.isna(val)

        if is_hab_src:
            if is_nan:
                features[col] = {
                    'value':   None,
                    'display': 'Not recorded',
                    'type':    'hab_src'
                }
            elif val == 1:
                features[col] = {
                    'value':   1,
                    'display': 'Present',
                    'type':    'hab_src'
                }
            # val == 0: skip entirely

        else:
            ctype = col_types.get(col, 'continuous')

            if is_nan:
                features[col] = {
                    'value':   None,
                    'display': 'Not tested / Not recorded',
                    'type':    ctype
                }
            elif ctype == 'binary_symmetric' or \
                    (ctype == 'binary' and not is_hab_src):
                features[col] = {
                    'value':   int(val),
                    'display': 'Present' if val == 1 else 'Absent',
                    'type':    'binary'
                }
            elif ctype == 'categorical':
                features[col] = {
                    'value':   str(val),
                    'display': str(val),
                    'type':    'categorical'
                }
            elif ctype == 'continuous':
                features[col] = {
                    'value':   float(val),
                    'display': str(round(float(val), 2)),
                    'type':    'continuous'
                }

    return features

def count_available_features(strain_row,hab_cols, src_cols, other_cols):

    count = 0

    # hab: count as 1 if strain has ANY hab data
    if hab_cols:
        hab_vals = strain_row[hab_cols]
        if hab_vals.notna().any():
            count += 1

    # src: count as 1 if strain has ANY src data
    if src_cols:
        src_vals = strain_row[src_cols]
        if src_vals.notna().any():
            count += 1

    # all other columns: count individually
    for col in other_cols:
        if pd.notna(strain_row[col]):
            count += 1

    return count

def build_full_results(df, strain_info, labels,
                       embedding, exemplars, col_types):

    hab_cols = [c for c in df.columns if c.startswith('hab_')]
    src_cols = [c for c in df.columns if c.startswith('src_')]
    other_cols = [c for c in df.columns
                  if not c.startswith('hab_')
                  and not c.startswith('src_')]
    n_features_total = len(other_cols) + (1 if hab_cols else 0) + (1 if src_cols else 0)

    n_strains = len(df)
    n_clusters = len(np.unique(labels))

    exemplar_lookup = {}

    # exemplar_strain_ids: set of strain_ids that are exemplars
    # used to flag individual strains in plot_data
    exemplar_strain_ids = set()

    for idx in exemplars:

        strain_id = int(df.index[idx])
        cluster_id = int(labels[idx])

        exemplar_lookup[cluster_id] = {
            'strain_id': strain_id,
            'strain_code': str(strain_info.loc[strain_id, 'strain_code']),
            'strain_name': str(strain_info.loc[strain_id, 'strain_name'])
        }

        exemplar_strain_ids.add(strain_id)

    # Step 1: Build plot_data (one entry per strain)
    print("\nBuilding plot data...")
    plot_data = []

    for idx in range(n_strains):
        strain_id = int(df.index[idx])
        cluster_id = int(labels[idx])
        n_available = count_available_features(
            df.iloc[idx],
            hab_cols, src_cols, other_cols
        )

        plot_data.append({
            'strain_id': strain_id,
            'strain_code': str(strain_info.loc[strain_id, 'strain_code']),
            'strain_name': str(strain_info.loc[strain_id, 'strain_name']),
            'cluster_id': cluster_id,
            'x': float(embedding[idx, 0]),
            'y': float(embedding[idx, 1]),
            'n_features_available': n_available,
            'n_features_total':     n_features_total,
            'is_exemplar': strain_id in exemplar_strain_ids
        })

    # Step 2: Build cluster details
    print("Building cluster details...")
    clusters = {}

    for cluster_id in np.unique(labels):
        cluster_id = int(cluster_id)
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_data = df.iloc[cluster_indices]

        defining = get_cluster_defining_features(
            cluster_data, col_types, top_n=10)

        missing_summary = get_missing_data_summary(cluster_data)

        morphometrics = get_cluster_morphometrics(cluster_data)

        # Build strain list for this cluster
        cluster_strains = []
        for idx in cluster_indices:
            sid = int(df.index[idx])
            cluster_strains.append({
                'strain_id': sid,
                'strain_code': str(strain_info.loc[sid, 'strain_code']),
                'strain_name': str(strain_info.loc[sid, 'strain_name']),
                'n_features': count_available_features(df.iloc[df.index.get_loc(sid)],hab_cols, src_cols, other_cols),
                'is_exemplar': sid in exemplar_strain_ids
            })

        # Sort: exemplar first, then rest
        cluster_strains.sort(
            key=lambda s: (0 if s['is_exemplar'] else 1)
        )

        clusters[cluster_id] = {
            'size': int(np.sum(cluster_mask)),
            'exemplar': exemplar_lookup.get(cluster_id),
            'strains': cluster_strains,
            'defining_features': defining,
            'morphometrics': morphometrics,
            'missing_summary': missing_summary
        }

    # Step 3: Build strain details
    print("Building strain details...")
    strain_details = {}

    for idx in range(n_strains):
        strain_id = int(df.index[idx])
        strain_row = df.iloc[idx]

        features = build_strain_features(strain_row, col_types)

        strain_details[strain_id] = {
            'strain_code': str(strain_info.loc[strain_id, 'strain_code']),
            'strain_name': str(strain_info.loc[strain_id, 'strain_name']),
            'cluster_id': int(labels[idx]),
            'is_exemplar': strain_id in exemplar_strain_ids,
            'features': features
        }

    print("\nResults built successfully.")

    return {
        'n_clusters': n_clusters,
        'n_strains': n_strains,
        'plot_data': plot_data,
        'clusters': clusters,
        'strain_details': strain_details
    }
