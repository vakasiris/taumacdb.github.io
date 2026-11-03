import numpy as np
import pandas as pd





def run_gower_matrix(df):
    print(f"\nStrains:  {len(df)}")
    print(f"Features: {len(df.columns)}")

    n = len(df)
    cols = df.columns.tolist()

    #separating the columns into types
    cat_cols = [c for c in cols if df[c].dtype == 'object']
    num_cols = [c for c in cols if df[c].dtype != 'object']
    bin_cols = [c for c in num_cols
                if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    cont_cols = [c for c in num_cols if c not in bin_cols]

    # Asymmetric: habitats and isolation sources
    # Symmetric: everything else (genes, antimicrobial etc)
    asym_bin_cols = [c for c in bin_cols
                     if c.startswith('hab_') or c.startswith('src_') or c.startswith('gene_')]
    sym_bin_cols = [c for c in bin_cols
                    if not c.startswith('hab_') and not c.startswith('src_') and not c.startswith('gene_')]

    print(f"  Continuous:        {len(cont_cols)} columns")
    print(f"  Binary symmetric:  {len(sym_bin_cols)} columns")
    print(f"  Binary asymmetric: {len(asym_bin_cols)} columns")
    print(f"  Categorical:       {len(cat_cols)} columns")

    ranges = {}
    for c in cont_cols:
        r = df[c].max() - df[c].min()
        ranges[c] = r if r > 0 else 1.0

        # convert to numpy arrays
    cont_arr = df[cont_cols].values.astype(float) if cont_cols else None
    sym_bin_arr = df[sym_bin_cols].values.astype(float) if sym_bin_cols else None
    asym_bin_arr = df[asym_bin_cols].values.astype(float) if asym_bin_cols else None
    cat_arr = df[cat_cols].values if cat_cols else None

    cont_ranges = np.array([ranges[c] for c in cont_cols]) if cont_cols else None

    #distance matrix {n}x{n}
    dist = np.zeros((n, n))
    #implementing Gower distance calculation, based on the original paper
    print("\nCalculating Gower matrix...")
    for i in range(n):
        if i % 50 == 0:
            print(f"  Row {i}/{n}...")

        for j in range(i + 1, n):

            total_d = 0.0
            total_w = 0

            # continuous
            if cont_arr is not None:
                a = cont_arr[i]
                b = cont_arr[j]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.any():
                    total_d += (np.abs(a[mask] - b[mask]) / cont_ranges[mask]).sum()
                    total_w += mask.sum()

            # binary symmetric
            if sym_bin_arr is not None:
                a = sym_bin_arr[i]
                b = sym_bin_arr[j]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.any():
                    total_d += float((a[mask] != b[mask]).sum())
                    total_w += mask.sum()

            #binary assymetric

            if asym_bin_arr is not None:
                a = asym_bin_arr[i]
                b = asym_bin_arr[j]
                # mask: both present (not NaN)
                # and not both zero (co-absence excluded)
                nan_mask = ~(np.isnan(a) | np.isnan(b))
                # exclude pairs where BOTH are 0
                both_zero = (a == 0) & (b == 0)
                mask = nan_mask & ~both_zero

                if mask.any():
                    total_d += float((a[mask] != b[mask]).sum())
                    total_w += mask.sum()

            # categorical
            if cat_arr is not None:
                a = cat_arr[i]
                b = cat_arr[j]
                for k in range(len(cat_cols)):
                    av = a[k]
                    bv = b[k]
                    a_missing = (av is None or
                                 av == 'nan' or
                                 (isinstance(av, float) and np.isnan(av)))
                    b_missing = (bv is None or
                                 bv == 'nan' or
                                 (isinstance(bv, float) and np.isnan(bv)))
                    if not a_missing and not b_missing:
                        total_d += 0.0 if av == bv else 1.0
                        total_w += 1

            d = total_d / total_w if total_w > 0 else 1.0
            dist[i, j] = d
            dist[j, i] = d

    return dist










