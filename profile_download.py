import numpy as np
import pandas as pd
import io


def format_genes_as_columns(genes_list):
    if not genes_list:
        return {}

    result = {}

    for gene_row in genes_list:

        # gene name becomes the column header
        gene_name = gene_row.get('gene_name', 'unknown')

        # all other fields become comma separated content
        pairs = []
        for key, val in gene_row.items():


            if key in ('gene_name', 'strain_id'):
                continue


            if val is None:
                formatted_val = 'NULL'
            elif isinstance(val, float) and np.isnan(val):
                formatted_val = 'NULL'
            elif isinstance(val, bool):
                formatted_val = '1' if val else '0'
            else:
                formatted_val = str(val)

            pairs.append(f"{key}:{formatted_val}")


        result[gene_name] = ','.join(pairs)

    return result


def format_multivalue_dict(data_list):
    # function to convert a list of dicts (table data) into
    # a semicolon-separated string with column prefixes
    if not data_list:
        return None

    entries = []
    for row in data_list:

        # key:value pairs for each row, comma separated
        pairs = []
        for key, val in row.items():

            # Format the value
            if val is None:
                formatted_val = 'NULL'
            elif isinstance(val, float) and np.isnan(val):
                formatted_val = 'NULL'
            elif isinstance(val, bool):
                formatted_val = '1' if val else '0'
            else:
                formatted_val = str(val)

            pairs.append(f"{key}:{formatted_val}")

        entries.append(','.join(pairs)
                       )

    # Entries separated by semicolons
    return '; '.join(entries)

def format_single_dict(data_dict):
    #function to convert a single dict into flat key:value pairs.
    #Already comma-separated values stay as-is.
    formatted = {}
    for key, val in data_dict.items():
        if val is None:
            formatted[key] = 'NULL'
        elif isinstance(val, float) and np.isnan(val):
            formatted[key] = 'NULL'
        elif isinstance(val, bool):
            formatted[key] = '1' if val else '0'
        else:
            formatted[key] = val

    return formatted

def build_strain_csv_row(single_dicts, multi_dicts,genes):
    #function that builds  a csv row per strain

    row = {}
    for prefix, data_dict in single_dicts:
        if not data_dict:
            continue

        formatted = format_single_dict(data_dict)

        for key, val in formatted.items():
            # Add prefix to column name if provided
            col_name = f"{prefix}{key}" if prefix else key
            row[col_name] = val

        if genes:
            gene_cols = format_genes_as_columns(genes)
            row.update(gene_cols)

        # Process multi-value dicts
        for col_name, data_list in multi_dicts:
            if not data_list:
                row[col_name] = 'NULL'
            else:
                row[col_name] = format_multivalue_dict(data_list)

        return row

def build_single_strain_csv(single_dicts, multi_dicts,genes):
    #function that builds a data csv for a single strain
    row = build_strain_csv_row(single_dicts, multi_dicts,genes)

    df = pd.DataFrame([row])

    return df.to_csv(index=False)

def build_all_strains_csv(all_strain_data):
    #function that builds a data csv for all the strain in the database
    rows = []
    for single_dicts, multi_dicts, genes in all_strain_data:
        row = build_strain_csv_row(single_dicts, multi_dicts,genes)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill any missing values with NULL
    df = df.fillna('NULL')

    return df.to_csv(index=False)