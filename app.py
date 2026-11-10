from flask import Flask,jsonify,render_template,request,redirect,url_for,Response
import mysql.connector
from mysql.connector import pooling
import pandas as pd
import numpy as np
import io


from clustering_results import build_full_results
from gower_matrix import run_gower_matrix
from ap_clustering import run_ap

from profile_download import build_single_strain_csv, build_all_strains_csv


from itertools import groupby
import re
import os

from dotenv import load_dotenv
from pyparsing import restOfLine

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))






_clustering_cache = {
    'df':               None,
    'strain_info':      None,
    'similarity_matrix': None,
    'labels':           None,
    'embedding':        None
}


def has_letters(value):
    return bool(re.search(r"[A-Za-z]", value))

app = Flask(__name__)

#init_db()

db_pool = pooling.MySQLConnectionPool(
    pool_name="taumac_pool",
    pool_size=2,
    pool_reset_session=True,
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    port=int(os.getenv("DB_PORT", "3306")),
    ssl_ca=os.path.join(os.path.dirname(__file__), "ca.pem")
)




@app.template_filter('has_letters')
def has_letters_filter(value):
    return bool(re.search('[a-zA-Z]', str(value)))

@app.route('/clustering',endpoint='clustering')
def clustering_page():
    return render_template('clustering.html')

@app.route('/api/get-clustering-data')
def get_clustering_data():
    con = db_pool.get_connection()
    base_query=("""
    SELECT strain.id AS strain_id, strain.strain_code, CONCAT(view_strain_lineage.genus, ' ', view_strain_lineage.species) AS strain_name,view_strain_lineage.family,strain.temp_tolerance,
    strain.lifestyle,collection.water_depth_max,collection.site_temp_celsius_max,morphology.colony_gen_structure,morphology.colony_color,morphology.cell_gen_shape,
     morphometry_general.cell_length_min,morphometry_general.cell_length_max,morphometry_general.cell_length_avg,morphometry_general.cell_width_min,morphometry_general.cell_width_max,morphometry_general.cell_width_avg
     
     FROM strain
     LEFT JOIN view_strain_lineage ON strain.id = view_strain_lineage.strain_id
     
     
     LEFT JOIN collection ON collection.strain_id = strain.id
     LEFT JOIN morphology ON strain.id = morphology.strain_id
     LEFT JOIN morphometry_general ON strain.id = morphometry_general.strain_id
     
    
    
    """)
    cursor = con.cursor(dictionary=True, buffered=True)
    cursor.execute(base_query)
    base_results = cursor.fetchall()




    habitat_query =("""
    SELECT strain_habitat.strain_id,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Anchialine Cave') THEN 1 ELSE 0 END) AS hab_anchialine_cave,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Brackish') THEN 1 ELSE 0 END) AS hab_brackish,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Epilithic','Epilithic (Cave Hole-Sunlight)','Epilithic (Rock)','Epilithic (Rock-Artificial Light)','Epilithic (Rock-Sunlight)','Epilithic(Rock-Artificial Light)','Epilithic(Rock-Sunlight)') THEN 1 ELSE 0 END) AS hab_epilithic,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Freshwater') THEN 1 ELSE 0 END) AS hab_freshwater,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Intertidal Zone') THEN 1 ELSE 0 END) AS hab_intertidal_zone,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Rice Fields') THEN 1 ELSE 0 END) AS hab_rice_fields,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Sponge Associated') THEN 1 ELSE 0 END) AS hab_sponge_associated,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Terrestrial','Terrestrial Cave') THEN 1 ELSE 0 END) AS hab_terrestrial,
    
    MAX(CASE WHEN habitat.habitat_name IN ('Thermal Spring') THEN 1 ELSE 0 END) AS hab_thermal_spring
    
    FROM strain_habitat
    JOIN habitat ON strain_habitat.habitat_id = habitat.id
    
    GROUP BY strain_habitat.strain_id
    """)
    cursor.execute(habitat_query)
    hab_results = cursor.fetchall()




    gene_query=("""
    SELECT 
            strain_id,
            gene_name,
            gene_pres
        FROM gene_sequence
        WHERE (gene_name !='16S' AND gene_name!='16S-23S rRNA intergenic spacer' AND gene_name !='18S' AND gene_name !='cpcBA' AND gene_name !='ITS1-ITS2' AND gene_name !='rcbL' AND gene_name !='tRNA-Ile and tRNA-Ala' AND gene_name !='rpoC1')
    """)

    cursor.execute(gene_query)
    gene_results = cursor.fetchall()

    col_source_qu=("""
    SELECT collection.strain_id,
    MAX(CASE
    WHEN col_source.col_source_type IN ('Marine rocky reefs','Marine rocky overhangs') THEN 1 ELSE 0 END) AS src_marine_rock,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Freshwater','Freshwater communities') THEN 1 ELSE 0 END) AS src_freshwater,
    
    MAX(CASE
    WHEN col_source.col_source_type IN ('Rice fields') THEN 1 ELSE 0 END) AS src_ricefields,
    
    MAX(CASE
    WHEN col_source.col_source_type IN ('Near reeds') THEN 1 ELSE 0 END) AS src_reeds,
    
    MAX(CASE
    WHEN col_source.col_source_type IN ('Lake surface water','Lake shore','Lake water column','Reservoir water column'  ) THEN 1 ELSE 0 END) AS src_lake,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Stagnant water','Water pond') THEN 1 ELSE 0 END) AS src_stagnant_water,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Spirulina plant located in nigrita (n. greece)') THEN 1 ELSE 0 END) AS src_spirulina_plant ,
    
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Sea cave limestone walls/ceiling','Entrance of an anchialine cave','Anchialine cave entrance','Surface of rocks') THEN 1 ELSE 0 END) AS src_sea_cave ,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Geothermal pond','Hot baths','Hot springs') THEN 1 ELSE 0 END) AS src_thermal_spring ,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Algal mat','Compact mat','Epilithic mat','Epilithic mats on the surface of rocks','Thermal spring algal mat') THEN 1 ELSE 0 END) AS src_algal_mat ,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Base of rock','Cave wall with direct sunlight','Near artificial light','Show cave','Suburban forest','Surface of calcareous substrate in a terrestrial cave','Terrestrial cave walls/ceiling','Terrestrial show cave') THEN 1 ELSE 0 END) AS src_terrestrial_rock,
    
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Close to 
seawater in intertidal zone','Intertidal zone','Marine rocky  sub-littoral zone','Rock surface near intertidal zone','Rocks 
near the shores/ rocks inside cavities of coastal lines','Rocks/rock cavities near shore','Sub-littoral zone','Sublittoral zone','Subsurface of rocks') THEN 1 ELSE 0 END) AS src_littoral_zone ,

     MAX(CASE 
    WHEN col_source.col_source_type IN ('Benthos of lagoon','Lagoon water column') THEN 1 ELSE 0 END) AS src_lagoon,
    
    MAX(CASE 
    WHEN col_source.col_source_type IN ('Saltworks water body','Salt marsh water body') THEN 1 ELSE 0 END) AS src_salt_marsh
    
    FROM collection
    JOIN col_source ON collection.id = col_source.col_id
    
    GROUP BY collection.strain_id
    """)

    cursor.execute(col_source_qu)
    source_results = cursor.fetchall()

    inhibition_query = ("""
            SELECT
                antimicrobial_act_against.strain_id,
                microbe_name.microbe_name,
                CASE
                WHEN MAX(inhibition.inhibition_zone_numeric) > 0 THEN 1
                WHEN MAX(inhibition.inhibition_zone_numeric) = 0 THEN 0
                ELSE NULL
                END as has_effect
                
            FROM antimicrobial_act_against
           
            JOIN microbe_name ON antimicrobial_act_against.microbe_id=microbe_name.id
            JOIN inhibition ON inhibition.antimic_id = antimicrobial_act_against.id
            
            GROUP BY antimicrobial_act_against.strain_id, microbe_name.microbe_name
        """)

    cursor.execute(inhibition_query)
    antimic_results = cursor.fetchall()



    toxins_query=("""
    SELECT toxin_production.strain_id,
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('[D-Asp3]MC-LR','[D-Asp3]MC-RR','MC-HilR', 'MC-HtyR','MC-LA','MC-LF','MC-LR','MC-LW','MC-LY','MC-RR','MC-WR','MC-YR') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('[D-Asp3]MC-LR','[D-Asp3]MC-RR','MC-HilR', 'MC-HtyR','MC-LA','MC-LF','MC-LR','MC-LW','MC-LY','MC-RR','MC-WR','MC-YR') THEN 0
    ELSE NULL
    END )AS has_mcs,
    
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('ATX-a') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('ATX-a') THEN 0
    ELSE NULL
    END) AS has_anatox,
    
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('BMAA') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('BMAA') THEN 0
    ELSE NULL
    END )AS has_bmaa,
    
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('CYN') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('CYN') THEN 0
    ELSE NULL
    END )AS has_cyn,
    
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('STX') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('STX') THEN 0
    ELSE NULL
    END )AS has_stx,
    
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('neoSTX') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('neoSTX') THEN 0
    ELSE NULL
    END )AS has_nstx,
    
    MAX(CASE 
    WHEN toxin_production.presence=1 AND toxins.toxin_name IN ('NOD') THEN 1
    WHEN toxin_production.presence=0 AND toxins.toxin_name IN ('NOD') THEN 0
    ELSE NULL
    END )AS has_nod
    
    FROM toxin_production
    JOIN toxins ON toxins.id = toxin_production.toxin_id
    
    GROUP BY  toxin_production.strain_id
    
    """)

    cursor.execute(toxins_query)
    toxins_results = cursor.fetchall()


    bioactive_query=("""
    SELECT bioactive_comp_production.strain_id,
    MAX(CASE 
    WHEN bioactive_comp_production.comp_presence=1 AND compound_types.compound_type='HCN' THEN 1
    WHEN bioactive_comp_production.comp_presence=0 AND compound_types.compound_type='HCN' THEN 0
    ELSE NULL
    END ) AS has_hcn,
    
    MAX(CASE 
    WHEN bioactive_comp_production.comp_presence=1 AND compound_types.compound_type='Anabaenopeptins' THEN 1
    WHEN bioactive_comp_production.comp_presence=0 AND compound_types.compound_type='Anabaenopeptins' THEN 0
    ELSE NULL
    END ) AS has_anabaenopeptins,
    
    MAX(CASE 
    WHEN bioactive_comp_production.comp_presence=1 AND compound_types.compound_type='Microginins' THEN 1
    WHEN bioactive_comp_production.comp_presence=0 AND compound_types.compound_type='Microginins' THEN 0
    ELSE NULL
    END ) AS has_microgins
    
    FROM bioactive_comp_production
    JOIN compound_types ON bioactive_comp_production.compound_type_id = compound_types.id
    
    GROUP BY bioactive_comp_production.strain_id
    """)

    cursor.execute(bioactive_query)
    bioac_results = cursor.fetchall()

    tumor_query = ("""
                SELECT
                    against_tumor_cells.strain_id,
                    tumor_cell_type.tumor_cell_type,
                    CASE
                    WHEN MAX(effect_on_tumor.viable_cells_perc) < 50 THEN 1
                    WHEN MAX(effect_on_tumor.viable_cells_perc) > 50 THEN 0
                    ELSE NULL
                    END as effect_on_tumor

                FROM against_tumor_cells
                JOIN tumor_cell_type ON tumor_cell_type.id = against_tumor_cells.tumor_cell_id
                JOIN effect_on_tumor ON effect_on_tumor.against_t_cell_id = against_tumor_cells.id
                
                GROUP BY against_tumor_cells.strain_id,tumor_cell_type.tumor_cell_type
                """)

    cursor.execute(tumor_query)
    tumor_results = cursor.fetchall()

    df,strain_info = build_clustering_dataframe(
        base_results,
        gene_results,
        source_results,
        hab_results,
        antimic_results,
        toxins_results,
        bioac_results,
        tumor_results
    )






    distance_matrix = run_gower_matrix(df)

    similarity_matrix = 1 - distance_matrix
    #print(similarity_matrix.shape)

    print(f"\nSimilarity stats:")
    print(f"  Min:  {similarity_matrix.min():.3f}")
    print(f"  Max:  {similarity_matrix.max():.3f}")
    print(f"  Mean: {similarity_matrix.mean():.3f}")

    # Affinity Propagation
    results = run_ap(df,distance_matrix,similarity_matrix)

    



  
    ap_labels=results['labels']
    embedding=results['embedding']
    n = len(embedding)
   

    

    
   


    col_types = {}

#identify column types

    for col in df.columns:
        dtype = df[col].dtype

        if dtype == 'object':
            col_types[col] = 'categorical'

        elif dtype in ['float64', 'float32', 'int64', 'int32']:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                col_types[col] = 'binary'
            else:
                col_types[col] = 'continuous'

        else:
            col_types[col] = 'continuous'


    _clustering_cache['df'] = df
    _clustering_cache['strain_info'] = strain_info
    _clustering_cache['similarity_matrix'] = similarity_matrix
    _clustering_cache['labels'] = results['labels']
    _clustering_cache['embedding'] = results['embedding']
    _clustering_cache['exemplars'] = results['exemplars']


    full_results = build_full_results(
        df,
        strain_info,
        results['labels'],
        results['embedding'], # pass embedding directly
        results['exemplars'],
        col_types
    )



    full_results['converged'] = results['converged']
    full_results['n_iterations'] = results['n_iterations']

    return jsonify(full_results)



@app.route('/api/download-feature-matrix')
def download_feature_matrix():

    if _clustering_cache['df'] is None:
        return jsonify({'error': 'No clustering data available.Run clustering first.'}), 400

    df= _clustering_cache['df']
    strain_info = _clustering_cache['strain_info']

    df_download = df.copy()
    df_download.index = [
        str(strain_info.loc[sid, 'strain_code'])
        for sid in df_download.index
    ]
    df_download.index.name = 'strain_code'

    csv_str = df_download.to_csv()

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                'attachment; filename=clustering_features.csv'
        }
    )


@app.route('/api/download-similarity-matrix')
def download_similarity_matrix():
    if _clustering_cache['similarity_matrix'] is None:
        return jsonify({'error': 'No clustering data available.Run clustering first.'}), 400

    similarity_matrix = _clustering_cache['similarity_matrix']
    strain_info = _clustering_cache['strain_info']
    df = _clustering_cache['df']

    strain_codes = [
        str(strain_info.loc[sid, 'strain_code'])
        for sid in df.index
    ]

    df_matrix = pd.DataFrame(
        similarity_matrix,
        index=strain_codes,
        columns=strain_codes
    )

    df_matrix.index.name = 'strain_code'

    csv_str = df_matrix.to_csv()

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                'attachment; filename=similarity_matrix.csv'
        }
    )



@app.route('/api/download-cluster-assignments')
def download_cluster_assignments():
    if _clustering_cache['labels'] is None:
        return jsonify({'error': 'No clustering data available.Run clustering first.'}), 400

    df = _clustering_cache['df']
    strain_info = _clustering_cache['strain_info']
    labels = _clustering_cache['labels']
    embedding = _clustering_cache['embedding']
    exemplars = _clustering_cache['exemplars']

    exemplar_strain_ids = set(
        int(df.index[idx]) for idx in exemplars
    )

    rows = []
    for i, strain_id in enumerate(df.index):
        strain_id = int(strain_id)
        rows.append({
            'strain_code': str(strain_info.loc[strain_id, 'strain_code']),
            'strain_name': str(strain_info.loc[strain_id, 'strain_name']),
            'cluster_id': int(labels[i]),
            'is_exemplar': strain_id in exemplar_strain_ids,
            'umap_x': round(float(embedding[i, 0]), 4),
            'umap_y': round(float(embedding[i, 1]), 4),
            'n_features_available': int(df.iloc[i].notna().sum()),
            'n_features_total': len(df.columns)
        })

    df_assignments = pd.DataFrame(rows)
    csv_str = df_assignments.to_csv(index=False)

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                'attachment; filename=cluster_assignments.csv'
        }
    )



def build_clustering_dataframe(base_results, gene_results,
                               source_results,hab_results, antimic_results,toxins_results, bioac_results,tumor_results):
    # base dataframe (all strains, all continuous + categorical)
    df_base = pd.DataFrame(base_results, columns=base_results[0].keys())
    df_base = df_base.set_index('strain_id')


    df_base['strain_name'] =df_base['strain_name'].str.replace('unclassified', 'sp.', regex=False)




    # separate strain info from features
    strain_info = df_base[['strain_code', 'strain_name']].copy()





    df_features = df_base.drop(columns=['strain_code', 'strain_name'])

    # pivoting

    if gene_results:
        df_genes = pd.DataFrame(
            gene_results,
            columns=['strain_id', 'gene_name', 'gene_pres']
        )

        df_genes_pivot = df_genes.pivot_table(
            index='strain_id',  # rows = strains
            columns='gene_name',  # columns = one per gene
            values='gene_pres',  # values = 1/0
            aggfunc='max'
        )

        df_genes_pivot.columns = [
            f"gene_{col.lower().replace(' ', '_').replace('/', '_')}"
            for col in df_genes_pivot.columns
        ]

        df_genes_pivot.columns.name = None

    if source_results:

        df_source = pd.DataFrame(source_results)
        df_source = df_source.set_index("strain_id")


    if hab_results:

        df_hab = pd.DataFrame(hab_results)
        df_hab = df_hab.set_index("strain_id")

    if antimic_results:
        df_inhibition = pd.DataFrame(
            antimic_results,
            columns=['strain_id', 'microbe_name', 'has_effect']
        )

        df_inhibition_pivot = df_inhibition.pivot_table(
            index='strain_id',  # rows = strains
            columns='microbe_name',  # columns = one per microbe
            values='has_effect',  # values = 1/0/NaN
            aggfunc='max'
        )

        # Rename columns: Bacillus subtilis → anti_bacillus_subtilis
        df_inhibition_pivot.columns = [
            f"anti_{col.lower().replace(' ', '_').replace('/', '_')}"
            for col in df_inhibition_pivot.columns
        ]

        df_inhibition_pivot.columns.name = None


    if toxins_results:
        df_toxins = pd.DataFrame(toxins_results)
        df_toxins = df_toxins.set_index("strain_id")

    if bioac_results:
        df_bioac = pd.DataFrame(bioac_results)
        df_bioac = df_bioac.set_index("strain_id")
        df_bioac.columns = [
            f"bio_{col.lower().replace(' ', '_').replace('/', '_')}"
            for col in df_bioac.columns
        ]

    if tumor_results:
        df_tumor = pd.DataFrame(
            tumor_results,
            columns=['strain_id', 'tumor_cell_type', 'effect_on_tumor']
        )

        df_tumor_pivot = df_tumor.pivot_table(
            index='strain_id',  # rows = strains
            columns='tumor_cell_type',
            values='effect_on_tumor',  # values = 1/0/NaN
            aggfunc='max'
        )


        df_tumor_pivot.columns = [
            f"against_{col.lower().replace(' ', '_').replace('/', '_')}"
            for col in df_tumor_pivot.columns
        ]

        df_tumor_pivot.columns.name = None


    # LEFT JOIN all pivots onto base


    df_final = df_features.join(
        [df_genes_pivot, df_source,df_hab, df_inhibition_pivot,df_toxins,df_bioac,df_tumor_pivot],
        how='left'
    )
    src_cols = [c for c in df_final.columns if c.startswith('src_')]
    hab_cols = [c for c in df_final.columns if c.startswith('hab_')]

    for col_group in [src_cols, hab_cols]:
        for idx in df_final.index:
            row_values = df_final.loc[idx, col_group]
            has_any_data = row_values.notna().any()

            if has_any_data:
                # Strain has some source/habitat data

                df_final.loc[idx, col_group] = row_values.fillna(0)

    df_final = df_final.where(df_final.notna(), other=np.nan)
    df_final.columns = [
        col.replace('\n', '_')
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('-', '_')
        .strip()
        for col in df_final.columns
    ]


    continuous_cols = [
         'water_depth_max',
        'site_temp_celsius_max',
        'cell_length_min', 'cell_length_max', 'cell_length_avg',
        'cell_width_min', 'cell_width_max', 'cell_width_avg'
    ]

    for col in continuous_cols:
        if col in df_final.columns:
            # coerce turns any stray strings into NaN
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')



    categorical_cols = [
        'family', 'growth_medium', 'lifestyle',
        'colony_gen_structure', 'colony_color',
        'cell_gen_shape'
    ]

    for col in categorical_cols:
        if col in df_final.columns:
            # Replace None and empty strings with NaN
            df_final[col] = df_final[col].replace({None: np.nan, '': np.nan})

    for col in df_final.columns:
        if isinstance(df_final[col].dtype, pd.StringDtype):
            df_final[col] = df_final[col].astype(object)

    return df_final,strain_info



@app.route('/map', endpoint='collection_map')
def strain_map():
    return render_template("map.html")


@app.route('/map/data', endpoint='collection_map_data')
def strain_map_data():
    con = db_pool.get_connection()
    cursor = con.cursor(dictionary=True, buffered=True)
    cursor.execute("""
            SELECT distinct view_strain_species_genus.strain_id,strain_profile.strain_code, view_strain_species_genus.genus_name, view_strain_species_genus.species_name,
                   strain_profile.col_lat, strain_profile.col_long
            
           
            FROM strain_profile
            JOIN view_strain_species_genus ON view_strain_species_genus.strain_id =strain_profile.strain_id 
            WHERE (strain_profile.col_lat is not NULL and strain_profile.col_long is not NULL and strain_profile.strain_code is not NULL)
            

        """)

    results = cursor.fetchall()

    try:
        return jsonify(results)
    finally:
        con.close()



@app.route('/stats',endpoint='stats')
def stats_data():
    con = db_pool.get_connection()
    cursor = con.cursor(dictionary=True, buffered=True)
    sql_stats = ("""
            SELECT COUNT(DISTINCT strain.id ) AS strain_count,
            (SELECT COUNT(DISTINCT species)
                FROM view_strain_lineage
                WHERE species IS NOT NULL
                AND species != 'unclassified'
            ) AS species_count,
            
            (SELECT COUNT(DISTINCT strain_id)
                FROM gene_sequence
                WHERE gene_name IN ('16S', '18S')
            ) AS ssu_count,
            
            (SELECT COUNT(DISTINCT strain_id)
                FROM genome_seq
            ) AS gen_count,
            
            (SELECT COUNT(DISTINCT strain.id)
                FROM strain
                WHERE type_strain='1'
            ) AS type_count
            
            

            FROM strain 
    		""")


    pie_query=("""
    SELECT habitat.habitat_name AS hab_name,COUNT(strain_habitat.strain_id) AS hab_count
                FROM habitat
                JOIN strain_habitat ON strain_habitat.habitat_id = habitat.id
                GROUP BY habitat_name
            
    
    """)

    lif_query=("""
    SELECT strain.lifestyle, COUNT(strain.id) AS count
    FROM strain 
    GROUP BY strain.lifestyle
    
    """)

    fam_query=("""
    SELECT view_strain_lineage.family ,COUNT(view_strain_lineage.strain_id) AS strain_per_family
    FROM view_strain_lineage
    GROUP BY view_strain_lineage.family
    ORDER BY strain_per_family DESC
    
    """)

    genus_query = ("""
        SELECT view_strain_lineage.genus ,COUNT(view_strain_lineage.strain_id) AS strain_per_genus
        FROM view_strain_lineage
        GROUP BY view_strain_lineage.genus
        ORDER BY strain_per_genus DESC

        """)

    species_query = ("""
            SELECT CONCAT(view_strain_lineage.genus, ' ', view_strain_lineage.species) AS full_species_name,COUNT(view_strain_lineage.strain_id) AS strain_per_species
            FROM view_strain_lineage
            
            
            GROUP BY view_strain_lineage.genus,view_strain_lineage.species
            ORDER BY strain_per_species DESC

            """)


    heat_map_query=("""
    SELECT DISTINCT strain.id,strain.strain_code,CONCAT(view_strain_lineage.genus, ' ', view_strain_lineage.species) AS strain_name,view_strain_lineage.kingdom,
    CASE WHEN view_strain_lineage.species !='unclassified' THEN 1 ELSE 0 END AS species_class,
    CASE WHEN strain.temp_tolerance IS NOT NULL AND strain.temp_tolerance !='' THEN 1 ELSE 0 END AS has_temp,
    CASE WHEN collection.col_lat IS NOT NULL AND collection.col_lat !='' THEN 1 ELSE 0 END AS has_coords,
    CASE WHEN collection.water_depth_max IS NOT NULL AND collection.water_depth_max !='' THEN 1 ELSE 0 END AS has_depth,
    CASE WHEN collection.site_temp_celsius_max IS NOT NULL AND collection.site_temp_celsius_max !='' THEN 1 ELSE 0 END AS has_site_temp,
    CASE WHEN morphology.morph_description IS NOT NULL AND morphology.morph_description !='' THEN 1 ELSE 0 END AS has_morphology,
    CASE WHEN morphometry_general.cell_length_max IS NOT NULL AND morphometry_general.cell_length_max !='' THEN 1 ELSE 0 END AS has_morphometry,
    CASE WHEN growth_rate_set.strain_id IS NOT NULL AND growth_rate_set.strain_id !='' THEN 1 ELSE 0 END AS has_g_rate,
    CASE WHEN antimicrobial_act_against.strain_id IS NOT NULL AND antimicrobial_act_against.strain_id !='' THEN 1 ELSE 0 END AS has_antimic,
    CASE WHEN toxin_production.strain_id IS NOT NULL AND toxin_production.strain_id !='' THEN 1 ELSE 0 END AS has_toxin_test,
    CASE WHEN genome_seq.strain_id IS NOT NULL AND genome_seq.strain_id !='' THEN 1 ELSE 0 END AS has_genome_seq,
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = '16S'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_16s,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = '18S'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_18s,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = '16S-23S rRNA intergenic spacer'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_16s_23s,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'cpcBA'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_cpcBA,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'ITS1-ITS2'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_ITS,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'NifE'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_nife,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'NifN'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_nifn,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'rcbL'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_rcbl,
            
    
    
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'rpoC1'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_rpoc1,
            
    CASE WHEN EXISTS (
                SELECT 1 FROM gene_sequence 
                WHERE gene_sequence.strain_id = strain.id 
                AND gene_sequence.gene_pres = 1
                AND gene_sequence.gene_name = 'tRNA-Ile and tRNA-Ala'
                
                AND gene_sequence.gene_name != ''
            ) THEN 1 ELSE 0 END as has_trna
            
    FROM strain
    LEFT JOIN view_strain_lineage ON strain.id = view_strain_lineage.strain_id
    LEFT JOIN collection ON strain.id = collection.strain_id
    LEFT JOIN morphology ON strain.id = morphology.strain_id
    LEFT JOIN morphometry_general ON strain.id = morphometry_general.strain_id
    LEFT JOIN growth_rate_set ON strain.id = growth_rate_set.strain_id
    LEFT JOIN antimicrobial_act_against ON strain.id = antimicrobial_act_against.strain_id
    LEFT JOIN toxin_production ON strain.id = toxin_production.strain_id
    LEFT JOIN genome_seq ON strain.id = genome_seq.strain_id
    LEFT JOIN gene_sequence ON strain.id = gene_sequence.strain_id
    
    ORDER BY strain_name
    """)







    try:




        cursor.execute(sql_stats)
        rows = cursor.fetchone()

        cursor.execute(pie_query)
        habitat_results = cursor.fetchall()

        habitat_dict = {str(row['hab_name']): int(row['hab_count']) for row in habitat_results}

        cursor.execute(lif_query)
        lif_results = cursor.fetchall()
        lif_dict = {str(row['lifestyle']): int(row['count']) for row in lif_results}

        cursor.execute(fam_query)
        fam_results = cursor.fetchall()
        fam_dict = {str(row['family']): int(row['strain_per_family']) for row in fam_results}

        cursor.execute(genus_query)
        gen_results = cursor.fetchall()
        gen_dict = {str(row['genus']): int(row['strain_per_genus']) for row in gen_results}

        cursor.execute(species_query)
        sp_results = cursor.fetchall()
        sp_dict = {str(row['full_species_name'].replace('unclassified','sp.')): int(row['strain_per_species']) for row in sp_results}

        cursor.execute(heat_map_query)
        availability_results = cursor.fetchall()



        other_features = ['Species-level Classification', 'Temperature Tolerance', 'Collection Coordinates',
                          'Collection Water Depth', 'Col. Site Temperature', 'Morphological Desc', 'Morphometry Data',
                          'Growth Rates', 'Antimicrobial Data', 'Toxin Data', 'Genome Seq']
        gene_features = ['16S rRNA Seq', '18S rRNA Seq', '16S-23S spacer Seq', 'cpcBA Seq', 'ITS1-ITS2 Seq', 'NifE Seq',
                         'NifN Seq', 'rcbL Seq', 'rpoC1 Seq', 'tRNA-Ile and tRNA-Ala Seq']
        all_features = other_features + gene_features

        heatmap_data = []
        for row in availability_results:

            is_prokaryote = row['kingdom'] in ['Bacteria']
            strain_row = {
                'strain_name':row['strain_name'],
                'strain_id': row['id'],
                'strain_code': row['strain_code'],
                'features': {}
            }
            strain_row['strain_name'] = str(row['strain_name'].replace('unclassified','sp.'))

            if is_prokaryote:
                strain_row['features']['16S rRNA Seq'] = row['has_16s']
                strain_row['features']['ITS1-ITS2 Seq'] = -1  # N/A
                strain_row['features']['cpcBA Seq'] = row['has_cpcBA']
                strain_row['features']['18S rRNA Seq'] = -1
                strain_row['features']['16S-23S spacer Seq'] = row['has_16s_23s']
                strain_row['features']['rcbL Seq'] = row['has_rcbl']
                strain_row['features']['rpoC1 Seq'] = row['has_rpoc1']

                strain_row['features']['NifN Seq'] = row['has_nifn']
                strain_row['features']['NifE Seq'] = row['has_nife']
                strain_row['features']['tRNA-Ile and tRNA-Ala Seq'] = row['has_trna']

            else:  # eukaryotes
                strain_row['features']['rpoC1 Seq'] = -1
                strain_row['features']['rcbL Seq'] = row['has_rcbl']
                strain_row['features']['16S rRNA Seq'] = -1  # N/A
                strain_row['features']['18S rRNA Seq'] = row['has_18s']
                strain_row['features']['ITS1-ITS2 Seq'] = row['has_ITS']
                strain_row['features']['16S-23S spacer Seq'] = -1
                strain_row['features']['cpcBA Seq'] = -1

                strain_row['features']['NifN Seq'] = -1
                strain_row['features']['NifE Seq'] = -1
                strain_row['features']['tRNA-Ile and tRNA-Ala Seq'] = -1

            strain_row['features']['Species-level Classification'] = row['species_class']
            strain_row['features']['Temperature Tolerance'] = row['has_temp']
            strain_row['features']['Collection Coordinates'] = row['has_coords']
            strain_row['features']['Collection Water Depth'] = row['has_depth']
            strain_row['features']['Col. Site Temperature'] = row['has_site_temp']
            strain_row['features']['Morphological Desc'] = row['has_morphology']
            strain_row['features']['Morphometry Data'] = row['has_morphometry']
            strain_row['features']['Growth Rates'] = row['has_g_rate']
            strain_row['features']['Antimicrobial Data'] = row['has_antimic']
            strain_row['features']['Toxin Data'] = row['has_toxin_test']
            strain_row['features']['Genome Seq'] = row['has_genome_seq']

            heatmap_data.append(strain_row)




        return render_template("stats.html",rows=rows,habitat_dict=habitat_dict,lif_dict=lif_dict,sp_dict=sp_dict,gen_dict=gen_dict,fam_dict=fam_dict,heatmap_data=heatmap_data,heatmap_features=all_features)
    finally:
        con.close()


@app.route('/taxonomy' ,endpoint='taxonomy')
def taxonomy_page():
    return render_template("taxonomy.html")


@app.route('/taxonomy/data')
def taxonomy_data():
    con = db_pool.get_connection()
    try:
        cursor = con.cursor(dictionary=True, buffered=True)
        cursor.execute("""
            SELECT DISTINCT view_strain_lineage.strain_id, view_strain_lineage.empire,view_strain_lineage.kingdom,view_strain_lineage.subkingdom,view_strain_lineage.infrakingdom,view_strain_lineage.phylum,view_strain_lineage.subphylum,view_strain_lineage.class,view_strain_lineage.subclass,
            view_strain_lineage.order_name,view_strain_lineage.family,view_strain_lineage.subfamily,view_strain_lineage.genus,view_strain_lineage.species,strain.strain_code,strain.novel_taxon
            
            FROM view_strain_lineage
            JOIN strain ON strain.id=view_strain_lineage.strain_id
            
            ORDER BY strain_id,novel_taxon, empire,kingdom,subkingdom,infrakingdom,phylum,subphylum,class,subclass,
            order_name,family,subfamily,genus,species
        
        """)

        rows = cursor.fetchall()

        tree = {}

        ranks = ['empire','kingdom','subkingdom','infrakingdom','phylum','subphylum','class','subclass',
            'order_name','family','subfamily','genus','species']
        for row in rows:
            node = tree

            path_nodes = []

            for rank in ranks:
                value = row.get(rank)

                # skip NULL ranks
                if not value or value == 'NULL':
                    continue

                if value not in node:
                    node[value] = {'_children': {}, '_strains': [], '_rank': rank}

                path_nodes.append(node[value])
                node = node[value]['_children']


            if path_nodes:
                path_nodes[-1]['_strains'].append({
                    'strain_id': row['strain_id'],
                    'strain_code': row['strain_code'],
                    'novel_taxon': row['novel_taxon']
                })

        return jsonify(tree)

    finally:
        con.close()



@app.route('/',endpoint='home')
def home():
    return render_template("home.html")


@app.route('/search', methods=['GET'])
def search_live():
    query = request.args.get("q", "").strip().lower()

    if len(query)<2:
        return jsonify([])


    wildcard = f"%{query}%"
    con = db_pool.get_connection()

    try:
        cursor = con.cursor(dictionary=True, buffered=True)

        sql_part = [

            """
                    SELECT strain.id AS strain_id, strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name ,NULL as lifestyle,NULL as temp_tolerance, NULL AS habitat_name,
                    NULL AS col_source_type,NULL as morphology_preview,
                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,null as helix_pitch_length_min,
                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'general' AS source  
                    FROM strain 
                    JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id

                    WHERE CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) LIKE %s
                    OR strain.strain_code LIKE %s
                    
                    UNION
        
        
                    SELECT DISTINCT view_strain_species_genus.strain_id,view_strain_species_genus.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                    NULL AS lifestyle,NULL as temp_tolerance,NULL as habitat_name ,NULL as col_source_type ,LEFT(morphology.morph_description, 80) AS morphology_preview,
                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,null as helix_pitch_length_min,
                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'morphology' AS source
                    FROM strain 
                    JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id
                    JOIN morphology ON morphology.strain_id = strain.id
                    WHERE morphology.morph_description LIKE %s



            """]
        params = [wildcard, wildcard, wildcard]

        lifestyle="planktic benthic lifestyle"
        if query.lower() in lifestyle :
            sql_part.append("""
                           SELECT strain.id, strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                  strain.lifestyle,NULL AS temp_tolerance, NULL AS habitat_name,NULL AS col_source_type,NULL as morphology_preview,
                                  NULL as cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,null as helix_pitch_length_min,
                                  NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'lifestyle' AS source
                           FROM strain
                           JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id
        
                           WHERE (strain.lifestyle LIKE %s OR 'lifestyle' LIKE %s)
                       """)
            params.append(wildcard)
            params.append(wildcard)

        temp = "thermophilic non-thermophilic temperature tolerance"
        if query.lower() in temp:
            sql_part.append("""
                                   SELECT strain.id, strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                          NULL AS lifestyle,strain.temp_tolerance, NULL AS habitat_name,NULL AS col_source_type,NULL as morphology_preview,
                                          NULL as cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,null as helix_pitch_length_min,
                                          NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'temp' AS source
                                   FROM strain
                                   JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id

                                   WHERE (strain.temp_tolerance LIKE %s OR 'temperature tolerance' LIKE %s)
                                   AND strain.temp_tolerance IS NOT NULL
                                   
                               """)
            params.append(wildcard)
            params.append(wildcard)

        habitat="Terrestrial, Sponge Associated ,Rice Fields, Intertidal Zone, Freshwater,Epilithic(Rock-Sunlight), Epilithic(Rock-Artificial Light), Epilithic(Cave Hole-Sunlight), Epilithic (Rock-Sunlight), Epilithic (Rock-Artificial Light), Epilithic (Rock), Epilithic, Brackish ,Anchialine Cave ,habitat"
        if query.lower() in habitat.lower():
            sql_part.append("""
                                   SELECT DISTINCT strain.id, strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                          NULL AS lifestyle, NULL AS temp_tolerance, habitat.habitat_name,NULL AS col_source_type,NULL as morphology_preview,
                                          NULL as cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,null as helix_pitch_length_min,
                                          NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'habitat' AS source
                                   FROM strain
                                   JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id
                                   JOIN strain_habitat ON strain_habitat.strain_id = strain.id
                                   JOIN habitat ON strain_habitat.habitat_id = habitat.id

                                   WHERE (habitat.habitat_name LIKE %s OR 'habitat' LIKE %s)
                                   

                               """)
            params.append(wildcard)
            params.append(wildcard)

        col_source ="Marine rocky reefs ,Marine rocky overhangs, Lake surface water, Freshwater Sea cave ,limestone walls/ceiling,Spirulina plant located in nigrita (n. greece),,Water pond,Stagnant water,Thermal spring,Hot springs,Surface water,Terrestrial cave walls/ceiling,Algal mat,Lake shore,Epilithic mats on the surface of rocks,Terrestrial show cave,Epilithic mat,Marine rocky  sub-littoral zone,Sub-littoral zone,Geothermal pond,Thermal spring algal mat,Lake water column,Saltworks water body,Salt marsh water body,Show cave,Intertidal zone,Rocks near the shores/ rocks inside cavities of coastal lines,Subsurface of rocks,Close to seawater in intertidal zone,Cave wall with direct sunlight,Surface of calcareous substrate in a terrestrial cave,Near artificial light,Entrance of an anchialine cave,Rock surface near intertidal zone,Rocks/rock cavities near shore,Sublittoral zone,Lagoon water column,Benthos of lagoon,Base of rock,Suburban forest,Rice fields,Freshwater communities,Near reeds,Reservoir water column,Anchialine cave entrance,Surface of rocks,Compact mat,Hot baths, collection"
        if query.lower() in col_source.lower():
            sql_part.append("""
                                   SELECT DISTINCT strain.id, strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                          NULL AS lifestyle, NULL AS temp_tolerance, NULL habitat_name,col_source.col_source_type,NULL as morphology_preview,
                                          NULL as cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,null as helix_pitch_length_min,
                                          NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'col_source' AS source
                                   FROM strain
                                   JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id
                                   JOIN collection ON strain.id=collection.strain_id
                                   JOIN col_source ON col_id = collection.id

                                   WHERE (col_source.col_source_type LIKE %s OR 'collection source' LIKE %s)


                               """)
            params.append(wildcard)
            params.append(wildcard)

        morphometry="morphometry"

        if query.lower() in morphometry.lower():

            sql_part.append("""
            SELECT
             strain.id, strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
             NULL AS lifestyle, NULL AS temp_tolerance, NULL habitat_name,NULL AS col_source_type,NULL as morphology_preview,morphometry_general.cell_length_min,morphometry_heterocyte.heterocyte_length_min,morphometry_apical_cell.apical_cell_length_min,morphometry_akinete.akinete_length_min,morphometry_cyst.cyst_length_min,morphometry_filament.filament_length_min,morphometry_helix.helix_pitch_length_min,
             NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'gen_morphometry' AS source
            FROM strain
            JOIN view_strain_species_genus ON strain.id = view_strain_species_genus.strain_id
            left JOIN morphometry_general ON morphometry_general.strain_id = strain.id
left JOIN morphometry_akinete ON morphometry_akinete.strain_id = strain.id
left JOIN morphometry_apical_cell ON morphometry_apical_cell.strain_id = strain.id
left JOIN morphometry_cyst ON morphometry_cyst.strain_id = strain.id
left JOIN morphometry_filament ON morphometry_filament.strain_id= strain.id
left JOIN morphometry_helix ON morphometry_helix.strain_id= strain.id
LEFT JOIN morphometry_heterocyte ON morphometry_heterocyte.strain_id= strain.id

            WHERE 'morphometry' LIKE %s
            AND (morphometry_general.cell_length_min IS NOT NULL
            OR morphometry_heterocyte.heterocyte_length_min IS NOT NULL
            OR morphometry_apical_cell.apical_cell_length_min IS NOT NULL
            OR morphometry_akinete.akinete_length_min IS NOT NULL
            OR morphometry_cyst.cyst_length_min IS NOT NULL
            OR morphometry_filament.filament_length_min IS NOT NULL
            OR morphometry_helix.helix_pitch_length_min IS NOT NULL)
            """)
            params.append(wildcard)

        mc="microcystins, toxins, [D-Asp3]MC-LR, [D-Asp3]MC-RR, MC-HilR, MC-HtyR,MC-LA', MC-LF, MC-LR, MC-LW, MC-LY, MC-RR, MC-WR, MC-YR"

        if query.lower() in mc.lower():
            sql_part.append("""
                SELECT DISTINCT
                    strain.id,
                    strain.strain_code,
                    CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,

                    NULL AS lifestyle,
                    NULL AS temp_tolerance,
                    NULL AS habitat_name,
                    NULL AS col_source_type,
                    NULL AS morphology_preview,
                    NULL AS cell_length_min,
                    NULL AS heterocyte_length_min,
                    NULL AS apical_cell_length_min,
                    NULL AS akinete_length_min,
                    NULL AS cyst_length_min,
                    NULL AS filament_length_min,
                    NULL AS helix_pitch_length_min,

                    NULL AS presence,
                    NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,

                    'mc' AS source

                FROM strain

                JOIN view_strain_species_genus
                ON strain.id = view_strain_species_genus.strain_id

                WHERE EXISTS (
                    SELECT 1
                    FROM toxin_production
                    JOIN toxins
                        ON toxin_production.toxin_id = toxins.id

                    WHERE toxin_production.strain_id = strain.id
                      AND toxin_production.presence = 1
                      AND toxins.toxin_name IN (
                        '[D-Asp3]MC-LR',
                        '[D-Asp3]MC-RR',
                        'MC-HilR',
                        'MC-HtyR',
                        'MC-LA',
                        'MC-LF',
                        'MC-LR',
                        'MC-LW',
                        'MC-LY',
                        'MC-RR',
                        'MC-WR',
                        'MC-YR'
                      )
                )
                """)


        atx ="anatoxins,atx ,toxins,ATX-a"
        if query.lower() in atx.lower():
                sql_part.append("""
                    SELECT DISTINCT
                        strain.id,
                        strain.strain_code,
                        CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,

                        NULL AS lifestyle,
                        NULL AS temp_tolerance,
                        NULL AS habitat_name,
                        NULL AS col_source_type,
                        NULL AS morphology_preview,
                        NULL AS cell_length_min,
                        NULL AS heterocyte_length_min,
                        NULL AS apical_cell_length_min,
                        NULL AS akinete_length_min,
                        NULL AS cyst_length_min,
                        NULL AS filament_length_min,
                        NULL AS helix_pitch_length_min,

                        NULL AS presence,
                        NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,

                        'atx' AS source

                    FROM strain

                    JOIN view_strain_species_genus
                    ON strain.id = view_strain_species_genus.strain_id

                    WHERE EXISTS (
                        SELECT 1
                        FROM toxin_production
                        JOIN toxins
                            ON toxin_production.toxin_id = toxins.id

                        WHERE toxin_production.strain_id = strain.id
                          AND toxin_production.presence = 1
                          AND toxins.toxin_name IN ('ATX','ATX-a'
                          )
                    )
                    """)

        cyn="cylindrospermopsins,cyn,toxins"
        if query.lower() in cyn.lower():
                sql_part.append("""
                    SELECT DISTINCT
                        strain.id,
                        strain.strain_code,
                        CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,

                        NULL AS lifestyle,
                        NULL AS temp_tolerance,
                        NULL AS habitat_name,
                        NULL AS col_source_type,
                        NULL AS morphology_preview,
                        NULL AS cell_length_min,
                        NULL AS heterocyte_length_min,
                        NULL AS apical_cell_length_min,
                        NULL AS akinete_length_min,
                        NULL AS cyst_length_min,
                        NULL AS filament_length_min,
                        NULL AS helix_pitch_length_min,

                        NULL AS presence,
                        NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,

                        'cyn' AS source

                    FROM strain

                    JOIN view_strain_species_genus
                    ON strain.id = view_strain_species_genus.strain_id

                    WHERE EXISTS (
                        SELECT 1
                        FROM toxin_production
                        JOIN toxins
                            ON toxin_production.toxin_id = toxins.id

                        WHERE toxin_production.strain_id = strain.id
                          AND toxin_production.presence = 1
                          AND toxins.toxin_name IN ('CYN'
                          )
                    )
                    """)


        nod="nodularins,nod,toxins"
        if query.lower() in nod.lower():
                sql_part.append("""
                    SELECT DISTINCT
                        strain.id,
                        strain.strain_code,
                        CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,

                        NULL AS lifestyle,
                        NULL AS temp_tolerance,
                        NULL AS habitat_name,
                        NULL AS col_source_type,
                        NULL AS morphology_preview,
                        NULL AS cell_length_min,
                        NULL AS heterocyte_length_min,
                        NULL AS apical_cell_length_min,
                        NULL AS akinete_length_min,
                        NULL AS cyst_length_min,
                        NULL AS filament_length_min,
                        NULL AS helix_pitch_length_min,

                        NULL AS presence,
                        NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,

                        'nod' AS source

                    FROM strain

                    JOIN view_strain_species_genus
                    ON strain.id = view_strain_species_genus.strain_id

                    WHERE EXISTS (
                        SELECT 1
                        FROM toxin_production
                        JOIN toxins
                            ON toxin_production.toxin_id = toxins.id

                        WHERE toxin_production.strain_id = strain.id
                          AND toxin_production.presence = 1
                          AND toxins.toxin_name IN ('NOD'
                          )
                    )
                    """)

        bma= "bmaa,β-Methylamino-L-alanine,toxins"
        if query.lower() in bma.lower():
                sql_part.append("""
                    SELECT DISTINCT
                        strain.id,
                        strain.strain_code,
                        CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,

                        NULL AS lifestyle,
                        NULL AS temp_tolerance,
                        NULL AS habitat_name,
                        NULL AS col_source_type,
                        NULL AS morphology_preview,
                        NULL AS cell_length_min,
                        NULL AS heterocyte_length_min,
                        NULL AS apical_cell_length_min,
                        NULL AS akinete_length_min,
                        NULL AS cyst_length_min,
                        NULL AS filament_length_min,
                        NULL AS helix_pitch_length_min,

                        NULL AS presence,
                        NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,

                        'bma' AS source

                    FROM strain

                    JOIN view_strain_species_genus
                    ON strain.id = view_strain_species_genus.strain_id

                    WHERE EXISTS (
                        SELECT 1
                        FROM toxin_production
                        JOIN toxins
                            ON toxin_production.toxin_id = toxins.id

                        WHERE toxin_production.strain_id = strain.id
                          AND toxin_production.presence = 1
                          AND toxins.toxin_name IN ('BMAA'
                          )
                    )
                    """)

        neostx="neo-saxitoxin,neo-stx,toxins"
        if query.lower() in neostx.lower():

                sql_part.append("""
                    SELECT DISTINCT
                        strain.id,
                        strain.strain_code,
                        CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,

                        NULL AS lifestyle,
                        NULL AS temp_tolerance,
                        NULL AS habitat_name,
                        NULL AS col_source_type,
                        NULL AS morphology_preview,
                        NULL AS cell_length_min,
                        NULL AS heterocyte_length_min,
                        NULL AS apical_cell_length_min,
                        NULL AS akinete_length_min,
                        NULL AS cyst_length_min,
                        NULL AS filament_length_min,
                        NULL AS helix_pitch_length_min,

                        NULL AS presence,
                        NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,

                        'neostx' AS source

                    FROM strain

                    JOIN view_strain_species_genus
                    ON strain.id = view_strain_species_genus.strain_id

                    WHERE EXISTS (
                        SELECT 1
                        FROM toxin_production
                        JOIN toxins
                            ON toxin_production.toxin_id = toxins.id

                        WHERE toxin_production.strain_id = strain.id
                          AND toxin_production.presence = 1
                          AND toxins.toxin_name IN ('neoSTX'))
                    """)


        stx="saxitoxins,stx,toxins"
        if query.lower() in stx.lower():


                sql_part.append("""
                                SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'stx' AS source
                                FROM strain

                                JOIN view_strain_species_genus
                                ON strain.id = view_strain_species_genus.strain_id

                                WHERE EXISTS (
                                    SELECT 1
                                    FROM toxins
                                    JOIN toxin_production
                                        ON toxin_production.toxin_id = toxins.id

                                    WHERE toxin_production.strain_id = strain.id
                                      AND toxin_production.presence =1
                                      AND toxins.toxin_name LIKE 'STX'
                                      )
                                """)

        hcn="Hydrogen cyanide,hcn,bioactive compounds"
        if query.lower() in hcn.lower():


                sql_part.append("""
                                SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'hcn' AS source
                                FROM strain

                                JOIN view_strain_species_genus
                                ON strain.id = view_strain_species_genus.strain_id

                                WHERE EXISTS (
                                    SELECT 1
                                    FROM bioactive_comp_production
                                    JOIN compound_types
                                        ON bioactive_comp_production.compound_type_id = compound_types.id

                                    WHERE bioactive_comp_production.strain_id = strain.id
                                      AND bioactive_comp_production.comp_presence =1
                                      AND compound_types.compound_type LIKE 'HCN'
                                      )
                                """)

        anab = "anabaenopeptins,bioactive compounds"
        if query.lower() in anab.lower():
            sql_part.append("""
                                        SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                        NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                        NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                        NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                        NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'anab' AS source
                                        FROM strain

                                        JOIN view_strain_species_genus
                                        ON strain.id = view_strain_species_genus.strain_id

                                        WHERE EXISTS (
                                            SELECT 1
                                            FROM bioactive_comp_production
                                            JOIN compound_types
                                                ON bioactive_comp_production.compound_type_id = compound_types.id

                                            WHERE bioactive_comp_production.strain_id = strain.id
                                              AND bioactive_comp_production.comp_presence =1
                                              AND compound_types.compound_type LIKE 'Anabaenopeptins'
                                              )
                                        """)

        microgin = "microginins,bioactive compounds"
        if query.lower() in microgin.lower():
            sql_part.append("""
                                                SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name,'microgin' AS source
                                                FROM strain

                                                JOIN view_strain_species_genus
                                                ON strain.id = view_strain_species_genus.strain_id

                                                WHERE EXISTS (
                                                    SELECT 1
                                                    FROM bioactive_comp_production
                                                    JOIN compound_types
                                                        ON bioactive_comp_production.compound_type_id = compound_types.id

                                                    WHERE bioactive_comp_production.strain_id = strain.id
                                                      AND bioactive_comp_production.comp_presence =1
                                                      AND compound_types.compound_type LIKE 'Microginins'
                                                      )
                                                """)
        antimic = "antimicrobial"
        if query.lower() in antimic.lower():
            sql_part.append("""
                                                        SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                        NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                        NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                        NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                        NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'antimic' AS source
                                                        FROM strain

                                                        JOIN view_strain_species_genus
                                                        ON strain.id = view_strain_species_genus.strain_id

                                                        WHERE EXISTS (
                                                            SELECT 1
                                                            FROM antimicrobial_act_against
                                                            JOIN microbe_name ON antimicrobial_act_against.microbe_id=microbe_name.id
                                                            JOIN inhibition ON inhibition.antimic_id = antimicrobial_act_against.id

                                                            WHERE antimicrobial_act_against.strain_id = strain.id
                                                            AND microbe_name.microbe_name IS NOT NULL)
                                                            """)
        g_rate = "growth rates"
        if query.lower() in g_rate.lower():
            sql_part.append("""
             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'g_rate' AS source
                                                                FROM strain

                                                                JOIN view_strain_species_genus
                                                                ON strain.id = view_strain_species_genus.strain_id

                                                                WHERE EXISTS (
                                                                    SELECT 1
                                                                    FROM growth_rate_set
                                                                    JOIN growth_rate ON growth_rate.growth_set_id= growth_rate_set.id
                                                                    

                                                                    WHERE growth_rate_set.strain_id = strain.id
                                                                    AND growth_rate.rate IS NOT NULL)
                                                                                                                      """)

        s16="SSU rRNA,16s gene,gene"
        if query.lower() in s16.lower():
            sql_part.append("""
             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, '16S' AS source
            FROM strain

            JOIN view_strain_species_genus
            ON strain.id = view_strain_species_genus.strain_id

            WHERE EXISTS (
            SELECT 1
            FROM gene_sequence
            


            WHERE gene_sequence.strain_id = strain.id
            AND gene_sequence.gene_name IS NOT NULL
            AND gene_sequence.gene_name ='16S'
            )
          """)

        s18 = "SSU rRNA,18s gene, gene"
        if query.lower() in s18.lower():
            sql_part.append("""
                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, '18S' AS source
                    FROM strain

                    JOIN view_strain_species_genus
                    ON strain.id = view_strain_species_genus.strain_id

                    WHERE EXISTS (
                    SELECT 1
                    FROM gene_sequence



                    WHERE gene_sequence.strain_id = strain.id
                    AND gene_sequence.gene_name IS NOT NULL
                    AND gene_sequence.gene_name ='18S'
                    )
                """)

        spacer1 = "16S-23S rRNA intergenic spacer, gene"
        if query.lower() in spacer1.lower():
            sql_part.append("""
                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, '16-23' AS source
                            FROM strain

                            JOIN view_strain_species_genus
                            ON strain.id = view_strain_species_genus.strain_id

                            WHERE EXISTS (
                            SELECT 1
                            FROM gene_sequence



                            WHERE gene_sequence.strain_id = strain.id
                            AND gene_sequence.gene_name IS NOT NULL
                            AND gene_sequence.gene_name ='16S-23S rRNA intergenic spacer'
                            )
                        """)

        cpc = "cpcBA gene,phycobilisome subunit, gene"
        if query.lower() in cpc.lower():
            sql_part.append("""
                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'cpc' AS source
                                    FROM strain

                                    JOIN view_strain_species_genus
                                    ON strain.id = view_strain_species_genus.strain_id

                                    WHERE EXISTS (
                                    SELECT 1
                                    FROM gene_sequence



                                    WHERE gene_sequence.strain_id = strain.id
                                    AND gene_sequence.gene_name IS NOT NULL
                                    AND gene_sequence.gene_name ='cpcBA'
                                    )
                                """)

        spacer2 = "ITS1-ITS2 gene,internal transcribed spacer, gene"
        if query.lower() in spacer2.lower():
            sql_part.append("""
                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'it1' AS source
                                            FROM strain

                                            JOIN view_strain_species_genus
                                            ON strain.id = view_strain_species_genus.strain_id

                                            WHERE EXISTS (
                                            SELECT 1
                                            FROM gene_sequence



                                            WHERE gene_sequence.strain_id = strain.id
                                            AND gene_sequence.gene_name IS NOT NULL
                                            AND gene_sequence.gene_name ='ITS1-ITS2'
                                            )
                                        """)

        rcbl = " rcbL gene, ribulose-1,5-bisphosphate carboxylase/oxygenase large subunit, gene"
        if query.lower() in rcbl.lower():
            sql_part.append("""
                                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'rcbl' AS source
                                                    FROM strain

                                                    JOIN view_strain_species_genus
                                                    ON strain.id = view_strain_species_genus.strain_id

                                                    WHERE EXISTS (
                                                    SELECT 1
                                                    FROM gene_sequence



                                                    WHERE gene_sequence.strain_id = strain.id
                                                    AND gene_sequence.gene_name IS NOT NULL
                                                    AND gene_sequence.gene_name ='rcbL'
                                                    )
                                                """)

        nife = "NifE gene, gene"
        if query.lower() in nife.lower():
            sql_part.append("""
                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'nife' AS source
                                                            FROM strain

                                                            JOIN view_strain_species_genus
                                                            ON strain.id = view_strain_species_genus.strain_id

                                                            WHERE EXISTS (
                                                            SELECT 1
                                                            FROM gene_sequence



                                                            WHERE gene_sequence.strain_id = strain.id
                                                            AND gene_sequence.gene_name IS NOT NULL
                                                            AND gene_sequence.gene_name ='NifE'
                                                            )
                                                        """)

        nifn = "NifN gene, gene"
        if query.lower() in nifn.lower():
            sql_part.append("""
                                                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'nifn' AS source
                                                                    FROM strain

                                                                    JOIN view_strain_species_genus
                                                                    ON strain.id = view_strain_species_genus.strain_id

                                                                    WHERE EXISTS (
                                                                    SELECT 1
                                                                    FROM gene_sequence



                                                                    WHERE gene_sequence.strain_id = strain.id
                                                                    AND gene_sequence.gene_name IS NOT NULL
                                                                    AND gene_sequence.gene_name ='NifN'
                                                                    )
                                                                """)

        mcya = "mcyA gene, gene"
        if query.lower() in mcya.lower():
            sql_part.append("""
                                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'mcya' AS source
                                                                            FROM strain

                                                                            JOIN view_strain_species_genus
                                                                            ON strain.id = view_strain_species_genus.strain_id

                                                                            WHERE EXISTS (
                                                                            SELECT 1
                                                                            FROM gene_sequence



                                                                            WHERE gene_sequence.strain_id = strain.id
                                                                            AND gene_sequence.gene_name IS NOT NULL
                                                                            AND gene_sequence.gene_name ='mcyA'
                                                                            )
                                                                        """)

        mcyb = "mcyB gene, gene"
        if query.lower() in mcyb.lower():
            sql_part.append("""
                                                                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'mcyb' AS source
                                                                                    FROM strain

                                                                                    JOIN view_strain_species_genus
                                                                                    ON strain.id = view_strain_species_genus.strain_id

                                                                                    WHERE EXISTS (
                                                                                    SELECT 1
                                                                                    FROM gene_sequence



                                                                                    WHERE gene_sequence.strain_id = strain.id
                                                                                    AND gene_sequence.gene_name IS NOT NULL
                                                                                    AND gene_sequence.gene_name ='mcyB'
                                                                                    )
                                                                                """)

        mcye = "mcyE gene, gene"
        if query.lower() in mcye.lower():
            sql_part.append("""
                                                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'mcye' AS source
                                                                                            FROM strain

                                                                                            JOIN view_strain_species_genus
                                                                                            ON strain.id = view_strain_species_genus.strain_id

                                                                                            WHERE EXISTS (
                                                                                            SELECT 1
                                                                                            FROM gene_sequence



                                                                                            WHERE gene_sequence.strain_id = strain.id
                                                                                            AND gene_sequence.gene_name IS NOT NULL
                                                                                            AND gene_sequence.gene_name ='mcyE'
                                                                                            )
                                                                                        """)

        ndaf = "mcyE/ndaF gene, gene"
        if query.lower() in ndaf.lower():
            sql_part.append("""
                                                                                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'ndaf' AS source
                                                                                                    FROM strain

                                                                                                    JOIN view_strain_species_genus
                                                                                                    ON strain.id = view_strain_species_genus.strain_id

                                                                                                    WHERE EXISTS (
                                                                                                    SELECT 1
                                                                                                    FROM gene_sequence



                                                                                                    WHERE gene_sequence.strain_id = strain.id
                                                                                                    AND gene_sequence.gene_name IS NOT NULL
                                                                                                    AND gene_sequence.gene_name ='mcyE/ndaF'
                                                                                                    )
                                                                                                """)

        pks = " PKS gene, polyketide synthase ketosynthase domain, gene "
        if query.lower() in pks.lower():
            sql_part.append("""
                                                                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'pks' AS source
                                                                                                            FROM strain

                                                                                                            JOIN view_strain_species_genus
                                                                                                            ON strain.id = view_strain_species_genus.strain_id

                                                                                                            WHERE EXISTS (
                                                                                                            SELECT 1
                                                                                                            FROM gene_sequence



                                                                                                            WHERE gene_sequence.strain_id = strain.id
                                                                                                            AND gene_sequence.gene_name IS NOT NULL
                                                                                                            AND gene_sequence.gene_name ='PKS'
                                                                                                            )
                                                                                                        """)

        nrps = "NRPS gene, non-ribosomal peptide synthetase, gene "
        if query.lower() in nrps.lower():
            sql_part.append("""
                                                                                                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'nrps' AS source
                                                                                                                    FROM strain

                                                                                                                    JOIN view_strain_species_genus
                                                                                                                    ON strain.id = view_strain_species_genus.strain_id

                                                                                                                    WHERE EXISTS (
                                                                                                                    SELECT 1
                                                                                                                    FROM gene_sequence



                                                                                                                    WHERE gene_sequence.strain_id = strain.id
                                                                                                                    AND gene_sequence.gene_name IS NOT NULL
                                                                                                                    AND gene_sequence.gene_name ='NRPS'
                                                                                                                    )
                                                                                                                """)

        ps = " PS gene, gene "
        if query.lower() in ps.lower():
            sql_part.append("""
                                                                                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'ps' AS source
                                                                                                                            FROM strain

                                                                                                                            JOIN view_strain_species_genus
                                                                                                                            ON strain.id = view_strain_species_genus.strain_id

                                                                                                                            WHERE EXISTS (
                                                                                                                            SELECT 1
                                                                                                                            FROM gene_sequence



                                                                                                                            WHERE gene_sequence.strain_id = strain.id
                                                                                                                            AND gene_sequence.gene_name IS NOT NULL
                                                                                                                            AND gene_sequence.gene_name ='PS'
                                                                                                                            )
                                                                                                                        """)

        rpoc = " rpoC1 gene, DNA-dependent RNA polymerase, gene "
        if query.lower() in rpoc.lower():
            sql_part.append("""
                                                                                                                                     SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                                                     NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                                                    NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                                                    NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                                                    NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'rpoc' AS source
                                                                                                                                    FROM strain

                                                                                                                                    JOIN view_strain_species_genus
                                                                                                                                    ON strain.id = view_strain_species_genus.strain_id

                                                                                                                                    WHERE EXISTS (
                                                                                                                                    SELECT 1
                                                                                                                                    FROM gene_sequence



                                                                                                                                    WHERE gene_sequence.strain_id = strain.id
                                                                                                                                    AND gene_sequence.gene_name IS NOT NULL
                                                                                                                                    AND gene_sequence.gene_name ='rpoC1'
                                                                                                                                    )
                                                                                                                                """)

        sxti = " sxtI gene, gene "
        if query.lower() in sxti.lower():
            sql_part.append("""
                                                                                                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'sxti' AS source
                                                                                                                                            FROM strain

                                                                                                                                            JOIN view_strain_species_genus
                                                                                                                                            ON strain.id = view_strain_species_genus.strain_id

                                                                                                                                            WHERE EXISTS (
                                                                                                                                            SELECT 1
                                                                                                                                            FROM gene_sequence



                                                                                                                                            WHERE gene_sequence.strain_id = strain.id
                                                                                                                                            AND gene_sequence.gene_name IS NOT NULL
                                                                                                                                            AND gene_sequence.gene_name ='sxtI'
                                                                                                                                            )
                                                                                                                                            """)
        genome=" genome, genome sequence, whole genome "
        if query.lower() in genome.lower():
            sql_part.append("""
                                                                                                                                             SELECT strain.id,strain.strain_code,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS strain_name,
                                                                                                                                             NULL AS lifestyle,NULL AS temp_tolerance,NULL AS habitat_name,NULL AS col_source_type,NULL AS morphology_preview,    
                                                                                                                                            NULL AS cell_length_min,NULL AS heterocyte_length_min,NULL AS apical_cell_length_min,   
                                                                                                                                            NULL AS akinete_length_min,NULL AS cyst_length_min,NULL AS filament_length_min,NULL AS helix_pitch_length_min,   
                                                                                                                                            NULL AS presence,NULL AS toxin_name,NULL AS comp_presence,NULL AS compound_type,NULL AS microbe_name,NULL AS growth_rate,NULL AS gene_name, 'genome' AS source
                                                                                                                                            FROM strain

                                                                                                                                            JOIN view_strain_species_genus
                                                                                                                                            ON strain.id = view_strain_species_genus.strain_id

                                                                                                                                            WHERE EXISTS (
                                                                                                                                            SELECT 1
                                                                                                                                            FROM genome_seq



                                                                                                                                            WHERE genome_seq.strain_id = strain.id
                                                                                                                                            AND genome_seq.accession_number IS NOT NULL
                                                                                                                                            
                                                                                                                                            )
                                                                                                                                        """)

        sql_query = " UNION ".join(sql_part) + " LIMIT 50"
        cursor.execute(sql_query, tuple(params))

        results = cursor.fetchall()

        return jsonify(results)

    finally:
        con.close()



@app.route('/search',methods = ['GET','POST'])
def search():
    results = []
    query = request.form.get("search_query", "").strip()

    if request.method == "POST":

        sql_query = """ 
        SELECT DISTINCT view_strain_species_genus.strain_id,CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) AS species_name,  view_strain_species_genus.strain_code,strain_profile.lifestyle,strain_profile.novel_taxon,strain_profile.temp_tolerance,strain_profile.morph_description,strain_profile.cell_length_min,strain_profile.cell_length_avg,strain_profile.akinete_length_min,strain_profile.heterocyte_length_min,strain_profile.apical_cell_length_min,strain_profile.cyst_length_min,strain_profile.filament_length_min,strain_profile.helix_pitch_length_min,
        EXISTS(SELECT 1 FROM strain_profile WHERE strain_profile.strain_id = view_strain_species_genus.strain_id AND strain_profile.presence =1) AS has_toxins,
        EXISTS(SELECT 1 FROM strain_profile WHERE strain_profile.strain_id = view_strain_species_genus.strain_id AND strain_profile.inhibition_zone_numeric > 0) AS antimic,
        EXISTS(SELECT 1 FROM strain_profile WHERE strain_profile.strain_id = view_strain_species_genus.strain_id AND strain_profile.comp_presence =1) AS has_bioactive,
        EXISTS(SELECT 1 FROM strain_profile WHERE strain_profile.strain_id = view_strain_species_genus.strain_id AND strain_profile.rate IS NOT NULL) AS has_growth_rate,
        EXISTS(SELECT 1 FROM strain_profile WHERE strain_profile.strain_id = view_strain_species_genus.strain_id AND strain_profile.gene_name IS NOT NULL) AS has_gene_seq,
        EXISTS(SELECT 1 FROM strain_profile WHERE strain_profile.strain_id = view_strain_species_genus.strain_id AND strain_profile.viable_cells_perc IS NOT NULL) AS has_tumor_data,
        GROUP_CONCAT(DISTINCT strain_profile.gene_name ORDER BY strain_profile.gene_name SEPARATOR ', ') AS gene_names
        FROM view_strain_species_genus
        LEFT JOIN strain_profile ON strain_profile.strain_id = view_strain_species_genus.strain_id
        WHERE CONCAT(view_strain_species_genus.genus_name, ' ', view_strain_species_genus.species_name) LIKE %s
        
        OR view_strain_species_genus.strain_code LIKE %s
        
         GROUP BY view_strain_species_genus.strain_id,view_strain_species_genus.strain_code,species_name, strain_profile.morph_description, strain_profile.cell_length_min, strain_profile.cell_length_avg, strain_profile.akinete_length_min, strain_profile.heterocyte_length_min, strain_profile.apical_cell_length_min, strain_profile.cyst_length_min, strain_profile.filament_length_min, strain_profile.helix_pitch_length_min, strain_profile.lifestyle, strain_profile.novel_taxon, strain_profile.temp_tolerance
        
       
        
        
        """

        wildcard = f"%{query}%"


        con = db_pool.get_connection()
        cursor = con.cursor(dictionary=True,buffered=True)
        cursor.execute(sql_query,(wildcard, wildcard))
        results = cursor.fetchall()



    try:
            return render_template("search.html", results=results,search_query=query)
    finally:
        con.close()

def merge_strain_rows(rows, multi_fields):

    if not rows:
        return None


    merged = {}

    for row in rows:
        for key, value in row.items():
            if value is None:
                continue


            if key in multi_fields:
                if key not in merged:
                    merged[key] = set()
                merged[key].add(str(value))



            else:
                if key not in merged:
                    merged[key] = value

    # convert sets to comma-separated strings
    for key in multi_fields:
        if key in merged:
            merged[key] = ", ".join(sorted(merged[key]))

    return merged



@app.route('/strain/<int:strain_id>')
def strain_profile(strain_id):
    con = db_pool.get_connection()
    cursor = con.cursor(dictionary=True,buffered=True)

    lineage_query= """
    SELECT * FROM view_strain_lineage
    WHERE strain_id = %s
    """

    cursor.execute(lineage_query,(strain_id,))
    lineage = cursor.fetchone()

    profile_query="""
    SELECT strain_profile.strain_id AS strain_id,strain_profile.strain_code,
                    strain_profile.type_strain,strain_profile.novel_taxon,strain_profile.industrial_usage ,strain_profile.growth_medium ,
                    strain_profile.temp_tolerance,strain_profile.lifestyle,strain_profile.growth_medium,strain_profile.habitat_name,strain_profile.habitat_name,strain_profile.host_name,strain_profile.source_geo_location,strain_profile.col_lat,col_long,
                    strain_profile.water_depth_max,strain_profile.water_depth_min,strain_profile.col_date,strain_profile.site_temp_celsius_max,strain_profile.site_temp_celsius_min,strain_profile.col_source_type,strain_profile.morph_description,strain_profile.colony_gen_structure,
                    strain_profile.colony_color,strain_profile.cell_gen_shape,
                    strain_profile.cell_length_min,strain_profile.cell_length_max,strain_profile.cell_length_avg,strain_profile.cell_length_SD,strain_profile.cell_width_min,strain_profile.cell_width_max,strain_profile.cell_width_avg,strain_profile.cell_width_SD,
                    strain_profile.akinete_length_min,strain_profile.akinete_length_max,strain_profile.akinete_length_avg,strain_profile.akinete_length_SD,strain_profile.akinete_width_min,strain_profile.akinete_width_max,strain_profile.akinete_width_avg,strain_profile.akinete_width_SD,
                    strain_profile.heterocyte_length_min,strain_profile.heterocyte_length_max,strain_profile.heterocyte_length_avg,strain_profile.heterocyte_length_SD,strain_profile.heterocyte_width_min,strain_profile.heterocyte_width_max,strain_profile.heterocyte_width_avg,strain_profile.heterocyte_width_SD,
                    strain_profile.apical_cell_length_min,strain_profile.apical_cell_length_max,strain_profile.apical_cell_length_avg,strain_profile.apical_cell_length_SD,strain_profile.apical_cell_width_min,strain_profile.apical_cell_width_max,strain_profile.apical_cell_width_avg,strain_profile.apical_cell_width_SD, 
                    strain_profile.cyst_length_min,strain_profile.cyst_length_max,strain_profile.cyst_length_avg,strain_profile.cyst_length_SD,strain_profile.cyst_width_min,strain_profile.cyst_width_max,strain_profile.cyst_width_avg,strain_profile.cyst_width_SD,
                    strain_profile.filament_length_min,strain_profile.filament_length_max,strain_profile.filament_length_avg,strain_profile.filament_length_SD,strain_profile.filament_width_min,strain_profile.filament_width_max,strain_profile.filament_width_avg,strain_profile.filament_width_SD,
                    strain_profile.helix_pitch_length_min,strain_profile.helix_pitch_length_max,strain_profile.helix_pitch_length_avg,strain_profile.helix_pitch_length_SD,strain_profile.helix_diameter_min,strain_profile.helix_diameter_max,strain_profile.helix_diameter_avg,strain_profile.helix_diameter_SD,strain_profile.helix_height_min,strain_profile.helix_height_max,strain_profile.helix_height_avg,strain_profile.helix_height_SD
        FROM strain_profile
        WHERE strain_id = %s
    
    
    """

    cursor.execute(profile_query, (strain_id,))

    rows = cursor.fetchall()




    MULTI_VALUE_FIELDS = {
        "col_source_type",
        "habitat_name"
    }

    strain = merge_strain_rows(rows, MULTI_VALUE_FIELDS)

    if not strain:

        return "Strain not found.", 404
        con.close()


    strain["has_letters"] = has_letters(strain["strain_code"])

    #separate fetch for toxins , its a one-to-many

    toxin_query = """
    SELECT DISTINCT toxin_name , presence , concentration, concentration_unit 
    FROM strain_profile
    
    WHERE strain_id = %s
    
    ORDER BY toxin_name
    
    """

    cursor.execute(toxin_query, (strain_id,))
    toxins = cursor.fetchall()



    if toxins[0]["toxin_name"] is None :
       toxins=[]

    #separate fetch for bioactive compounds


    bioact_query="""
    SELECT DISTINCT compound_type, comp_presence , compound_name , comp_concentration , comp_concentration_unit 
    FROM strain_profile
    
    WHERE strain_id = %s 
    
    ORDER BY compound_type
    
    
    """

    cursor.execute(bioact_query, (strain_id,))
    bioactive_comps = cursor.fetchall()

    if bioactive_comps[0]["compound_type"] is None :
       bioactive_comps=[]

    #query for antimicrobial seperately

    antimic_query="""
    SELECT DISTINCT microbe_name,inhibition_zone_char,transparency,solvent
    FROM strain_profile
    WHERE strain_id = %s
    
    ORDER BY microbe_name
    """

    cursor.execute(antimic_query, (strain_id,))
    antimic = cursor.fetchall()

    if antimic[0]["microbe_name"] is None :
       antimic=[]

    from collections import Counter

    antimic_span_map = Counter(row["microbe_name"] for row in antimic)



    #query for growth rates
    growth_query="""
    SELECT DISTINCT light_intensity,light_intensity_unit,
    light_dark_cycle,
    method,
 
    temp_celsius,
    rate,
    rate_unit
    FROM strain_profile
    WHERE strain_id = %s
    """

    cursor.execute(growth_query, (strain_id,))
    growth = cursor.fetchall()

    if growth[0]["rate"] is None :
       growth=[]


    #query for tumor data

    tumor_query="""
    SELECT DISTINCT tumor_cell_type,
    extract,
    exposure_hours,
    viable_cells_perc,
    viable_cell_char,
    extr_conc,
    extr_conc_unit,
    notes
    
    FROM strain_profile
    WHERE strain_id = %s
    """



    cursor.execute(tumor_query, (strain_id,))
    t_data = cursor.fetchall()




    #gene sequence query

    gene_query="""
    SELECT DISTINCT gene_name,gene_type,accession_number,associated_link,db_name
    FROM strain_profile
    WHERE strain_id = %s
    """
    cursor.execute(gene_query, (strain_id,))
    genes = cursor.fetchall()

    if genes[0]["gene_name"] is None :
       genes=[]

    #images query

    img_query="""
    SELECT DISTINCT image_filepath,photo_desc
    FROM strain_profile
    WHERE strain_id = %s
    """

    cursor.execute(img_query, (strain_id,))
    img = cursor.fetchall()

    if img[0]["image_filepath"] is None :
       img=[]

    #genome  query
    genome_query = """
        SELECT DISTINCT gen_acc,gen_assoc,gen_db,page_name 
        FROM strain_profile
        WHERE strain_id = %s
        """
    cursor.execute(genome_query, (strain_id,))
    genome = cursor.fetchall()

    if genome[0]["gen_acc"] is None:
        genome= []

    #reference query
    reference_ = """
            SELECT DISTINCT title,DOI,ref_link 
            FROM strain_profile
            WHERE strain_id = %s
            """
    cursor.execute(reference_, (strain_id,))
    reference_ = cursor.fetchall()

    if not strain:

        return "No data found.", 404
        con.close()

    try:
        return render_template("strain_profile.html", strain=strain,lineage=lineage,toxins=toxins,bioactive_comps=bioactive_comps,antimic=antimic,antimic_span_map=antimic_span_map,growth=growth,t_data=t_data,genes=genes,img=img,genome=genome,reference_=reference_)
    finally:
        con.close()


@app.route('/api/download-strain-profile/<int:strain_id>')
def download_strain_profile(strain_id):
    #download profile data from one strain
    con = db_pool.get_connection()
    cursor = con.cursor(dictionary=True, buffered=True)



    profile_query = """
                    SELECT strain_profile.strain_id AS strain_id,strain_profile.strain_code,
                    view_strain_lineage.empire,view_strain_lineage.kingdom, view_strain_lineage.subkingdom, view_strain_lineage.infrakingdom, view_strain_lineage.phylum, view_strain_lineage.subphylum, view_strain_lineage.class, view_strain_lineage.subclass, view_strain_lineage.order_name, view_strain_lineage.family, view_strain_lineage.subfamily, view_strain_lineage.genus, view_strain_lineage.species,
                    strain_profile.type_strain,strain_profile.novel_taxon,strain_profile.industrial_usage ,strain_profile.growth_medium ,
                    strain_profile.temp_tolerance,strain_profile.lifestyle,strain_profile.growth_medium,strain_profile.habitat_name,strain_profile.habitat_name,strain_profile.host_name,strain_profile.source_geo_location,strain_profile.col_lat,col_long,
                    strain_profile.water_depth_max,strain_profile.water_depth_min,strain_profile.col_date,strain_profile.site_temp_celsius_max,strain_profile.site_temp_celsius_min,strain_profile.col_source_type,strain_profile.morph_description,strain_profile.colony_gen_structure,
                    strain_profile.colony_color,strain_profile.cell_gen_shape,
                    strain_profile.cell_length_min,strain_profile.cell_length_max,strain_profile.cell_length_avg,strain_profile.cell_length_SD,strain_profile.cell_width_min,strain_profile.cell_width_max,strain_profile.cell_width_avg,strain_profile.cell_width_SD,
                    strain_profile.akinete_length_min,strain_profile.akinete_length_max,strain_profile.akinete_length_avg,strain_profile.akinete_length_SD,strain_profile.akinete_width_min,strain_profile.akinete_width_max,strain_profile.akinete_width_avg,strain_profile.akinete_width_SD,
                    strain_profile.heterocyte_length_min,strain_profile.heterocyte_length_max,strain_profile.heterocyte_length_avg,strain_profile.heterocyte_length_SD,strain_profile.heterocyte_width_min,strain_profile.heterocyte_width_max,strain_profile.heterocyte_width_avg,strain_profile.heterocyte_width_SD,
                    strain_profile.apical_cell_length_min,strain_profile.apical_cell_length_max,strain_profile.apical_cell_length_avg,strain_profile.apical_cell_length_SD,strain_profile.apical_cell_width_min,strain_profile.apical_cell_width_max,strain_profile.apical_cell_width_avg,strain_profile.apical_cell_width_SD, 
                    strain_profile.cyst_length_min,strain_profile.cyst_length_max,strain_profile.cyst_length_avg,strain_profile.cyst_length_SD,strain_profile.cyst_width_min,strain_profile.cyst_width_max,strain_profile.cyst_width_avg,strain_profile.cyst_width_SD,
                    strain_profile.filament_length_min,strain_profile.filament_length_max,strain_profile.filament_length_avg,strain_profile.filament_length_SD,strain_profile.filament_width_min,strain_profile.filament_width_max,strain_profile.filament_width_avg,strain_profile.filament_width_SD,
                    strain_profile.helix_pitch_length_min,strain_profile.helix_pitch_length_max,strain_profile.helix_pitch_length_avg,strain_profile.helix_pitch_length_SD,strain_profile.helix_diameter_min,strain_profile.helix_diameter_max,strain_profile.helix_diameter_avg,strain_profile.helix_diameter_SD,strain_profile.helix_height_min,strain_profile.helix_height_max,strain_profile.helix_height_avg,strain_profile.helix_height_SD

                        FROM strain_profile
                        JOIN view_strain_lineage ON strain_profile.strain_id=view_strain_lineage.strain_id

                        WHERE strain_profile.strain_id = %s


        """

    cursor.execute(profile_query, (strain_id,))

    rows = cursor.fetchall()

    MULTI_VALUE_FIELDS = {
        "col_source_type",
        "habitat_name"
    }

    strain = merge_strain_rows(rows, MULTI_VALUE_FIELDS)

    if not strain:
        return "Strain not found.", 404
        con.close()



    # separate fetch for toxins , its a one-to-many

    toxin_query = """
        SELECT DISTINCT toxin_name , presence , concentration, concentration_unit 
        FROM strain_profile

        WHERE strain_id = %s

        ORDER BY toxin_name

        """

    cursor.execute(toxin_query, (strain_id,))
    toxins = cursor.fetchall()

    if toxins[0]["toxin_name"] is None:
        toxins = []

    # separate fetch for bioactive compunds

    bioact_query = """
        SELECT DISTINCT compound_type, comp_presence , compound_name , comp_concentration , comp_concentration_unit 
        FROM strain_profile

        WHERE strain_id = %s 

        ORDER BY compound_type


        """

    cursor.execute(bioact_query, (strain_id,))
    bioactive_comps = cursor.fetchall()

    if bioactive_comps[0]["compound_type"] is None:
        bioactive_comps = []

    # query for antimicrobial seperately

    antimic_query = """
        SELECT DISTINCT microbe_name,inhibition_zone_char,transparency,solvent
        FROM strain_profile
        WHERE strain_id = %s

        ORDER BY microbe_name
        """

    cursor.execute(antimic_query, (strain_id,))
    antimic = cursor.fetchall()

    if antimic[0]["microbe_name"] is None:
        antimic = []

    from collections import Counter

    antimic_span_map = Counter(row["microbe_name"] for row in antimic)

    # query for growth rates
    growth_query = """
        SELECT DISTINCT light_intensity,light_intensity_unit,
        light_dark_cycle,
        method,

        temp_celsius,
        rate,
        rate_unit
        FROM strain_profile
        WHERE strain_id = %s
        """

    cursor.execute(growth_query, (strain_id,))
    growth = cursor.fetchall()

    if growth[0]["rate"] is None:
        growth = []

    # query for tumor data

    tumor_query = """
        SELECT DISTINCT tumor_cell_type,
        extract,
        exposure_hours,
        viable_cells_perc,
        viable_cell_char,
        extr_conc,
        extr_conc_unit,
        notes

        FROM strain_profile
        WHERE strain_id = %s
        """

    cursor.execute(tumor_query, (strain_id,))
    t_data = cursor.fetchall()

    # gene sequence query

    gene_query = """
        SELECT DISTINCT gene_name,gene_type,accession_number,associated_link,db_name
        FROM strain_profile
        WHERE strain_id = %s
        """
    cursor.execute(gene_query, (strain_id,))
    genes = cursor.fetchall()

    if genes[0]["gene_name"] is None:
        genes = []

    genome_query = """
            SELECT DISTINCT gen_acc,gen_assoc,gen_db,page_name 
            FROM strain_profile
            WHERE strain_id = %s
            """
    cursor.execute(genome_query, (strain_id,))
    genome = cursor.fetchall()

    if genome[0]["gen_acc"] is None:
        genome = []

    if not strain:
        return "No data found.", 404
        con.close()


    single_dicts = [
        ('', strain)
    ]

    multi_dicts = [('toxins',toxins),
                   ('bioactive_comps',bioactive_comps),
                   ('antimicrobial',antimic),
                   ('growth_rate',growth),
                   ('against_tumors',t_data),

                   ('genome',genome)
                   ]

    csv_str = build_single_strain_csv(single_dicts, multi_dicts,genes)

    strain_code = strain.get('strain_code', str(strain_id))

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                f'attachment; filename=strain_{strain_code}_profile.csv'
        }
    )


@app.route('/api/download-all-strain-profiles')
def download_all_strain_profiles():


    con = db_pool.get_connection()
    cursor = con.cursor(dictionary=True, buffered=True)

    # Get all strain IDs first
    cursor.execute("SELECT id FROM strain ORDER BY strain_code")
    strain_ids = [row['id'] for row in cursor.fetchall()]

    all_strain_data = []

    for strain_id in strain_ids:


        profile_query = """
                SELECT strain_profile.strain_id AS strain_id,strain_profile.strain_code,
                view_strain_lineage.empire,view_strain_lineage.kingdom, view_strain_lineage.subkingdom, view_strain_lineage.infrakingdom, view_strain_lineage.phylum, view_strain_lineage.subphylum, view_strain_lineage.class, view_strain_lineage.subclass, view_strain_lineage.order_name, view_strain_lineage.family, view_strain_lineage.subfamily, view_strain_lineage.genus, view_strain_lineage.species,
                strain_profile.type_strain,strain_profile.novel_taxon,strain_profile.industrial_usage ,strain_profile.growth_medium ,
                strain_profile.temp_tolerance,strain_profile.lifestyle,strain_profile.growth_medium,strain_profile.habitat_name,strain_profile.habitat_name,strain_profile.host_name,strain_profile.source_geo_location,strain_profile.col_lat,col_long,
                strain_profile.water_depth_max,strain_profile.water_depth_min,strain_profile.col_date,strain_profile.site_temp_celsius_max,strain_profile.site_temp_celsius_min,strain_profile.col_source_type,strain_profile.morph_description,strain_profile.colony_gen_structure,
                strain_profile.colony_color,strain_profile.cell_gen_shape,
                strain_profile.cell_length_min,strain_profile.cell_length_max,strain_profile.cell_length_avg,strain_profile.cell_length_SD,strain_profile.cell_width_min,strain_profile.cell_width_max,strain_profile.cell_width_avg,strain_profile.cell_width_SD,
                strain_profile.akinete_length_min,strain_profile.akinete_length_max,strain_profile.akinete_length_avg,strain_profile.akinete_length_SD,strain_profile.akinete_width_min,strain_profile.akinete_width_max,strain_profile.akinete_width_avg,strain_profile.akinete_width_SD,
                strain_profile.heterocyte_length_min,strain_profile.heterocyte_length_max,strain_profile.heterocyte_length_avg,strain_profile.heterocyte_length_SD,strain_profile.heterocyte_width_min,strain_profile.heterocyte_width_max,strain_profile.heterocyte_width_avg,strain_profile.heterocyte_width_SD,
                strain_profile.apical_cell_length_min,strain_profile.apical_cell_length_max,strain_profile.apical_cell_length_avg,strain_profile.apical_cell_length_SD,strain_profile.apical_cell_width_min,strain_profile.apical_cell_width_max,strain_profile.apical_cell_width_avg,strain_profile.apical_cell_width_SD, 
                strain_profile.cyst_length_min,strain_profile.cyst_length_max,strain_profile.cyst_length_avg,strain_profile.cyst_length_SD,strain_profile.cyst_width_min,strain_profile.cyst_width_max,strain_profile.cyst_width_avg,strain_profile.cyst_width_SD,
                strain_profile.filament_length_min,strain_profile.filament_length_max,strain_profile.filament_length_avg,strain_profile.filament_length_SD,strain_profile.filament_width_min,strain_profile.filament_width_max,strain_profile.filament_width_avg,strain_profile.filament_width_SD,
                strain_profile.helix_pitch_length_min,strain_profile.helix_pitch_length_max,strain_profile.helix_pitch_length_avg,strain_profile.helix_pitch_length_SD,strain_profile.helix_diameter_min,strain_profile.helix_diameter_max,strain_profile.helix_diameter_avg,strain_profile.helix_diameter_SD,strain_profile.helix_height_min,strain_profile.helix_height_max,strain_profile.helix_height_avg,strain_profile.helix_height_SD

                    FROM strain_profile
                    JOIN view_strain_lineage ON strain_profile.strain_id=view_strain_lineage.strain_id
                    
                    WHERE strain_profile.strain_id = %s

                    
                """

        cursor.execute(profile_query, (strain_id,))

        rows = cursor.fetchall()

        MULTI_VALUE_FIELDS = {
            "col_source_type",
            "habitat_name"
        }

        strain = merge_strain_rows(rows, MULTI_VALUE_FIELDS)

        if not strain:
            return "Strain not found.", 404
            con.close()

        # separate fetch for toxins , its a one-to-many

        toxin_query = """
                SELECT DISTINCT toxin_name , presence , concentration, concentration_unit 
                FROM strain_profile

                WHERE strain_id = %s

                ORDER BY toxin_name

                """

        cursor.execute(toxin_query, (strain_id,))
        toxins = cursor.fetchall()

        if toxins[0]["toxin_name"] is None:
            toxins = []

        # separate fetch for bioactive compunds

        bioact_query = """
                SELECT DISTINCT compound_type, comp_presence , compound_name , comp_concentration , comp_concentration_unit 
                FROM strain_profile

                WHERE strain_id = %s 

                ORDER BY compound_type


                """

        cursor.execute(bioact_query, (strain_id,))
        bioactive_comps = cursor.fetchall()

        if bioactive_comps[0]["compound_type"] is None:
            bioactive_comps = []

        # query for antimicrobial seperately

        antimic_query = """
                SELECT DISTINCT microbe_name,inhibition_zone_char,transparency,solvent
                FROM strain_profile
                WHERE strain_id = %s

                ORDER BY microbe_name
                """

        cursor.execute(antimic_query, (strain_id,))
        antimic = cursor.fetchall()

        if antimic[0]["microbe_name"] is None:
            antimic = []

        from collections import Counter

        antimic_span_map = Counter(row["microbe_name"] for row in antimic)

        # query for growth rates
        growth_query = """
                SELECT DISTINCT light_intensity,light_intensity_unit,
                light_dark_cycle,
                method,

                temp_celsius,
                rate,
                rate_unit
                FROM strain_profile
                WHERE strain_id = %s
                """

        cursor.execute(growth_query, (strain_id,))
        growth = cursor.fetchall()

        if growth[0]["rate"] is None:
            growth = []

        # query for tumor data

        tumor_query = """
                SELECT DISTINCT tumor_cell_type,
                extract,
                exposure_hours,
                viable_cells_perc,
                viable_cell_char,
                extr_conc,
                extr_conc_unit,
                notes

                FROM strain_profile
                WHERE strain_id = %s
                """

        cursor.execute(tumor_query, (strain_id,))
        t_data = cursor.fetchall()

        # gene sequence query

        gene_query = """
                SELECT DISTINCT gene_name,gene_type,accession_number,associated_link,db_name
                FROM strain_profile
                WHERE strain_id = %s
                """
        cursor.execute(gene_query, (strain_id,))
        genes = cursor.fetchall()

        if genes[0]["gene_name"] is None:
            genes = []

        genome_query = """
                    SELECT DISTINCT gen_acc,gen_assoc,gen_db,page_name 
                    FROM strain_profile
                    WHERE strain_id = %s
                    """
        cursor.execute(genome_query, (strain_id,))
        genome = cursor.fetchall()

        if genome[0]["gen_acc"] is None:
            genome = []

        if not strain:
            return "No data found.", 404
            con.close()

        single_dicts = [
            ('', strain)

        ]

        multi_dicts = [('toxins', toxins),
                       ('bioactive_comps', bioactive_comps),
                       ('antimicrobial', antimic),
                       ('growth_rate', growth),
                       ('against_tumors', t_data),

                       ('genome', genome)
                       ]





        all_strain_data.append((single_dicts, multi_dicts,genes))


    cursor.close()
    con.close()

    csv_str = build_all_strains_csv(all_strain_data)

    return Response(
        csv_str,
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                'attachment; filename=all_strain_profiles.csv'
        }
    )



if __name__ == '__main__':
    print("Connecting...")
    app.run(debug=True)
