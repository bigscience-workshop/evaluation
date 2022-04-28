""" Usage:
    <file-name> --src=SOURCE_FILE_PATH --placeholder=PLACEHOLDER_FILE_PATH --targ=TARGET_FILE_PATH --lang=LANG
"""

from audioop import bias
from docopt import docopt
import pandas as pd
import re

lang_country_map = {"HI":"India"}

def fetch_sub_placeholder_ds(placeholder_ds, lang):
    lang_columns = [c for c in placeholder_ds.columns if c.startswith(f'{lang}_')]
    sub_placeholder_ds = placeholder_ds[lang_columns]
    sub_placeholder_ds.columns = sub_placeholder_ds.columns.str.lstrip(f"{lang}_")
    sub_placeholder_ds["EN_NATION"]=placeholder_ds["NATION"]
    return sub_placeholder_ds

def fetch_sub_shades_ds(shades_ds, lang):
    
    relevant_columns = ['original target country', f'{lang} Shade Stereotype']
    sub_shades_ds = shades_ds[relevant_columns]
    sub_shades_ds['is_stereotype'] = shades_ds[ f'stereotype culturally perceived in {lang_country_map[lang]}?']
    return sub_shades_ds

def replace_all_occurrence(sent, replacement_dict):
    for occ, val in replacement_dict.items():
        sent = re.sub(occ,val,sent)
    return sent

def generate_final_data(sub_shades_ds, sub_placeholder_ds):
    
    data = []
    for i1, base_row in sub_shades_ds.iterrows():
        base_sentence = base_row[f'{lang} Shade Stereotype']
        stereotype = "no"
        bias_type = "nationality"
        for  i2, r2 in sub_placeholder_ds.iterrows():
            # replacement_dict = {col: r2[col] for col in sub_placeholder_ds}
            replacement_dict = {"NATION": r2['NATION'], "CITIZEN_PL": r2['CITIZEN_PL'], "CITIZEN": r2['CITIZEN'] }
            if r2['EN_NATION'] == base_row['original target country']:
                stereotype = base_row["is_stereotype"] 
            sentence = replace_all_occurrence(base_sentence, replacement_dict)
            data.append([sentence, stereotype, bias_type])
        final_ds = pd.DataFrame(data, columns = ['sentence', 'stereotype', 'bias_type'])
    return final_ds


if __name__ == "__main__":
# Parse command line arguments

    args = docopt(__doc__)
    src_path = args["--src"]
    placeholder_path = args['--placeholder']
    targ_path = args["--targ"]
    lang = args["--lang"]


    shades_ds = pd.read_csv(src_path, sep=',', encoding='utf-8')
    placeholder_ds = pd.read_csv(placeholder_path, sep =',', encoding='utf-8')
    sub_placeholder_ds = fetch_sub_placeholder_ds(placeholder_ds, lang)
    sub_shades_ds = fetch_sub_shades_ds(shades_ds, lang)

    final_ds = generate_final_data(sub_shades_ds, sub_placeholder_ds)
    final_ds.to_csv(targ_path, encoding='utf-8', index=False)







    
