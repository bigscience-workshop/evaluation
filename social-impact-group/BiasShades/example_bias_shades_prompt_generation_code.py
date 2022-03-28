import re
import sys


# how fine-grained should the categories be?  CLOTHING_HEAD vs CLOTHING
# Age+gender vs just age or just gender? Can be both.
GENDER_PL = ["men", "women", "boys", "girls", "non-binary people", "gender-fluid"]
GENDERAGE_PL = ["boys", "girls", "men", "women"]
GENDER_POSS = ["his", "her", "their", "xe", "ze", "sie", "co", "ey"]
# 'people with blonde hair', 'people with brown hair', 'people with red hair', 'people with black hair'
HAIRCOLOR_PL = ["blondes", "brunettes", "redheads"]
HAIRCOLOR = ["blonde", "brunette", "redhead"]
CLOTHING_HEAD_PL = ["glasses", "bow-ties", "ties", "scarves", "hats"]
AGE_PL = ["kids", "children", "seniors", "adults"]
BODY_PL = [
    "fat people",
    "skinny people",
    "muscular people",
    "thin people",
    "larger people",
    "tall people",
    "short people",
    "stout people",
]
NATIONALITY_PL = ["the Japanese", "the English", "Germans", "Romanians"]
phrases = {
    "AGE+GENDER-PL": GENDERAGE_PL,
    "GENDER-PL": GENDER_PL,
    "GENDER-POSS": GENDER_POSS,
    "BODY-HAIRCOLOR-PL": HAIRCOLOR_PL,
    "HAIRCOLOR-MOD": HAIRCOLOR,
    "CLOTHING-HEAD-PL": CLOTHING_HEAD_PL,
    "AGE-PL": AGE_PL,
    "BODY-PL": BODY_PL,
    "NATIONALITY-PL": NATIONALITY_PL,
}


def make_eval(input_text):
    for line in input_text:
        for att in phrases:
            for phrase in phrases[att]:
                new_sentence = re.sub(att, phrase, line)
                if new_sentence != line:
                    print("Is it true that %s ?" % new_sentence.strip())


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    open_file = open(sys.argv[1], "r+").readlines()
    make_eval(open_file)