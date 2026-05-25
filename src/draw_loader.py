import re
import pandas as pd
import pdfplumber


PDF_PATH = "roland-garros-2026-mens-singles-draw_5_24-2.pdf"


def is_player_line(text):

    if not text:
        return False

    text = text.strip()

    bad_tokens = [
        "1st ROUND",
        "2nd ROUND",
        "3rd ROUND",
        "FINAL",
        "Top",
        "Bottom",
        "ROLAND-GARROS",
        "2026",
        "24/05/2026",
    ]

    if any(token.lower() in text.lower() for token in bad_tokens):
        return False

    if len(text) < 4:
        return False

    if not any(c.isalpha() for c in text):
        return False

    return True


def clean_player_name(text):

    text = re.sub(r"\[.*?\]", "", text)

    # Remove winner initials like A.ZVEREV
    text = re.sub(r"\b[A-Z]\.[A-ZÀ-ÿ-]+\b", "", text)

    # Remove score fragments
    text = re.sub(r"\b\d+/\d+(?:\(\d+\))?\b", "", text)
    text = re.sub(r"\b\d+-\d+(?:\(\d+\))?\b", "", text)
    text = re.sub(r"\(\d+\)", "", text)

    # Remove match status fragments
    text = re.sub(r"\bAb\b", "", text)
    text = re.sub(r"\bRet\b", "", text)

    countries = {
        "ITA", "FRA", "USA", "ESP", "GER", "ARG", "SRB",
        "AUS", "GBR", "CZE", "CAN", "KAZ", "GRE", "NOR",
        "POL", "BRA", "NED", "CHI", "BEL", "CRO", "AUT",
        "POR", "MON", "CHN", "BIH", "BOL", "PER", "HKG",
        "PAR", "---",
    }

    markers = {"(Q)", "(W)", "(L)"}

    tokens = text.split()

    filtered = [
        t for t in tokens
        if t not in countries
        and t not in markers
    ]

    return " ".join(filtered).strip()


def normalize_draw_name(name):

    parts = name.split()

    if len(parts) < 2:
        return name.title()

    surname = parts[0]
    first_names = parts[1:]

    normalized = " ".join(first_names + [surname])

    return normalized.title()


def load_roland_garros_draw():

    lines = []

    with pdfplumber.open(PDF_PATH) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                lines.extend(text.split("\n"))

    lines = [
        line.strip()
        for line in lines
        if is_player_line(line)
    ]

    players = []

    for line in lines:

        cleaned = clean_player_name(line)

        if len(cleaned.split()) >= 2:
            players.append(cleaned)

    rows = []

    for i in range(0, len(players) - 1, 2):

        player_a = normalize_draw_name(players[i])
        player_b = normalize_draw_name(players[i + 1])

        rows.append(
            {
                "player_a": player_a,
                "player_b": player_b,
            }
        )

    draw = pd.DataFrame(rows)

    draw = draw.drop_duplicates().reset_index(drop=True)
    try:
        aliases = pd.read_csv("data/name_aliases.csv")
        alias_map = dict(zip(aliases["draw_name"], aliases["player_name"]))

        draw["player_a"] = draw["player_a"].replace(alias_map)
        draw["player_b"] = draw["player_b"].replace(alias_map)

    except FileNotFoundError:
        pass

    return draw


if __name__ == "__main__":

    draw = load_roland_garros_draw()

    print(draw.head(50))
    print(draw.shape)