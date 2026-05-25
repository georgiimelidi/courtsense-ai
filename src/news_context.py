import urllib.parse
import xml.etree.ElementTree as ET

import requests


POSITIVE_KEYWORDS = [
    "fit",
    "confident",
    "returns",
    "wins",
    "dominates",
    "strong",
    "healthy",
    "impressive",
    "advances",
]

NEGATIVE_KEYWORDS = [
    "injury",
    "injured",
    "withdraws",
    "withdrawal",
    "retires",
    "pain",
    "fatigue",
    "struggles",
    "doubt",
    "illness",
]


def fetch_google_news_rss(query: str, max_items: int = 8):
    encoded_query = urllib.parse.quote(query)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    items = []

    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")

        if title:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "published": pub_date,
                }
            )

    return items


def score_headlines(items):
    score = 0
    matched_signals = []

    for item in items:
        title = item["title"].lower()

        for keyword in POSITIVE_KEYWORDS:
            if keyword in title:
                score += 1
                matched_signals.append(f"+ {keyword}")

        for keyword in NEGATIVE_KEYWORDS:
            if keyword in title:
                score -= 1
                matched_signals.append(f"- {keyword}")

    return score, matched_signals


def analyze_player_news(player_name: str):
    query = f'"{player_name}" tennis Roland Garros injury form'

    try:
        items = fetch_google_news_rss(query)
    except Exception as e:
        return {
            "player_name": player_name,
            "score": 0,
            "headlines": [],
            "signals": [],
            "error": str(e),
        }

    score, signals = score_headlines(items)

    return {
        "player_name": player_name,
        "score": score,
        "headlines": items,
        "signals": signals,
        "error": None,
    }


def analyze_news(player_a_name: str, player_b_name: str):
    player_a_news = analyze_player_news(player_a_name)
    player_b_news = analyze_player_news(player_b_name)

    return {
        "player_a": player_a_news,
        "player_b": player_b_news,
    }


if __name__ == "__main__":
    result = analyze_news("Jannik Sinner", "Alexander Zverev")

    print(result["player_a"]["player_name"], result["player_a"]["score"])
    for h in result["player_a"]["headlines"][:5]:
        print("-", h["title"])

    print(result["player_b"]["player_name"], result["player_b"]["score"])
    for h in result["player_b"]["headlines"][:5]:
        print("-", h["title"])