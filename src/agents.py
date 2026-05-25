def _name(player):
    return player["player_name"]


def _result(agent, signal, favored_player, evidence, message):
    return {
        "agent": agent,
        "signal": signal,
        "favored_player": favored_player,
        "evidence": evidence,
        "message": message,
    }


def ranking_agent(player_a, player_b):
    evidence = (
        f"{_name(player_a)}: rank {int(player_a['rank'])}, "
        f"{int(player_a['rank_points'])} pts | "
        f"{_name(player_b)}: rank {int(player_b['rank'])}, "
        f"{int(player_b['rank_points'])} pts"
    )

    if player_a["rank"] < player_b["rank"]:
        return _result(
            "Ranking",
            "ATP rank and ranking points",
            _name(player_a),
            evidence,
            "Better current ATP ranking.",
        )

    if player_b["rank"] < player_a["rank"]:
        return _result(
            "Ranking",
            "ATP rank and ranking points",
            _name(player_b),
            evidence,
            "Better current ATP ranking.",
        )

    return _result(
        "Ranking",
        "ATP rank and ranking points",
        None,
        evidence,
        "Rankings are equal.",
    )


def elo_agent(player_a, player_b, surface):
    surface_col = {
        "Clay": "clay_elo",
        "Hard": "hard_elo",
        "Grass": "grass_elo",
    }[surface]

    a_score = 0.4 * player_a["overall_elo"] + 0.6 * player_a[surface_col]
    b_score = 0.4 * player_b["overall_elo"] + 0.6 * player_b[surface_col]

    evidence = (
        f"{_name(player_a)}: overall {player_a['overall_elo']:.0f}, "
        f"{surface} {player_a[surface_col]:.0f}, combined {a_score:.0f} | "
        f"{_name(player_b)}: overall {player_b['overall_elo']:.0f}, "
        f"{surface} {player_b[surface_col]:.0f}, combined {b_score:.0f}"
    )

    if a_score > b_score + 35:
        return _result(
            "Elo",
            f"Overall Elo + {surface} Elo",
            _name(player_a),
            evidence,
            "Stronger combined Elo profile.",
        )

    if b_score > a_score + 35:
        return _result(
            "Elo",
            f"Overall Elo + {surface} Elo",
            _name(player_b),
            evidence,
            "Stronger combined Elo profile.",
        )

    return _result(
        "Elo",
        f"Overall Elo + {surface} Elo",
        None,
        evidence,
        "Elo profiles are close.",
    )


def surface_agent(player_a, player_b, surface):
    if surface == "Clay":
        a_value = player_a["recent_clay_win_rate"]
        b_value = player_b["recent_clay_win_rate"]
        signal = "Recent clay win rate"
    else:
        a_value = player_a["recent_win_rate"]
        b_value = player_b["recent_win_rate"]
        signal = "Recent win rate"

    evidence = f"{_name(player_a)}: {a_value:.2f} | {_name(player_b)}: {b_value:.2f}"

    if a_value > b_value + 0.12:
        return _result(
            "Surface",
            signal,
            _name(player_a),
            evidence,
            f"Better {surface.lower()}-relevant form.",
        )

    if b_value > a_value + 0.12:
        return _result(
            "Surface",
            signal,
            _name(player_b),
            evidence,
            f"Better {surface.lower()}-relevant form.",
        )

    return _result(
        "Surface",
        signal,
        None,
        evidence,
        "Surface-relevant form is close.",
    )


def form_agent(player_a, player_b, surface=None):
    a_value = player_a["recent_win_rate"]
    b_value = player_b["recent_win_rate"]

    evidence = f"{_name(player_a)}: {a_value:.2f} | {_name(player_b)}: {b_value:.2f}"

    if a_value > b_value + 0.12:
        return _result(
            "Form",
            "Recent overall win rate",
            _name(player_a),
            evidence,
            "Better recent overall form.",
        )

    if b_value > a_value + 0.12:
        return _result(
            "Form",
            "Recent overall win rate",
            _name(player_b),
            evidence,
            "Better recent overall form.",
        )

    return _result(
        "Form",
        "Recent overall win rate",
        None,
        evidence,
        "Recent form is close.",
    )


def fatigue_agent(player_a, player_b):
    a_load = player_a.get("recent_match_count", 0)
    b_load = player_b.get("recent_match_count", 0)

    evidence = f"{_name(player_a)}: {int(a_load)} recent matches | {_name(player_b)}: {int(b_load)} recent matches"

    if a_load >= b_load + 2:
        return _result(
            "Fatigue",
            "Recent match load",
            _name(player_b),
            evidence,
            "Lighter recent match load.",
        )

    if b_load >= a_load + 2:
        return _result(
            "Fatigue",
            "Recent match load",
            _name(player_a),
            evidence,
            "Lighter recent match load.",
        )

    return _result(
        "Fatigue",
        "Recent match load",
        None,
        evidence,
        "Recent match load is similar.",
    )


def age_agent(player_a, player_b):
    evidence = f"{_name(player_a)}: {int(player_a['age'])} | {_name(player_b)}: {int(player_b['age'])}"

    diff = player_a["age"] - player_b["age"]

    if abs(diff) < 4:
        return _result(
            "Age",
            "Age difference",
            None,
            evidence,
            "No major age gap.",
        )

    younger = player_a if diff < 0 else player_b

    return _result(
        "Age",
        "Age difference",
        _name(younger),
        evidence,
        "Younger player may have a physical freshness advantage.",
    )


def h2h_agent_from_summary(player_a_name, player_b_name, h2h_summary, surface):
    a_wins = h2h_summary["player_a_wins"]
    b_wins = h2h_summary["player_b_wins"]
    clay_a = h2h_summary["clay_player_a_wins"]
    clay_b = h2h_summary["clay_player_b_wins"]

    evidence = f"Overall H2H: {player_a_name} {a_wins}-{b_wins} {player_b_name}; clay H2H: {clay_a}-{clay_b}"

    if a_wins + b_wins == 0:
        return _result(
            "Head-to-head",
            "Direct meetings",
            None,
            evidence,
            "No direct historical meetings found.",
        )

    if surface == "Clay" and clay_a + clay_b > 0:
        if clay_a > clay_b:
            return _result(
                "Head-to-head",
                "Overall and clay H2H",
                player_a_name,
                evidence,
                "Better clay head-to-head record.",
            )
        if clay_b > clay_a:
            return _result(
                "Head-to-head",
                "Overall and clay H2H",
                player_b_name,
                evidence,
                "Better clay head-to-head record.",
            )

    if a_wins > b_wins:
        return _result(
            "Head-to-head",
            "Overall H2H",
            player_a_name,
            evidence,
            "Better direct historical record.",
        )

    if b_wins > a_wins:
        return _result(
            "Head-to-head",
            "Overall H2H",
            player_b_name,
            evidence,
            "Better direct historical record.",
        )

    return _result(
        "Head-to-head",
        "Direct meetings",
        None,
        evidence,
        "Head-to-head record is balanced.",
    )


def matchup_agent(player_a, player_b):
    a_serve = player_a.get("serve_strength", 0.5)
    b_serve = player_b.get("serve_strength", 0.5)
    a_return = player_a.get("return_strength", 0.5)
    b_return = player_b.get("return_strength", 0.5)

    a_score = 0.55 * a_serve + 0.45 * a_return
    b_score = 0.55 * b_serve + 0.45 * b_return

    evidence = (
        f"{_name(player_a)}: serve {a_serve:.2f}, return {a_return:.2f}, score {a_score:.2f} | "
        f"{_name(player_b)}: serve {b_serve:.2f}, return {b_return:.2f}, score {b_score:.2f}"
    )

    if a_score > b_score + 0.03:
        return _result(
            "Matchup",
            "Serve/return profile",
            _name(player_a),
            evidence,
            "Stronger recent serve/return profile.",
        )

    if b_score > a_score + 0.03:
        return _result(
            "Matchup",
            "Serve/return profile",
            _name(player_b),
            evidence,
            "Stronger recent serve/return profile.",
        )

    return _result(
        "Matchup",
        "Serve/return profile",
        None,
        evidence,
        "Serve/return profiles are close.",
    )


def similarity_rag_agent(similar_matches, player_a_name, player_b_name):
    if similar_matches.empty:
        return _result(
            "Similarity/RAG",
            "Retrieved similar matches",
            None,
            "No matches retrieved.",
            "No similar historical matches found.",
        )

    a_wins = (similar_matches["winner_name"] == player_a_name).sum()
    b_wins = (similar_matches["winner_name"] == player_b_name).sum()
    total = len(similar_matches)

    evidence = f"Top {total} similar matches: {player_a_name} wins {a_wins}, {player_b_name} wins {b_wins}"

    if a_wins > b_wins:
        return _result(
            "Similarity/RAG",
            "Retrieved similar matches",
            player_a_name,
            evidence,
            "Retrieved evidence favors this player.",
        )

    if b_wins > a_wins:
        return _result(
            "Similarity/RAG",
            "Retrieved similar matches",
            player_b_name,
            evidence,
            "Retrieved evidence favors this player.",
        )

    return _result(
        "Similarity/RAG",
        "Retrieved similar matches",
        None,
        evidence,
        "Retrieved matches do not clearly favor either player.",
    )


def news_context_agent(player_a_name, player_b_name):
    from news_context import analyze_news

    news = analyze_news(player_a_name, player_b_name)

    a = news["player_a"]
    b = news["player_b"]

    a_score = a["score"]
    b_score = b["score"]

    a_headlines = len(a["headlines"])
    b_headlines = len(b["headlines"])

    evidence = (
        f"{player_a_name}: score {a_score}, {a_headlines} headlines | "
        f"{player_b_name}: score {b_score}, {b_headlines} headlines"
    )

    if a.get("error") or b.get("error"):
        return _result(
            "News context",
            "Recent news retrieval",
            None,
            evidence,
            "News retrieval partially failed, so this signal is treated as neutral.",
        )

    if a_score > b_score:
        return _result(
            "News context",
            "Recent headlines",
            player_a_name,
            evidence,
            "Recent news context is more favorable.",
        )

    if b_score > a_score:
        return _result(
            "News context",
            "Recent headlines",
            player_b_name,
            evidence,
            "Recent news context is more favorable.",
        )

    return _result(
        "News context",
        "Recent headlines",
        None,
        evidence,
        "News context is balanced or unclear.",
    )


def upset_risk_agent(result, player_a, player_b):
    winner = result["predicted_winner"]
    winner_prob = result["winner_probability"]

    rank_favorite = player_a if player_a["rank"] < player_b["rank"] else player_b

    evidence = f"Model winner probability: {winner_prob:.1%}; ranking favorite: {_name(rank_favorite)}"

    if winner_prob < 0.56:
        return _result(
            "Upset risk",
            "Prediction uncertainty",
            None,
            evidence,
            "High uncertainty: probabilities are very close.",
        )

    if _name(rank_favorite) != winner:
        return _result(
            "Upset risk",
            "Model vs ranking disagreement",
            winner,
            evidence,
            "Possible upset structure: model favorite is not the higher-ranked player.",
        )

    if winner_prob < 0.62:
        return _result(
            "Upset risk",
            "Prediction uncertainty",
            None,
            evidence,
            "Moderate upset risk: model confidence is not strong.",
        )

    return _result(
        "Upset risk",
        "Prediction uncertainty",
        winner,
        evidence,
        "Limited upset risk based on current model confidence.",
    )


def judge_agent(result, agent_outputs):
    winner = result["predicted_winner"]
    prob = result["winner_probability"]

    votes = {}
    for output in agent_outputs:
        favored = output.get("favored_player")
        if favored is not None:
            votes[favored] = votes.get(favored, 0) + 1

    if prob > 0.70:
        confidence = "high"
    elif prob > 0.58:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "agent": "Judge",
        "favored_player": winner,
        "message": (
            f"Judge predicts {winner} with {prob:.1%} probability. "
            f"Overall confidence is {confidence}. Agent votes: {votes}."
        ),
    }