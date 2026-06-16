from vocab import SWING_VOCAB

def discretize_swing(ratio):
    if   ratio < 0.54: return SWING_VOCAB["SW0"]
    elif ratio < 0.58: return SWING_VOCAB["SW1"]
    elif ratio < 0.62: return SWING_VOCAB["SW2"]
    elif ratio < 0.67: return SWING_VOCAB["SW3"]
    elif ratio < 0.72: return SWING_VOCAB["SW4"]
    else:              return SWING_VOCAB["SW5"]

def compute_swing_prepass(merged):
    beat_grid = (merged[["bar", "beat", "beat_onset"]]
                    .drop_duplicates()
                    .sort_values(["bar", "beat"])
                    .reset_index(drop=True))

    swing_map = {}   # (bar, beat) → SW token

    for i in range(len(beat_grid) - 1):
        curr = beat_grid.iloc[i]
        next_ = beat_grid.iloc[i + 1]

        # only pair odd beats with their following even beat
        # beat 1→2, beat 3→4 within same bar
        is_valid_pair = (
            curr["beat"] in [1, 3] and
            next_["beat"] == curr["beat"] + 1 and
            next_["bar"] == curr["bar"]
        )

        if not is_valid_pair:
            swing_map[(curr["bar"], curr["beat"])] = SWING_VOCAB["SW_UNKNOWN"]
            continue

        # find the beat after the pair for denominator
        after = beat_grid.iloc[i + 2] if i + 2 < len(beat_grid) else None

        if after is None:
            swing_map[(curr["bar"], curr["beat"])] = SWING_VOCAB["SW_UNKNOWN"]
            swing_map[(next_["bar"], next_["beat"])] = SWING_VOCAB["SW_UNKNOWN"]
            continue

        pair_duration = after["beat_onset"] - curr["beat_onset"]
        long_duration = next_["beat_onset"] - curr["beat_onset"]
        ratio = long_duration / pair_duration

        sw_token = discretize_swing(ratio)
        swing_map[(curr["bar"], curr["beat"])]  = sw_token
        swing_map[(next_["bar"], next_["beat"])] = sw_token   # both beats share same SW

    # map back onto merged
    merged["swing_ratio"] = merged.apply(
        lambda r: swing_map.get((r["bar"], r["beat"]), SWING_VOCAB["SW_UNKNOWN"]),
        axis=1
    )
    return merged