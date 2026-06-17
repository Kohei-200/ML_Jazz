import torch, os
from vocab import *
from beat_aligner import *
from chord_parser import *
from chorus_tagger import *
from noteshape import *
from tension_tagger import *
from wjazzd_query import *


def build_beat_header(bar_row):
    beat_num = bar_row["beat"]
    chord_tokens = parse_chord(bar_row["chord"])   # → 7 strings
    chord_ids = [
        ROOT_VOCAB[chord_tokens[0]],
        SHAPE_VOCAB[chord_tokens[1]],
        TEN7_VOCAB[chord_tokens[2]],
        ALTER_VOCAB[chord_tokens[3]],
        TEN9_VOCAB[chord_tokens[4]],
        TEN11_VOCAB[chord_tokens[5]],
        TEN13_VOCAB[chord_tokens[6]]
    ]
    return [beat_num] + chord_ids

def build_bar_header(bar_row, chorus_tags):

    instrument = INSTRUMENT_VOCAB[bar_row["instrument"]]
    tempo_id      = TEMPO_VOCAB.get(bar_row["tempoclass"], 5) # USED for stratify
    section_id    = SECTION_VOCAB.get(bar_row["form"], 4)

    chorus_label = chorus_tags.get(bar_row["chorus_id"], "UNKNOWN")  # EARLY/MID/PEAK/LATE classification
    chorus_id    = CHORUS_VOCAB.get(chorus_label, 4)

    return [tempo_id, section_id, chorus_id, instrument]

def chord_carry_forward(notes):
    notes = notes.copy()
    last_known = "UNKOWN"
    for i, row in notes.iterrows():
        if notes.at[i, "chord"].strip() == "":
            notes.at[i, "chord"] = last_known
        else:
            last_known = notes.at[i, "chord"]
    return notes

def compute_velocity_bins(merged):
    loud_vals = merged.loc[~merged["is_rest"], "loud_max"]
    loud_vals = loud_vals[loud_vals > loud_vals.quantile(0.01)]
    return loud_vals.quantile([0.2, 0.4, 0.6, 0.8]).values

def treat_nans(merged):
    solo_cols = ["instrument", "tempoclass", "key", "chorus_count", "style", "avgtempo"]
    merged[solo_cols] = merged[solo_cols].ffill().bfill()

    merged["eventid"]       = merged["eventid"].fillna(-1).astype(int)
    merged["is_rest"] =  merged["eventid"] == -1
    merged["tatum"]         = merged["tatum"].fillna(-1).astype(int)
    merged["pitch"]         = merged["pitch"].fillna(0).astype(int)
    merged["note_onset"]    = merged["note_onset"].fillna(merged["beat_onset"])

    rest_mask = merged["is_rest"]
    merged.loc[rest_mask, "note_duration"] = (
        merged["beat_onset"].shift(-1) - merged["beat_onset"]
    )[rest_mask].fillna(0)
    merged["loud_max"]      = merged["loud_max"].fillna(0.0)
    merged["loud_relpos"]   = merged["loud_relpos"].fillna(0.0)
    merged["f0_mod"]        = merged["f0_mod"].fillna(0).replace("", 0)
    merged["beat_duration"] = merged["beat_duration"].ffill().bfill()
    return merged

def assign_velocity(loud_max, bins):
    if loud_max == 0.0:
        return VEL_VOCAB["REST"]
    labels = ["pp", "mp", "mf", "f", "ff"]
    for i, threshold in enumerate(bins):
        if loud_max < threshold:
            return VEL_VOCAB[labels[i]]
    return VEL_VOCAB["ff"]

def compute_phrase_roles(merged, phrases):
    """
    map eventid to phrase role before the note loop
    # assume no inter-phrase gaps
    """
    role_map = {}

    for _, phrase in phrases.iterrows():
        phrase_notes = merged[
            (merged["eventid"] >= phrase["start"]) &
            (merged["eventid"] <= phrase["end"]) &
            (~merged["is_rest"])
        ]

        if phrase_notes.empty:
            continue

        peak_eventid  = phrase_notes.loc[phrase_notes["pitch"].idxmax(), "eventid"]
        start_eventid = phrase["start"]
        end_eventid   = phrase["end"]

        for _, row in phrase_notes.iterrows():
            eid = row["eventid"]
            if eid == start_eventid:
                role_map[eid] = PHRASEROLE_VOCAB["START"]
            elif eid == end_eventid:
                role_map[eid] = PHRASEROLE_VOCAB["LAND"]
            elif eid == peak_eventid:
                role_map[eid] = PHRASEROLE_VOCAB["PEAK"]
            else:
                role_map[eid] = PHRASEROLE_VOCAB["CONT"]

    # map back onto merged
    merged["phrase_role"] = merged.apply(
        lambda r: PHRASEROLE_VOCAB["REST"] if r["is_rest"]
                    else role_map.get(int(r["eventid"]), PHRASEROLE_VOCAB["UNKNOWN"]),
        axis=1
    )
    return merged

def build_note_tuple(row, vel_bins):
    """
    row -> all precomputed columns
    vel_bins -> solo-level quantile range
    """
    if row["is_rest"]:
        return {
            "bar":        int(row["bar"]),
            "beat":       int(row["beat"]),
            "beatpos":    compute_beatpos(row["beat"], row["beat_onset"],
                                        row["beat_onset"], row["beat_duration"]),
            "swing":      row["swing_ratio"],
            "duration":   quantise_duration(row["note_duration"], row["beat_duration"]),
            "pitch":      0,
            "interval":   13,
            "octave": 10, 
            "tension":    TENSION_VOCAB["REST"],
            "phrase_role":PHRASEROLE_VOCAB["REST"],
            "vel":        VEL_VOCAB["REST"],
            "noteshape":  NOTESHAPE_VOCAB["REST"],
            "is_rest":    True
        }

    return {
        "bar":        int(row["bar"]),
        "beat":       int(row["beat"]),
        "beatpos":    compute_beatpos(row["beat"], row["note_onset"],
                                    row["beat_onset"], row["beat_duration"]),
        "swing":      row["swing_ratio"],
        "duration":   quantise_duration(row["note_duration"], row["beat_duration"]),
        "pitch":      int(round(row["pitch"])),
        "interval":   int(row["interval"]),
        "octave": int(round(row["pitch"])) // 12,
        "tension":    int(row["tension_role"]),
        "phrase_role":int(row["phrase_role"]),
        "vel":        assign_velocity(row["loud_max"], vel_bins),
        "noteshape":  assign_noteshape(row["loud_relpos"]),
        "is_rest":    False
    }

def assemble(token_seq, bar_headers, beat_headers):
    seq = []
    beat_header = []
    current_bar = -1
    current_beat = -1
    BAR_TK = SPECIAL_TOKENS["BAR"]
    BEAT_TK = SPECIAL_TOKENS["BEAT"]
    NOTE_TK = SPECIAL_TOKENS["NOTE"]
    for note in token_seq:
        bar_num = note["bar"]
        beat_num = note["beat"]
        if bar_num != current_bar:
            seq.extend([BAR_TK] + bar_headers[bar_num])
        current_bar = bar_num
        if beat_num != current_beat:
            seq.extend([BEAT_TK] + beat_headers[(bar_num, beat_num)])
        current_beat = beat_num

        note_tuple =[
            NOTE_TK,
            note["beatpos"],
            note["swing"],
            note["duration"],
            note["pitch"],
            note["interval"],
            note["octave"],
            note["tension"],
            note["phrase_role"],
            note["vel"],
            note["noteshape"]
        ]
        seq.extend(note_tuple)
    return seq

def process_and_save_corpus(conn, output_dir):
    success_count = 0
    error_count = 0
    melids = get_valid_ids(conn) # 343 ids available

    for melid in melids:
        try:
            ####### QUERIES #########
            notes   = get_solo(melid, conn)
            beats   = get_beats(melid, conn)
            phrases = get_phrases(melid, conn)

            ####### MERGE #########
            merged = beats.merge(notes, on=["bar", "beat"], how="left")
            merged = treat_nans(merged)
            merged  = chord_carry_forward(merged)

            ####### SOLO INFO #########
            chorus_tags = classify_chorus(merged)
            vel_bins = compute_velocity_bins(merged)   # once per solo
            merged = compute_swing_prepass(merged)
            last_bar = merged["bar"].max()

            internal_bars = merged[merged["bar"] < last_bar]

            if (internal_bars.groupby("bar")["beat"].max() != 4).any():
                continue

            ####### BAR INFO #########
            bar_headers = {}
            for bar_num, bar_group in merged.groupby("bar"):
                bar_row = bar_group.iloc[0]
                bar_headers[bar_num] = build_bar_header(bar_row, chorus_tags)
            
            beat_headers = {}
            for (bar_num, beat_num), beat_group in merged.groupby(["bar", "beat"]):
                beat_row = beat_group.iloc[0]
                beat_headers[(bar_num, beat_num)] = build_beat_header(beat_row)

            ####### NOTE INFO #########
            merged["interval"] = merged.apply(
            lambda r: 13 if r["is_rest"] else compute_interval(
                r["pitch"],
                bar_headers[r["bar"]]),
            axis=1
            )
            merged["next_interval"] = merged["interval"].shift(-1).fillna(-1).astype(int)
            merged.loc[merged["is_rest"], "interval"]      = 13
            merged.loc[merged["is_rest"], "next_interval"] = 13

            merged["tension_role"] = merged.apply(
                lambda r: TENSION_VOCAB["REST"] if r["is_rest"]
                else compute_tension_role(
                    r["interval"],
                    beat_headers[(r["bar"], r["beat"])],
                    r["next_interval"]
                ),
                axis = 1
                )
            merged = compute_phrase_roles(merged, phrases)

            ####### TOKENIZE NOTES #######
            token_seq = []
            for _, row in merged.iterrows():
                note = build_note_tuple(row, vel_bins)
                token_seq.append(note)
        # [bar, beatpos, swing, duration, pitch, interval, octave, tension, phrase_role, vel, noteshape, is_rest]

            seq_1D  = assemble(token_seq, bar_headers, beat_headers)
            seq_1D.append(SPECIAL_TOKENS["SOLO_END"])

            seq_tensor = torch.tensor(seq_1D, dtype = torch.long)

            save_path = os.path.join(output_dir, f"melid_{melid}.pt")
            torch.save(seq_tensor, save_path)

            success_count += 1

        except Exception as e:
            print(f"Fail on {melid}: {e}")
            error_count += 1
    print(f"saved to {output_dir}")