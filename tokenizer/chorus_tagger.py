import pandas as pd
from chord_parser import parse_chord

def classify_chorus(notes):
    """
    # Intensity =
    note_density(number of notes in chorus / number of bars in chorus)
    × tension_ratio (number of notes outside {R, 3, 5, 7} / total notes in chorus)
    """
    chorus_ids = notes["chorus_id"].dropna().unique()
    num_choruses = len(chorus_ids)
    if num_choruses == 0:
        return {}
    intensities = {}

    for chorus_idx in chorus_ids:
        chorus_block = notes[notes["chorus_id"] == chorus_idx]
        notes_in_cho = len(chorus_block)

        if notes_in_cho == 0:
            intensities[chorus_idx] = 0
            continue

        bars_in_cho = len(chorus_block["bar"].unique())
        bars_in_cho = max(1, bars_in_cho)

        note_density = notes_in_cho / bars_in_cho

        notes_outside_in_cho = build_chorus_map(chorus_block, chorus_idx)
        tension_ratio = notes_outside_in_cho / notes_in_cho

        intensity = note_density * tension_ratio
        intensities[chorus_idx] = intensity
    
    peak_chorus_id = max(intensities, key = intensities.get)
    chorus_tags = {}
    for i, chorus_idx in enumerate(chorus_ids):
        if chorus_idx == peak_chorus_id:
            chorus_tags[chorus_idx] = "PEAK"
        elif i == 0:
            chorus_tags[chorus_idx] = "EARLY"
        elif i == num_choruses - 1:
            chorus_tags[chorus_idx] = "LATE"
        else:
            chorus_tags[chorus_idx] = "MID"

    return chorus_tags


def build_chorus_map(chorus_block, chorus_idx):
    notes_outside = 0
    active_chord = ""
    fallback_key = chorus_block["key"].iloc[0].replace("-", "") if "key" in chorus_block.columns else "C"

    for i in range(len(chorus_block)):
        row = chorus_block.iloc[i]
        pitch = row["pitch"]
        current_chord = row.get("chord", "")
        if pd.notna(current_chord) and current_chord != "":
            active_chord = current_chord

        chord_to_parse = active_chord if active_chord != "" else fallback_key

        core_tones = chord_to_notes(chord_to_parse)

        if (pitch % 12) not in core_tones:
            notes_outside += 1

        return notes_outside

def chord_to_notes(input: str):
    """
    # ['Ab', 'MIN', 'MIN7', 'b5', 'b9', 'NAT11', 'NAT13']
    chord_to_notes(input)
    return [1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0]
    """
    chord = parse_chord(input)
    chord_num_list = []
    root_num = {
        "C":  0.0, "B#": 0.0,
        "C#": 1.0, "Db": 1.0,
        "D":  2.0,
        "D#": 3.0, "Eb": 3.0,
        "E":  4.0, "Fb": 4.0,
        "F":  5.0, "E#": 5.0,
        "F#": 6.0, "Gb": 6.0,
        "G":  7.0,
        "G#": 8.0, "Ab": 8.0,
        "A":  9.0,
        "A#": 10.0, "Bb": 10.0,
        "B":  11.0, "Cb": 11.0}
    root_val = root_num.get(chord[0], 0.0)

    shape_num = {"SUS": [5, 7], "MAJ": [4, 7], "MIN": [3, 7], "DIM": [3, 6], "AUG": [4, 8], "DOM": [4, 7]}
    ten7_num  = {"MAJ7": [11], "MIN7": [10], "NONE": []}
    alter_num = {"b5": [6], "#5": [8], "NONE": []}
    ten9_num  = {"b9": [1], "NAT9": [2], "#9": [3], "UNKNOWN": []}
    ten11_num = {"b11": [4], "NAT11": [5], "#11": [6], "UNKNOWN": []}
    ten13_num = {"b13": [8], "NAT13": [9], "#13": [10], "UNKNOWN": []}

    relative_intervals = [0] # 0 represents the root itself
    dictionaries = [shape_num, ten7_num, alter_num, ten9_num, ten11_num, ten13_num]

    for vocab_dict, token in zip(dictionaries, chord[1:]):
        relative_intervals.extend(vocab_dict.get(token, []))

    chord_num_list = []
    for interval in relative_intervals:
        absolute_pc = (root_val + interval) % 12
        chord_num_list.append(absolute_pc)

    return sorted(list(set(chord_num_list)))