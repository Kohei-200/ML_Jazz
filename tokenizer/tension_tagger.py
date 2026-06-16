from vocab import SHAPE_VOCAB, TEN7_VOCAB, TEN9_VOCAB, TEN11_VOCAB, TEN13_VOCAB, TENSION_VOCAB, ALTER_VOCAB

def get_chord_tones(bar_header):
    shape  = bar_header[1]
    ten7   = bar_header[2]
    alter  = bar_header[3]

    base = {
        SHAPE_VOCAB["MAJ"]: {0, 4, 7},
        SHAPE_VOCAB["MIN"]: {0, 3, 7},
        SHAPE_VOCAB["DOM"]: {0, 4, 7},
        SHAPE_VOCAB["DIM"]: {0, 3, 6},
        SHAPE_VOCAB["AUG"]: {0, 4, 8},
        SHAPE_VOCAB["SUS"]: {0, 5, 7},
    }.get(shape, {0})

    if ten7 == TEN7_VOCAB["MIN7"]: base.add(10)
    if ten7 == TEN7_VOCAB["MAJ7"]: base.add(11)

    return base

def get_extensions(bar_header):
    ext = set()
    ten9_map  = {TEN9_VOCAB["NAT9"]:2,  TEN9_VOCAB["b9"]:1,   TEN9_VOCAB["#9"]:3}
    ten11_map = {TEN11_VOCAB["NAT11"]:5, TEN11_VOCAB["#11"]:6}
    ten13_map = {TEN13_VOCAB["NAT13"]:9, TEN13_VOCAB["b13"]:8}

    if bar_header[4] in ten9_map:  ext.add(ten9_map[bar_header[4]])
    if bar_header[5] in ten11_map: ext.add(ten11_map[bar_header[5]])
    if bar_header[6] in ten13_map: ext.add(ten13_map[bar_header[6]])
    return ext

def get_alterations(bar_header):
    alt_map = {
        ALTER_VOCAB["b5"]: 6,
        ALTER_VOCAB["#5"]: 8,
    }
    return {alt_map[bar_header[3]]} if bar_header[3] in alt_map else set()

def compute_tension_role(interval, bar_header, next_interval):
    if interval == 13:
        return TENSION_VOCAB["REST"]
    if interval == 12:
        return TENSION_VOCAB["UNKNOWN"]

    chord_tones = get_chord_tones(bar_header)
    extensions  = get_extensions(bar_header)
    alterations = get_alterations(bar_header)

    if interval in chord_tones:
        return TENSION_VOCAB["CHORD_TONE"]

    if interval in extensions:
        return TENSION_VOCAB["EXTENSION"]

    if interval in alterations:
        return TENSION_VOCAB["ALTERATION"]

    # APPROACH — semitone from a chord tone, resolves to chord tone on next note
    if next_interval not in (12, 13): # requires one-row look-ahead
        semitone_away = abs(interval - next_interval) == 1 or \
                        abs(interval - next_interval) == 11  # wrap-around (e.g. 11→0)
        if semitone_away and next_interval in chord_tones:
            return TENSION_VOCAB["APPROACH"]

    return TENSION_VOCAB["OUTSIDE"]

def compute_interval(pitch, bar_header):
    root_pitch = bar_header[0]
    if root_pitch == 13:
        return 12
    note_pitch = int(round(pitch)) % 12
    interval = (note_pitch - root_pitch) % 12
    return interval