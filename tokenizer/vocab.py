
SPECIAL_TOKENS = {
    "PAD": 1000,
    "SOLO_END": 1001,
    "END": 1002,  # before padding
    "BAR": 1101,
    "BEAT": 1102,
    "NOTE": 1103
}

SPECIAL_TOKEN_IDX = {
    1000: 0,
    1001: 1,
    1002: 2, 
    1101: 3,
    1102: 4,
    1103: 5
}

######### 5 [BAR] [tempo|section|chorus|instrument]  ########
TEMPO_VOCAB = {
    "SLOW":0,
    "MEDIUM SLOW":1,
    "MEDIUM":2,
    "MEDIUM UP":3,
    "UP":4
}


SECTION_VOCAB = {
    "A1": 0,
    "A2": 0,
    "A3": 0,
    "B1": 1,
    "B2": 1,
    "C1": 2,
    "C2": 2,
    "C3": 2,
    "I1": 3,
    "I2": 3,
    "I3": 3,
    "I4": 3,
    "": 4,
}

CHORUS_VOCAB = {
    "EARLY":   0,
    "MID":     1,
    "PEAK":    2,
    "LATE":    3,
    "UNKNOWN": 4
}
INSTRUMENT_VOCAB = {
    "tp":0,
    "as":1,
    "ts":2,
    "ss":3,
    "UNKNOWN":4,
}

######### 9 [BEAT] [beat|root|shape|7th|alter|9|11|13]  ######

ROOT_VOCAB  = {"C":0,
                "C#":1, "Db":1,
                "D":2,
                "D#":3, "Eb":3,
                "E":4, "Fb":4,
                "F":5,
                "F#":6, "Gb":6,
                "G":7,
                "G#":8, "Ab":8,
                "A":9,
                "A#":10, "Bb":10,
                "B":11, "Cb": 11,
                "NC":12, "UNKNOWN":13}

SHAPE_VOCAB = {"MAJ":0, "MIN":1, "DOM":2, "DIM":3, "AUG":4, "SUS":5, "UNKNOWN":6}
TEN7_VOCAB  = {"NONE":0, "MIN7":1, "MAJ7":2, "UNKNOWN":3}
ALTER_VOCAB = {"NONE":0, "b5":1, "#5":2, "UNKNOWN":3, "b9": 4} # for alt b9 #9
TEN9_VOCAB  = {"NONE":0, "NAT9":1, "b9":2, "#9":3, "UNKNOWN":4}
TEN11_VOCAB = {"NONE":0, "NAT11":1, "b11":2, "#11":3, "UNKNOWN":4}
TEN13_VOCAB = {"NONE":0, "NAT13":1, "b13":2, "#13":3, "UNKNOWN":4}

########## 11 [Note] [beatpos|swing|dur|pitch|interval|octave|tension|phrase_role|vel|noteshape] """"""

BEATPOS_VOCAB = {i: i for i in range(16)}   # 0–15 direct

SWING_VOCAB = {
    "SW0":      0,   # 0.50–0.54  straight - > up, fast tempo
    "SW1":      1,   # 0.54–0.58  slight lilt
    "SW2":      2,   # 0.58–0.62  medium swing
    "SW3":      3,   # 0.62–0.67  full swing
    "SW4":      4,   # 0.67–0.72  heavy swing
    "SW5":      5,   # 0.72+      extreme
    "SW_UNKNOWN": 6
}
DURATION_VOCAB = {i: i for i in range(32)}  # 0–31 sixteenth units, cap at 2 bars

INTERVAL_VOCAB = {
    0:  "R",     # root
    1:  "b2",    # chromatic approach
    2:  "2",     # major 2nd / 9th
    3:  "b3",    # minor 3rd
    4:  "3",     # major 3rd
    5:  "4",     # perfect 4th / 11th
    6:  "b5",    # tritone
    7:  "5",     # perfect 5th
    8:  "#5",    # augmented 5th / b13
    9:  "6",     # major 6th / 13th
    10: "b7",    # minor 7th
    11: "7",     # major 7th
    12: "UNKNOWN",
    13: "REST"
}

TENSION_VOCAB = {
    "CHORD_TONE":  0,
    "EXTENSION":   1,
    "ALTERATION":  2,
    "APPROACH":    3,
    "OUTSIDE":     4,
    "REST":        5,
    "UNKNOWN":     6
}

PHRASEROLE_VOCAB = {
    "START": 0,
    "CONT":  1,
    "PEAK":  2,
    "LAND":  3,
    "REST":  4,
    "UNKNOWN": 5
}

VEL_VOCAB = {
    "pp":   0,
    "mp":   1,
    "mf":   2,
    "f":    3,
    "ff":   4,
    "REST": 5
}

NOTESHAPE_VOCAB = {
    "accent-release": 0,
    "decay":          1,
    "flat":           2,
    "swell":          3,
    "REST":           4
}

def get_vocab_sizes():
    """
    returns 22 tuple vocab size, special token size
    """
    tempo_size     = max(TEMPO_VOCAB.values()) + 1
    section_size   = max(SECTION_VOCAB.values()) + 1
    chorus_size    = max(CHORUS_VOCAB.values()) + 1
    instrument_size = max(INSTRUMENT_VOCAB.values()) + 1

    beat_size     = 5
    root_size     = max(ROOT_VOCAB.values()) + 1
    shape_size    = max(SHAPE_VOCAB.values()) + 1
    ten7_size     = max(TEN7_VOCAB.values()) + 1
    alt_size      = max(ALTER_VOCAB.values()) + 1
    ten9_size     = max(TEN9_VOCAB.values()) + 1
    ten11_size    = max(TEN11_VOCAB.values()) + 1
    ten13_size    = max(TEN13_VOCAB.values()) + 1

    beatpos_size       = max(BEATPOS_VOCAB.values()) + 1
    swing_size         = max(SWING_VOCAB.values()) + 1
    dur_size           = max(DURATION_VOCAB.values()) + 1
    pitch_size  = 128
    interval_size      = len(INTERVAL_VOCAB)
    octave_size = 11
    tension_size       = max(TENSION_VOCAB.values()) + 1
    phrase_role_size   = max(PHRASEROLE_VOCAB.values()) + 1
    vel_size           = max(VEL_VOCAB.values()) + 1
    noteshape_size     = max(NOTESHAPE_VOCAB.values()) + 1

    vocabsize_table = [tempo_size, section_size, chorus_size, instrument_size,
                        beat_size, root_size, shape_size, ten7_size, alt_size, ten9_size, ten11_size, ten13_size,
                        beatpos_size, swing_size, dur_size, pitch_size, interval_size,
                        octave_size, tension_size, phrase_role_size, vel_size, noteshape_size]
    
    special_tk_size = len(SPECIAL_TOKENS)
    return vocabsize_table, special_tk_size

BAR = SPECIAL_TOKENS["BAR"]
BEAT = SPECIAL_TOKENS["BEAT"]
NOTE = SPECIAL_TOKENS["NOTE"]
SOLO_END = SPECIAL_TOKENS["SOLO_END"]

NOTESHAPE_SLOT = 21      # last slot in a [NOTE] group -- the genuine branch point
DECISION_HEAD = 22       # index of the extra 4-way head in JazzLanguageModel.out_head

NEXT_GROUP_CLASS = {NOTE: 0, BEAT: 1, BAR: 2, SOLO_END: 3}

def get_prediction_events(tokens):
    events = []
    slot_idx = 0
    n = len(tokens)
    for i in range(n):
        token_val = tokens[i].item()
        if token_val >= 1000:
            if token_val == BAR:
                slot_idx = 0
            elif token_val == BEAT:
                slot_idx = 4
            elif token_val == NOTE:
                slot_idx = 12
            continue
        
        events.append((i -1, slot_idx, token_val))

        if slot_idx == NOTESHAPE_SLOT:
            next_val = tokens[i + 1].item()
            events.append((i, DECISION_HEAD, NEXT_GROUP_CLASS[next_val]))
        slot_idx += 1
    return events