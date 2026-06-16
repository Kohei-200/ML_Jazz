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

INSTRUMENT_VOCAB = {
    "tp":0,
    "as":1,
    "ts":2,
    "ss":3,
    "UNKNOWN":4,
                    }
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

BEATPOS_VOCAB = {i: i for i in range(16)}   # 0–15 direct

DURATION_VOCAB = {i: i for i in range(32)}  # 0–31 sixteenth units, cap at 2 bars

SWING_VOCAB = {
    "SW0":      0,   # 0.50–0.54  straight - > up, fast tempo
    "SW1":      1,   # 0.54–0.58  slight lilt
    "SW2":      2,   # 0.58–0.62  medium swing
    "SW3":      3,   # 0.62–0.67  full swing
    "SW4":      4,   # 0.67–0.72  heavy swing
    "SW5":      5,   # 0.72+      extreme
    "SW_UNKNOWN": 6
}

INTERVAL_VOCAB = {
    0:  "R",     # root
    1:  "b2",    # rare in jazz, chromatic approach
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
    12: "UNKNOWN"
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

SPECIAL_TOKENS = {
    "PAD": 1000,
    "SOLO_END": 1001,
    "END": 1002  # before padding
}