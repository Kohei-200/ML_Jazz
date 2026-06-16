def extract_root(symbol):
    roots = ["A#", "Ab", "A",
            "G#", "Gb", "G",
            "F#", "Fb", "F",
            "E#", "Eb", "E",
            "D#", "Db", "D",
            "C#", "Cb", "C",
            "B#", "Bb", "B", "NC"]
    for root in roots:
        if symbol.startswith(root):
            return root, symbol[len(root):]
    return "UNKNOWN", symbol

def extract_shape(symbol):
    shapes = [
        ("sus", "SUS"), ("sus2", "SUS"), ("sus4", "SUS"),
        ("maj", "MAJ"), ("Maj", "MAJ"), ("j",  "MAJ"), ("M",  "MAJ"),
        ("-",   "MIN"), ("m",   "MIN"),
        ("dim", "DIM"), ("o",   "DIM"),
        ("aug", "AUG"), ("+",   "AUG")]
    for marker, shape in shapes:
        if symbol.startswith(marker):
            if symbol == "j":
                return shape, symbol
            else:
                return shape, symbol[len(marker):]
    return "DOM", symbol          # no marker = dominant

def extract_7th(symbol):
    lis = [("j7", "MAJ7"),
        ("7", "MIN7")]
    for sig, name in lis:
        if symbol.startswith(sig):
            return name, symbol[len(sig):]
    return "NONE", symbol

def extract_alteration(symbol):
    lis = [("b5", "b5"),
        ("#5", "#5")]
    for sig, name in lis:
        if symbol.startswith(sig):
            return name, symbol[len(sig):]
    return "NONE", symbol

def extract_9th(symbol):
    lis = [
        ("9b", "b9"),
            ("9#", "#9"),
            ("b9", "b9"),
            ("#9", "#9"),
            ("9", "NAT9"),]
    for sig, name in lis:
        if symbol.startswith(sig):
            return name, symbol[len(sig):]
    return "UNKNOWN", symbol

def extract_11th(symbol):
    lis = [
        ("11b", "b11"),
        ("11#", "#11"),
        ("b11", "b11"),
        ("#11", "#11"),
        ("11", "NAT11")]
    for sig, name in lis:
        if symbol.startswith(sig):
            return name, symbol[len(sig):]
    return "UNKNOWN", symbol

def extract_13th(symbol):
    lis = [
        ("13b", "b13"),
        ("13#", "#13"),
        ("b13", "b13"),
        ("#13", "#13"),
        ("13", "NAT13")]
    for sig, name in lis:
        if symbol.startswith(sig):
            return name, symbol[len(sig):]
    return "UNKNOWN", symbol

def parse_chord(symbol: str) -> list[int]:
    """
    parse_chord("Abm7b59b1113")
    ['Ab', 'MIN', 'MIN7', 'b5', 'b9', 'NAT11', 'NAT13']
    """
    if not symbol:
        return ["UNKNOWN"] * 7
    root, symbol    = extract_root(symbol)
    shape, symbol   = extract_shape(symbol)
    ten7, symbol   = extract_7th(symbol) # hope spy chord is not lurking in
    alter, symbol   = extract_alteration(symbol)
    if symbol.startswith("alt"):
        return [root, shape, ten7, "b9", "#9", "#11", "b13"]
    ten9, symbol    = extract_9th(symbol)
    ten11, symbol   = extract_11th(symbol)
    ten13, symbol   = extract_13th(symbol)
    return [root, shape, ten7, alter, ten9, ten11, ten13]