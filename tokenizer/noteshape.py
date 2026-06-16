from vocab import NOTESHAPE_VOCAB

def assign_noteshape(loud_relpos):
    if   loud_relpos < 0.2: return NOTESHAPE_VOCAB["accent-release"]
    elif loud_relpos < 0.4: return NOTESHAPE_VOCAB["decay"]
    elif loud_relpos < 0.6: return NOTESHAPE_VOCAB["flat"]
    else:                   return NOTESHAPE_VOCAB["swell"]

def quantise_duration(note_duration, beat_duration):
    sixteenths = note_duration / beat_duration * 4
    return min(int(round(sixteenths)), 31)

def compute_beatpos(beat, note_onset, beat_onset, beat_duration):
    position_in_beat = (note_onset - beat_onset) / beat_duration
    position_in_beat = max(position_in_beat, 0.0)        # clamp negative laid-back
    subdivision      = min(int(position_in_beat * 4), 3) # 0–3 within beat
    return (beat - 1) * 4 + subdivision                  # 0–15 across bar
