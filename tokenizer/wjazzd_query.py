import pandas as pd
import sqlite3


def get_solo(melid: int, # all note+beat+meta
            conn: sqlite3.Connection) -> pd.DataFrame:
    query = f"""
    SELECT
    -- notes (melody)
    m.eventid,
    m.bar,                               -- join key
    m.pitch,                              -- pitch
    m.duration    AS note_duration,       -- duration
    m.beat,                               -- beat_pos (coarse) join key
    m.tatum,                              -- phrase segmentation: reset to 1 at phrase boundaries
    m.beatdur     AS beat_duration,       --
    m.onset       AS note_onset,          -- swing_ratio = (beat2_onset - beat1_onset) / (beat3_onset - beat1_onset)
    m.loud_max,                           -- vel
    m.loud_relpos,                        -- note_shape
    m.f0_mod,                             -- future expressive token (vibrato/fall-off/bend/slide)


    -- solo meatdata
    s.instrument,
    s.tempoclass,
    s.key,
    s.chorus_count,
    s.style,
    s.avgtempo

    FROM melody m

    JOIN beats b
        ON b.melid = m.melid
        AND b.bar = m.bar
        AND b.beat = m.beat

    JOIN solo_info s
        ON s.melid = m.melid
    WHERE m.melid = {melid}
    ORDER BY m.eventid
    """

    return pd.read_sql_query(query, conn)

def get_beats(melid, conn): #  all bars, all beats, including silent
    query = f"""SELECT
        b.bar,
        b.beat,
        b.onset      AS beat_onset,
        b.chord,
        b.form,
        b.chorus_id


    FROM beats b
    WHERE b.melid = {melid}
    ORDER BY b.onset"""

    df = pd.read_sql_query(query, conn)
    df= df.set_index(["bar", "beat"])
    return df

def get_phrases(melid: int, # phrase spans
                conn: sqlite3.Connection) -> pd.DataFrame:
    query = f"""
    SELECT start, end, value
    FROM sections
    WHERE melid = {melid}
    AND type = 'PHRASE'
    ORDER BY start
    """
    return  pd.read_sql_query(query, conn)

def get_valid_ids(conn: sqlite3.Connection) -> list[int]:
    query = """
    SELECT melid
    FROM solo_info
    WHERE instrument IN (
            'tp', 'as',
            'ts', 'ss')
    AND signature = '4/4'
    """
    return pd.read_sql_query(query, conn)["melid"].tolist()