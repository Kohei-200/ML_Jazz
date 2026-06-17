# Machine Learns to Speak Jazz
## Context
**Goal**: train a Transformer to generate new idiomatic Jazz improvizations for single-note instruments.\
**Motivation**: Standard music generation architectures like basic MIDI models treat music purely as absolute pitch and duration, which ails to capture the nuances in jazz language, such as tension, idiom, feel, etc.\
**Furthermore**, former state of the art architectures like MusicBERT provice excellet representaions but are inherently not suitable for autoregressive generation due to its bidirectional nature.

## Data
### Weimar Jazz Database ([WJazzD](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html))
As of June 2026, the data source for the training is only from Weimar Jazz DB, which is an amazing  database with over 400 jazz improvisation transcriptions and metadata available to the public to "further enhance and improve jazz and MIR research".
```bibtex
@Book{Pfleiderer:2017:BOOK,
  title     = {{I}nside the {J}azzomat - {N}ew {P}erspectives for {J}azz  {R}esearch},
  publisher = {Schott Campus},
  year      = {2017},
  editor    = {Pfleiderer, Martin and Frieler, Klaus and Abe{\ss}er, Jakob and Zaddach, Wolf-Georg and Burkhart, Benjamin},
}
```
## Representation
### Preprocessing ([tokenizer](https://github.com/Kohei-200/ML_Jazz/tree/main/tokenizer))
### Data criteria (currently)
- `solo_info.instrument`: trumpet(tp), alto saxophone(as), tenor saxophone(ts), soprano saxophone(ss)

- `solo_info.signature = '4/4'` only (tunes with changes inside 4/4 tune are also omitted)
### Module structure
```
tokeniser/
├── assembler.py        <= wire all modules → Jazz tuple token 
├── beat_aligner.py     <= swing dicretization
├── chord_parser.py     <= turn chord into 7 tuple representation
├── chorus_tagger.py    <= chorus_id + intensity → EARLY/MID/PEAK/LATE
├── noteshape.py        <= loud_relpos → flat/decay/swell/accent-release
├── tension_tagger.py   <= chord + pitch → capture tensions/alteration/outside notes
├── vocab.py            <= all token vocabularies, vocab_sizes
└── wjazzd_query.py     <= sqlite3 query to fetch data from db
```
### Token table
| Group | Token | 
|---|---|
| **Special** | PAD/ SOLO_END/ END/ BAR/ BEAT/ NOTE
|**Bar**| tempo/ section/ chorus/ instrument |
| **Beat** | beat/ root/ shape/ 7th/ alter/ 9th/ 11th/ 13th |
| **Note** | beatpos/  swing/ dur/ pitch/ interval/ octave/ tension/ phrase_role/ vel/ noteshape
### Processed Data
1D Tensor with Special tokens separating each group of tokens.
```
[BAR] [4 tuple]
    [BEAT] [8 tuple]
        [NOTE] [10 tuple]
        [NOTE] [...]
        [NOTE] [...]
    [BEAT] [...]
        [NOTE] [...]
        [NOTE] [...]
    ...    
[BAR] [...]
...
[SOLO_END]
```
For now, sequence that Bar header contains almost never changes during on a same performance, and therefore may be redundant. What is more, END token and SOLO_END token were introduced so that END separates each phrases, and SOLO_END imply the termination of a solo. However, END is not used for now because the phrase to phrase gap are implicitly represented as REST with many tokens dedicated for, which I think in a better way to understand this that silence is also one of the important factors that give colors to music.


## Architecture Decision log
### Autoregressive, Decoder-only architecture
