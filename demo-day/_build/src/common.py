# %%
# Sözlük: özel token'lar + metin karakterleri + ses token'ları. Bütün notebook'larda aynı.
import unicodedata
PAD, BOS, SEP, EOS, MALE, FEMALE, ME = 0, 1, 2, 3, 4, 5, 6
CHARS = " abcçdefgğhıijklmnoöpqrsştuüvwxyzâîû0123456789.,?!'-:"
CHAR_BASE = 16
stoi = {c: CHAR_BASE + i for i, c in enumerate(CHARS)}
SR, FRAME_RATE = 24000, 12.5             # Mimi codec: 24 kHz ses, saniyede 12.5 kare
N_CB, CB_SIZE = 8, 2048                  # her karede 8 codebook, her biri 2048 kod
AUDIO_BASE = 128
VOCAB = AUDIO_BASE + N_CB * CB_SIZE      # 16512
TOK_PER_S = int(N_CB * FRAME_RATE)       # saniyede 100 ses token'ı
assert CHAR_BASE + len(CHARS) <= AUDIO_BASE

_MAP = str.maketrans({"’": "'", "‘": "'", "“": "", "”": "", '"': "", "–": "-", "—": "-", "…": "."})
def norm_text(s):
    s = unicodedata.normalize("NFC", s).replace("I", "ı").replace("İ", "i").lower().translate(_MAP)
    return " ".join("".join(c for c in s if c in stoi).split())
def encode_text(s):
    return [stoi[c] for c in norm_text(s)]
# Bir örnek: [BOS, ses etiketi, metin karakterleri..., SEP, kare0 cb0..cb7, kare1 cb0..cb7, ..., EOS]
