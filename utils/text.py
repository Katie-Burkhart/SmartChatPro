import re

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def tokens_len(text, encoding) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, encoding, chunk_tokens=700, overlap_tokens=80):
    words = text.split()
    out, cur, cur_len = [], [], 0
    for w in words:
        n = tokens_len(w+" ", encoding)
        if cur_len + n > chunk_tokens:
            out.append(" ".join(cur))
            # overlap by last ~overlap_tokens worth of text
            back = []
            cur_text = " ".join(cur)
            # pull from end until reach ~overlap_tokens
            while back and tokens_len(" ".join(back), encoding) < overlap_tokens:
                back.insert(0, cur.pop())
            cur = back + [w]
            cur_len = tokens_len(" ".join(cur), encoding)
        else:
            cur.append(w); cur_len += n
    if cur: out.append(" ".join(cur))
    return out
