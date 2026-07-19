import re
import functools
import pandas as pd

DATA_DIR = "candidate_public/candidate_data/"

def load_data(data_dir=DATA_DIR):
    art = pd.read_feather(data_dir + "articles.f")
    cal = pd.read_feather(data_dir + "calibration.f")
    test = pd.read_feather(data_dir + "test.f")
    return art, cal, test


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)

_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&laquo;": "«", "&raquo;": "»",
    "&mdash;": "—", "&ndash;": "–", "&rsquo;": "'", "&hellip;": "…",
}


def clean_html(html: str) -> str:
    if not isinstance(html, str):
        return ""
    txt = _STYLE_RE.sub(" ", html)
    txt = _TAG_RE.sub(" ", txt)
    for k, v in _HTML_ENTITIES.items():
        txt = txt.replace(k, v)
    txt = re.sub(r"&#\d+;", " ", txt)
    txt = _WS_RE.sub(" ", txt)
    return txt.strip()


_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)

RU_STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда
лучше чуть том нельзя такой им более всегда конечно всю между это мне мной вами
нам наш ваш свой который эта эти этих
здравствуйте добрый день вечер утро пожалуйста подскажите скажите спасибо
""".split())

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


@functools.lru_cache(maxsize=200000)
def _lemma(word: str) -> str:
    return _get_morph().parse(word)[0].normal_form


def tokenize(text: str, lemmatize=True, drop_stop=True, min_len=2):
    text = text.lower().replace("ё", "е")
    toks = _TOKEN_RE.findall(text)
    out = []
    for t in toks:
        if len(t) < min_len:
            continue
        if lemmatize:
            t = _lemma(t)
        if drop_stop and t in RU_STOP:
            continue
        out.append(t)
    return out


def apk(gt_set, ranked, k=10):
    if not gt_set:
        return 0.0
    hits = 0
    score = 0.0
    for i, p in enumerate(ranked[:k], start=1):
        if p in gt_set:
            hits += 1
            score += hits / i
    return score / min(len(gt_set), k)


def mapk(gt_lists, ranked_lists, k=10):
    return sum(apk(g, r, k) for g, r in zip(gt_lists, ranked_lists)) / len(gt_lists)


def parse_gt(series):
    return [set(int(x) for x in g.split()) for g in series]
