# -*- coding: utf-8 -*-
# Vivollo – LLM vs Manuel Etiket Karşılaştırma (Teslim dosyaları otomatik + Groq backoff & dedup)

import os, json, time, re, datetime
import pandas as pd
import nltk
from dotenv import load_dotenv
import google.genai as genai
from groq import Groq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

# ------------ CONFIG ------------
JSON_PATH = "first_hundred_message.json"
MANUAL_CSV_IN = "manuel_etiket_last.csv"
MANUAL_CSV_OUT = "manual_labels.csv"         # teslim için kopya
OUT_JSON  = "llm_outputs.json"
OUT_CSV   = "etiket_karsilastirma.csv"
OUT_MD    = "accuracy_report.md"
OUT_PROMPTS = "prompt_versions.txt"

# Quota/Rate
N_MAX = 100
N_GEMINI = 50          # Gemini free-tier günlük 50
SLEEP_GEMINI = 4.2     # dk'da 15 istek sınırı için

# Groq ayarları
GROQ_MODEL = "llama-3.1-8b-instant"  # token ve hız için ideal; istersen "llama-3.3-70b-versatile"
GROQ_MAX_RETRIES = 5                 # 429 backoff denemeleri
# Dedup: tekrar eden metinler tek çağrı

# ------------ Allowed Categories ------------
ALLOWED_CATEGORIES = [
    # --Düğün Mekanları
    "Düğün Salonları","Kır Düğünü Mekanları","Davet Alanları","Otelde Düğün","Tarihi Düğün Mekanları",
    "Havuz Başı Düğün Mekanları","Teknede Düğün ve Davet","Sosyal Tesisler","Luxury Wedding",
    "Nikah Salonu ve Evlendirme Dairesi","Sünnet Düğünü Mekanları","Tüm Düğün Mekanları",
    # --Diğer Davet Mekanları
    "Kına Gecesi Mekanları","Gelin Hamamı Mekanları","Söz, İsteme, Nişan Mekanları, Davet Evleri",
    "Nikah Sonrası Yemek","Mezuniyet ve Balo Mekanları","After Party İçin Eğlence Mekanları",
    "Doğum Günü Party Evleri ve Baby Shower Mekanları","Evlilik Teklifi Mekanları",
    "Konferans ve Toplantı Salonları","Tüm Diğer Davet Mekanları",
    # --Düğün Firmaları
    "Dış Çekim ve Düğün Fotoğrafçıları","Gelin Saçı ve Makyajı","Gelinlik ve Moda Evleri","Gelin Arabası",
    "Düğün Orkestrası, DJ ve Müzik Grupları","Düğün Dans Kursu","Düğün Davetiyesi","Nikah Şekeri ve Hediyelik",
    "Düğün Yemeği İçin Catering Firmaları","Düğün Pastası Firmaları","Gelin Çiçeği","Damatlık Modelleri",
    "Gelin Ayakkabısı ve Aksesuarları","Nişanlık ve Abiye Modelleri","Alyans ve Tektaş Yüzük","Tüm Düğün Firmaları",
    # --Organizasyon Firmaları
    "Kına Gecesi Organizasyonu","Nişan Organizasyonu Firmaları","Düğün Organizasyonu Firmaları",
    "Doğum Günü ve Baby Shower Organizasyonu Firmaları","Mezuniyet ve Balo Organizasyonu Firmaları",
    "Evlilik Teklifi Organizasyonu Firmaları","Sünnet Organizasyonu Firmaları","Tüm Organizasyon Firmaları",
    # --Balayı
    "Balayı Otelleri","Balayı Evleri","Balayı Gemi Turları","Tüm Balayı",
    # Güvenli sepet
    "Diğer"
]

# keyword -> kategori (rule-based destek)
CATEGORY_KEYWORDS = [
    # Düğün Mekanları
    (["düğün salon","balo salon"], "Düğün Salonları"),
    (["kır düğün","kır bahçe","çim alan"], "Kır Düğünü Mekanları"),
    (["davet alan","etkinlik alan","event alan"], "Davet Alanları"),
    (["otel","otelde düğün","hotel wedding"], "Otelde Düğün"),
    (["tarihi mekan","konak","yalı","han","köşk"], "Tarihi Düğün Mekanları"),
    (["havuz başı","poolside"], "Havuz Başı Düğün Mekanları"),
    (["tekne","yatta düğün","boğaz turu"], "Teknede Düğün ve Davet"),
    (["sosyal tesis"], "Sosyal Tesisler"),
    (["luxury wedding","lüks düğün"], "Luxury Wedding"),
    (["nikah salon","evlendirme dairesi"], "Nikah Salonu ve Evlendirme Dairesi"),
    (["sünnet düğün"], "Sünnet Düğünü Mekanları"),
    # Diğer Davet Mekanları
    (["kına mekanı","kına gecesi"], "Kına Gecesi Mekanları"),
    (["gelin hamam"], "Gelin Hamamı Mekanları"),
    (["söz","isteme","nişan mekanı","davet evi"], "Söz, İsteme, Nişan Mekanları, Davet Evleri"),
    (["nikah sonrası yemek"], "Nikah Sonrası Yemek"),
    (["mezuniyet","balo"], "Mezuniyet ve Balo Mekanları"),
    (["after party","after-party","afterparty"], "After Party İçin Eğlence Mekanları"),
    (["doğum günü","baby shower","party evi"], "Doğum Günü Party Evleri ve Baby Shower Mekanları"),
    (["evlilik teklifi"], "Evlilik Teklifi Mekanları"),
    (["konferans","toplantı salon"], "Konferans ve Toplantı Salonları"),
    # Düğün Firmaları
    (["dış çekim","fotoğrafçı","video"], "Dış Çekim ve Düğün Fotoğrafçıları"),
    (["gelin saçı","gelin makyaj"], "Gelin Saçı ve Makyajı"),
    (["gelinlik","modaevi","moda evi"], "Gelinlik ve Moda Evleri"),
    (["gelin arabası","gelin arab"], "Gelin Arabası"),
    (["orkestra","dj","müzik grup"], "Düğün Orkestrası, DJ ve Müzik Grupları"),
    (["dans kurs","ilk dans"], "Düğün Dans Kursu"),
    (["davetiy"], "Düğün Davetiyesi"),
    (["nikah şekeri","hediyelik"], "Nikah Şekeri ve Hediyelik"),
    (["catering","düğün yemeği"], "Düğün Yemeği İçin Catering Firmaları"),
    (["düğün pastası","butik pasta"], "Düğün Pastası Firmaları"),
    (["gelin çiçeği","buketi"], "Gelin Çiçeği"),
    (["damatlık"], "Damatlık Modelleri"),
    (["gelin ayakkabı","aksesuar"], "Gelin Ayakkabısı ve Aksesuarları"),
    (["nişanlık","abiye"], "Nişanlık ve Abiye Modelleri"),
    (["alyans","tektaş"], "Alyans ve Tektaş Yüzük"),
    # Organizasyon Firmaları
    (["kına organizasyon"], "Kına Gecesi Organizasyonu"),
    (["nişan organizasyon"], "Nişan Organizasyonu Firmaları"),
    (["düğün organizasyon"], "Düğün Organizasyonu Firmaları"),
    ((["doğum günü organizasyon","baby shower organizasyon"]), "Doğum Günü ve Baby Shower Organizasyonu Firmaları"),
    (["mezuniyet organizasyon","balo organizasyon"], "Mezuniyet ve Balo Organizasyonu Firmaları"),
    (["evlilik teklifi organizasyon"], "Evlilik Teklifi Organizasyonu Firmaları"),
    (["sünnet organizasyon"], "Sünnet Organizasyonu Firmaları"),
    # Balayı
    (["balayı otel"], "Balayı Otelleri"),
    (["balayı evi","villa"], "Balayı Evleri"),
    (["balayı gemi","cruise"], "Balayı Gemi Turları"),
]

# ------------ Setup ------------
load_dotenv()
nltk.download('vader_lexicon', quiet=True)

# API key'leri .env'den al (kodda key hardcode etme)
client_gemini = genai.Client(api_key='API KEY')
client_groq   = Groq(api_key='API KEY')
_vader = SentimentIntensityAnalyzer()

# ------------ IO ------------
def read_json_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    frames = []
    for conv in raw:
        msgs = pd.json_normalize(conv.get("messages", []))
        msgs["conversation_id"] = conv.get("conversation_id")
        frames.append(msgs)
    df = pd.concat(frames, ignore_index=True)
    if "content.text" not in df.columns and "content" in df.columns:
        df["content.text"] = df["content"].apply(
            lambda x: x.get("text","") if isinstance(x, dict) else (x if isinstance(x,str) else "")
        )
    df["content.text"] = df["content.text"].fillna("")
    return df

def read_manual_labels(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except Exception:
            sep = ";"
    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        quotechar='"',
        escapechar="\\",
        encoding="utf-8-sig",
        dtype=str,
        on_bad_lines="skip",
    ).fillna("")
    df.columns = [c.strip() for c in df.columns]
    if "content.text" not in df.columns:
        cand = [c for c in df.columns if "content" in c.lower()]
        if not cand:
            raise SystemExit("Manuel dosyada içerik sütunu yok (content.text bekleniyor).")
        df = df.rename(columns={cand[0]: "content.text"})
    return df

# ------------ Rule-based helpers ------------
def is_answered(row, df):
    next_idx = df.index[df.index > row.name].min()
    if pd.isna(next_idx): return "Hayır"
    return "Evet" if df.loc[next_idx, "sender_id"] == "bf17272dc3f0" else "Hayır"

def classify_intent(text):
    if not isinstance(text, str): return "Diğer"
    t = text.lower()
    if "mekan" in t or "yer" in t: return "Mekan Arayışı"
    if "fiyat" in t or "bütçe" in t: return "Bütçe Sorusu"
    if any(k in t for k in ["ürün","abiye","damatlık","gelinlik","fotoğrafçı","nişanlık"]): return "Ürün Arayışı"
    return "Diğer"

def rb_sentiment(text):
    if not isinstance(text, str): return "Nötr"
    c = _vader.polarity_scores(text)["compound"]
    return "Pozitif" if c >= 0.05 else ("Negatif" if c <= -0.05 else "Nötr")

def classify_category_rule(text):
    if not isinstance(text, str) or not text.strip():
        return "Diğer"
    t = text.lower()
    for kws, cat in CATEGORY_KEYWORDS:
        if any(k in t for k in kws):
            return cat
    if any(k in t for k in ["mekan","düğün","nikah","salon","organizasyon","davet"]):
        return "Tüm Düğün Mekanları"
    return "Diğer"

def normalize_topic_to_allowed(topic_in: str) -> str:
    if not isinstance(topic_in, str) or not topic_in.strip():
        return "Diğer"
    t = topic_in.strip().lower()
    for cat in ALLOWED_CATEGORIES:
        if t == cat.lower():
            return cat
    for cat in ALLOWED_CATEGORIES:
        base = cat.lower().replace("mekanları","").replace("firmaları","").strip()
        if base and base in t:
            return cat
    aliases = {
        "kına gecesi": "Kına Gecesi Mekanları",
        "kına organizasyonu": "Kına Gecesi Organizasyonu",
        "fotoğrafçı": "Dış Çekim ve Düğün Fotoğrafçıları",
        "catering": "Düğün Yemeği İçin Catering Firmaları",
        "dj": "Düğün Orkestrası, DJ ve Müzik Grupları",
        "dans kursu": "Düğün Dans Kursu",
        "gelinlik": "Gelinlik ve Moda Evleri",
        "damatlık": "Damatlık Modelleri",
        "tektaş": "Alyans ve Tektaş Yüzük",
        "balayı": "Tüm Balayı",
        "otel": "Otelde Düğün",
        "havuz başı": "Havuz Başı Düğün Mekanları",
        "tekne": "Teknede Düğün ve Davet",
        "sosyal tesis": "Sosyal Tesisler",
    }
    for k, v in aliases.items():
        if k in t:
            return v
    return "Diğer"

# ------------ JSON parse helpers ------------
def parse_first_json_block(text: str):
    if not text: return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m: return None
    js = m.group(0)
    try:
        return json.loads(js)
    except Exception:
        js2 = js.replace("`","")
        js2 = re.sub(r",\s*}", "}", js2)
        js2 = re.sub(r",\s*]", "]", js2)
        try:
            return json.loads(js2)
        except Exception:
            return None

def fallback_from_text(text: str):
    low = (text or "").lower()
    sent = "Nötr"
    if any(w in low for w in ["harika","güzel","teşekkür","pozitif","memnun"]): sent = "Pozitif"
    if any(w in low for w in ["şikayet","kötü","negatif","beğenmed"]): sent = "Negatif"
    topic = classify_category_rule(text or "")
    return {"sentiment": sent, "topic": topic, "bot_answered": "Hayır"}

# ------------ Prompts ------------
ALLOWED_CATS_PROMPT = "\n".join(f"- {c}" for c in ALLOWED_CATEGORIES if c != "Diğer") + "\n- Diğer"

PROMPT_TEMPLATE = (
    "Sen bir semantik etiketleme asistanısın.\n"
    "Sadece JSON döndür ve şu anahtarları kullan:\n"
    '{"sentiment":"Pozitif|Nötr|Negatif","topic":"Aşağıdaki listeden birebir seç veya Diğer","bot_answered":"Evet|Hayır"}\n'
    "KATEGORİ LİSTESİ:\n" + ALLOWED_CATS_PROMPT + "\n"
    'METİN: "{message_text}"'
)

def build_prompt(message_text: str) -> str:
    return PROMPT_TEMPLATE.replace("{message_text}", message_text.replace('"','\\"'))

# --- Kısa prompt (Groq için token dostu) ---
def build_prompt_min(message_text: str) -> str:
    return (
        'Sadece şu JSON\'u döndür: '
        '{"sentiment":"Pozitif|Nötr|Negatif","topic":"<listedekilerden biri veya Diğer>","bot_answered":"Evet|Hayır"}. '
        'Metin: "' + message_text.replace('"', '\\"') + '"'
    )

# ------------ LLM callers ------------
def call_gemini(message_text: str):
    if not isinstance(message_text, str) or not message_text.strip():
        return {"gemini_sentiment":"Nötr","gemini_topic":"Diğer","gemini_bot_answered":"Hayır"}
    try:
        prompt = build_prompt(message_text)
        resp = client_gemini.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        txt = getattr(resp, "text", None) or str(resp)
        parsed = parse_first_json_block(txt) or fallback_from_text(message_text)
        topic_norm = normalize_topic_to_allowed(parsed.get("topic","Diğer"))
        return {
            "gemini_sentiment": parsed.get("sentiment","Nötr"),
            "gemini_topic": topic_norm,
            "gemini_bot_answered": parsed.get("bot_answered", parsed.get("botAnswered","Hayır"))
        }
    except Exception as e:
        print("Gemini hata:", e)
        return {"gemini_sentiment":"Nötr","gemini_topic":"Diğer","gemini_bot_answered":"Hayır"}

def _parse_retry_seconds(msg: str) -> float:
    # "try again in 2m28.297s" kalıbından saniyeyi çek
    m = re.search(r"in\s+((\d+)m)?(\d+(?:\.\d+)?)s", msg)
    if not m:
        return 10.0
    minutes = float(m.group(2) or 0)
    seconds = float(m.group(3) or 0)
    return minutes*60 + seconds

def call_groq_with_backoff(message_text: str, model=GROQ_MODEL, max_retries=GROQ_MAX_RETRIES):
    if not isinstance(message_text, str) or not message_text.strip():
        return {"groq_sentiment":"Nötr","groq_topic":"Diğer","groq_bot_answered":"Hayır"}
    prompt = build_prompt_min(message_text)
    last_err = None
    for attempt in range(max_retries):
        try:
            r = client_groq.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            content = r.choices[0].message.content
            parsed = parse_first_json_block(content) or fallback_from_text(message_text)
            topic_norm = normalize_topic_to_allowed(parsed.get("topic","Diğer"))
            return {
                "groq_sentiment": parsed.get("sentiment","Nötr"),
                "groq_topic": topic_norm,
                "groq_bot_answered": parsed.get("bot_answered", parsed.get("botAnswered","Hayır"))
            }
        except Exception as e:
            err_str = str(e)
            last_err = e
            if "rate_limit_exceeded" in err_str or "429" in err_str:
                wait_s = _parse_retry_seconds(err_str)
                wait_s = min(max(wait_s, 5.0), 90.0)
                print(f"[Groq 429] deneme {attempt+1}/{max_retries} – {wait_s:.1f}s bekliyorum…")
                time.sleep(wait_s)
                continue
            else:
                print("Groq hata:", e)
                break
    print("Groq hata (final):", last_err)
    return {"groq_sentiment":"Nötr","groq_topic":"Diğer","groq_bot_answered":"Hayır"}

# ------------ Accuracy ------------
def calc_accuracy(df, pred_col, gt_col):
    p = df[pred_col].astype(str).str.strip().str.lower()
    g = df[gt_col].astype(str).str.strip().str.lower()
    mask = g != ""
    total = int(mask.sum())
    correct = int((p[mask] == g[mask]).sum())
    acc = round((correct/total)*100, 2) if total>0 else 0.0
    return correct, total, acc

def print_report(model_name, df):
    man_sent = "manuel_sentiment"
    man_topic = "manuel_category"
    man_bot   = "manuel_is_answered" if "manuel_is_answered" in df.columns else ("is_answered" if "is_answered" in df.columns else None)
    rows = []
    rows.append(("Sentiment", *calc_accuracy(df, f"{model_name}_sentiment", man_sent)))
    rows.append(("Konu", *calc_accuracy(df, f"{model_name}_topic",     man_topic)))
    if man_bot:
        rows.append(("Yanıtladı mı?", *calc_accuracy(df, f"{model_name}_bot_answered", man_bot)))
    print(f"\n{model_name.upper()} Doğruluk Raporu")
    print("Başlık         Doğru Sayısı   Toplam   Doğruluk (%)")
    for name, c, t, a in rows:
        print(f"{name:<14} {c:<13} {t:<7} %{a}")

def build_accuracy_md(merged: pd.DataFrame) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def row(model):
        s_c,s_t,s_a = calc_accuracy(merged, f"{model}_sentiment", "manuel_sentiment")
        t_c,t_t,t_a = calc_accuracy(merged, f"{model}_topic",     "manuel_category")
        b_col = "manuel_is_answered" if "manuel_is_answered" in merged.columns else "is_answered"
        b_c,b_t,b_a = calc_accuracy(merged, f"{model}_bot_answered", b_col)
        return s_c,s_t,s_a,t_c,t_t,t_a,b_c,b_t,b_a

    gs_c,gs_t,gs_a, gt_c,gt_t,gt_a, gb_c,gb_t,gb_a = row("gemini")
    rs_c,rs_t,rs_a, rt_c,rt_t,rt_a, rb_c,rb_t,rb_a = row("groq")

    lines = []
    lines.append(f"# Doğruluk Raporu\n\nTarih: **{ts}**\n")
    lines.append("## Gemini")
    lines.append("Başlık | Doğru Sayısı | Toplam | Doğruluk (%)")
    lines.append("---|---:|---:|---:")
    lines.append(f"Sentiment | {gs_c} | {gs_t} | %{gs_a}")
    lines.append(f"Konu | {gt_c} | {gt_t} | %{gt_a}")
    lines.append(f"Yanıtladı mı? | {gb_c} | {gb_t} | %{gb_a}\n")
    lines.append("## Groq")
    lines.append("Başlık | Doğru Sayısı | Toplam | Doğruluk (%)")
    lines.append("---|---:|---:|---:")
    lines.append(f"Sentiment | {rs_c} | {rs_t} | %{rs_a}")
    lines.append(f"Konu | {rt_c} | {rt_t} | %{rt_a}")
    lines.append(f"Yanıtladı mı? | {rb_c} | {rb_t} | %{rb_a}\n")

    # Hatalı örneklerden küçük vitrin (şeffaflık)
    def mismatch_samples(model, col_pred, col_gt, title, n=5):
        df = merged[[ "content.text", col_pred, col_gt ]].copy()
        df[col_pred] = df[col_pred].astype(str)
        df[col_gt]   = df[col_gt].astype(str)
        bad = df[df[col_pred].str.strip().str.lower() != df[col_gt].str.strip().str.lower()]
        bad = bad.head(n)
        if bad.empty:
            lines.append(f"**{model} – {title}**: Fark yok ✅\n")
            return
        lines.append(f"**{model} – {title} (örnek {len(bad)})**")
        for _,r in bad.iterrows():
            msg = (r["content.text"][:140] + "…") if len(r["content.text"])>140 else r["content.text"]
            lines.append(f"- Metin: _{msg}_\n  - Tahmin: **{r[col_pred]}**\n  - Manuel: **{r[col_gt]}**")
        lines.append("")

    mismatch_samples("Gemini","gemini_sentiment","manuel_sentiment","Sentiment")
    mismatch_samples("Gemini","gemini_topic","manuel_category","Konu")
    tgt_bot_col = "manuel_is_answered" if "manuel_is_answered" in merged.columns else "is_answered"
    mismatch_samples("Gemini","gemini_bot_answered", tgt_bot_col,"Yanıtladı mı?")

    mismatch_samples("Groq","groq_sentiment","manuel_sentiment","Sentiment")
    mismatch_samples("Groq","groq_topic","manuel_category","Konu")
    mismatch_samples("Groq","groq_bot_answered", tgt_bot_col,"Yanıtladı mı?")

    return "\n".join(lines)

def write_prompts_file():
    with open(OUT_PROMPTS, "w", encoding="utf-8") as f:
        f.write("# Prompt Versiyonları\n\n")
        f.write("## Ortak Şablon\n")
        f.write(PROMPT_TEMPLATE.replace("{message_text}", "<mesaj_metni>") + "\n\n")
        f.write("## Notlar\n")
        f.write("- `topic` alanı **yalnızca** aşağıdaki listedeki kategorilerden biri veya `Diğer` olmalıdır.\n")
        f.write("- LLM JSON harici bir şey döndürürse kodumuz JSON'u metinden ayıklamayı dener; yine de başarısız olursa fallback çalışır.\n")

# ------------ Main ------------
if __name__ == "__main__":
    # 1) Veriyi oku
    df = read_json_lines(JSON_PATH)
    if df is None: raise SystemExit("JSON okunamadı.")
    df = df.iloc[:min(N_MAX, len(df))].copy().reset_index(drop=True)

    # 2) Rule-based (opsiyonel)
    df["is_answered"] = df.apply(lambda r: is_answered(r, df), axis=1)
    df["intent_rb"]   = df["content.text"].apply(classify_intent)
    df["category_rb"] = df["content.text"].apply(lambda t: normalize_topic_to_allowed(classify_category_rule(t)))
    df["sentiment_rb"]= df["content.text"].apply(rb_sentiment)

    # 3) LLM çağrıları
    texts = df["content.text"].astype(str).fillna("").tolist()

    # --- GEMINI (ilk N_GEMINI; quota dostu) ---
    gemini_rows = []
    for i, txt in enumerate(texts):
        if i < N_GEMINI:
            gemini_rows.append(call_gemini(txt))
            time.sleep(SLEEP_GEMINI)
        else:
            gemini_rows.append({"gemini_sentiment":"Nötr","gemini_topic":"Diğer","gemini_bot_answered":"Hayır"})

    # --- GROQ (dedup + backoff) ---
    groq_cache = {}
    unique_texts = list(dict.fromkeys(texts))  # sıra korunarak benzersizler
    for ut in unique_texts:
        res = call_groq_with_backoff(ut, model=GROQ_MODEL, max_retries=GROQ_MAX_RETRIES)
        groq_cache[ut] = res
    groq_rows = [groq_cache.get(t, {"groq_sentiment":"Nötr","groq_topic":"Diğer","groq_bot_answered":"Hayır"}) for t in texts]

    gemini_df = pd.DataFrame(gemini_rows)
    groq_df   = pd.DataFrame(groq_rows)
    out = pd.concat([df, gemini_df, groq_df], axis=1)

    # 4) Ham LLM çıktısı
    out.to_json(OUT_JSON, orient="records", force_ascii=False, indent=2)

    # 5) Manuel etiketleri oku, kopyasını teslim formatında kaydet, merge et
    if not os.path.exists(MANUAL_CSV_IN):
        raise SystemExit(f"Manuel etiket dosyası yok: {MANUAL_CSV_IN}")
    man = read_manual_labels(MANUAL_CSV_IN)
    try:
        man.to_csv(MANUAL_CSV_OUT, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    merged = pd.merge(out, man, on="content.text", how="inner")

    # 6) Konsola rapor
    print_report("gemini", merged)
    print_report("groq", merged)

    # 7) CSV ve MD rapor
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    md_text = build_accuracy_md(merged)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)

    # 8) Prompt versiyon dosyası
    write_prompts_file()

    print(f"\n✅ Tamamlandı: {OUT_JSON}, {MANUAL_CSV_OUT}, {OUT_CSV}, {OUT_MD}, {OUT_PROMPTS}")
