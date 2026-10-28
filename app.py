import difflib
import glob
import os
import pickle
import zlib

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_algorithm.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
CLASSES = [int(c) for c in model.classes_]

# ------------------------------------------------------------------
# Output classes: 0 = Bull Market, 1 = Recession, 2 = Sideways Market
# (change the order here if your training labels were encoded differently)
# ------------------------------------------------------------------
OUTCOMES = [
    {"label": "Bull Market", "key": "bull",
     "note": "Low layoffs, healthy hiring and growth: conditions look strong."},
    {"label": "Recession", "key": "bear",
     "note": "Heavy layoffs and weak signals: conditions point to a downturn."},
    {"label": "Sideways Market", "key": "flat",
     "note": "Mixed signals: neither strong growth nor a sharp decline."},
]

# ------------------------------------------------------------------
# Text fields the user types into (all start empty) and the number of
# categories the model was trained on.
# ------------------------------------------------------------------
CATEGORICAL = {
    "company_name": 20, "industry": 7, "country": 6, "company_size": 4,
    "month": 12, "reason_for_layoffs": 5, "hiring_trend": 4, "top_hiring_role": 8,
}
PLACEHOLDERS = {
    "company_name": "e.g. Microsoft", "industry": "e.g. Technology", "country": "e.g. United States",
    "company_size": "e.g. Large", "month": "e.g. March", "reason_for_layoffs": "e.g. AI automation",
    "hiring_trend": "e.g. Growing", "top_hiring_role": "e.g. Data Scientist",
}

# OPTIONAL: type your real names here, in LabelEncoder order (sorted A-Z).
# Example: "company_name": ["Amazon", "Google", "Microsoft", ...]
# Leave empty to load them automatically from a dataset CSV next to app.py.
MANUAL_LABELS = {}


def build_labels():
    """name -> model code. Names are sorted like sklearn's LabelEncoder,
    so the position in the list is the number the model was trained on."""
    labels = {k: [] for k in CATEGORICAL}
    for k, v in MANUAL_LABELS.items():
        if k in labels:
            labels[k] = [str(x) for x in v]
    for path in glob.glob(os.path.join(BASE_DIR, "*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not all(c in df.columns for c in CATEGORICAL):
            continue
        for col, n in CATEGORICAL.items():
            if labels[col]:
                continue
            vals = sorted(df[col].dropna().astype(str).unique())
            if len(vals) == n:
                labels[col] = vals
            else:
                print(f"[labels] {col}: CSV has {len(vals)} values, model expects {n}. Skipped.")
        print(f"[labels] Loaded real names from {os.path.basename(path)}")
        break
    if not any(labels.values()):
        print("[labels] No names loaded. Add your dataset CSV next to app.py so names like Microsoft are recognised.")
    return labels


LABELS = build_labels()
TEXT_FIELDS = list(CATEGORICAL)


def encode(field, raw):
    """Return (code, approximate). Exact when the name is known; otherwise a
    stable, in-range fallback code is used and flagged as approximate."""
    text = str(raw).strip()
    names = LABELS.get(field, [])
    lowered = [n.lower() for n in names]
    if text.lower() in lowered:
        return lowered.index(text.lower()), False
    if text.isdigit() and int(text) < CATEGORICAL[field]:
        return int(text), False          # the encoded number itself
    # Unknown name: deterministic code inside the valid range (same name -> same code)
    return zlib.crc32(text.lower().encode("utf-8")) % CATEGORICAL[field], True


# Numeric inputs: (min, max, step, default)
NUMERIC = {
    "layoffs_count":          (0, 15000, 50, 2400),
    "layoff_percentage":      (0, 30, 0.1, 9),
    "ai_automation_impact":   (0, 18, 0.1, 6),
    "ai_replacement_risk":    (0, 10, 0.1, 6),
    "open_roles":             (0, 10000, 50, 2300),
    "remote_jobs_percentage": (0, 100, 1, 48),
    "stock_growth_percent":   (-50, 100, 0.5, 24),
    "revenue_growth_percent": (-30, 65, 0.5, 15),
    "salary_budget_change":   (-25, 40, 0.5, 7),
    "ai_adoption_level":      (0, 10, 0.1, 6),
    "employee_sentiment":     (0, 10, 0.1, 7),
    "job_security_score":     (0, 10, 0.1, 6),
}

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Outlook Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  color-scheme: light;
  --bg:#f2f6ff; --surface:#ffffff; --ink:#0f2147; --muted:#5d6b8a; --line:#dde5f5; --field:#f7f9ff;
  --brand:#3b5bff; --brand-dark:#2540d6;
  --bull:#12a36b; --bull-bg:#e4f8ef;
  --bear:#e5484d; --bear-bg:#fdeaea;
  --flat:#f2a20c; --flat-bg:#fff4dc;
  --radius:16px;
  --display:"Bricolage Grotesque","DM Sans",system-ui,sans-serif;
  --body:"DM Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);font-family:var(--body);font-size:16px;line-height:1.5;-webkit-font-smoothing:antialiased;
  background:
    radial-gradient(900px 420px at 8% -8%, #dfe7ff 0%, transparent 70%),
    radial-gradient(700px 380px at 100% 0%, #d9f5ea 0%, transparent 70%),
    var(--bg);
  min-height:100vh}
.wrap{max-width:1180px;margin:0 auto;padding:44px 24px 72px}

header{max-width:660px;margin-bottom:34px}
h1{font-family:var(--display);font-weight:700;font-size:clamp(2.1rem,4.6vw,3.1rem);line-height:1.06;letter-spacing:-.025em;margin:0 0 12px}
header p{margin:0;color:var(--muted);font-size:1.05rem}

.layout{display:grid;grid-template-columns:minmax(0,1fr) 370px;gap:28px;align-items:start}
form{display:grid;gap:18px}

fieldset{--tone:var(--brand);border:0;background:var(--surface);border-radius:var(--radius);padding:22px 24px 24px;margin:0;min-width:0;
  box-shadow:inset 0 4px 0 var(--tone),0 1px 2px rgba(15,33,71,.05),0 8px 24px rgba(40,70,160,.07)}
fieldset.t-green{--tone:var(--bull)} fieldset.t-red{--tone:var(--bear)} fieldset.t-amber{--tone:var(--flat)} fieldset.t-violet{--tone:#8b5cf6} fieldset.t-sky{--tone:#0ea5e9}
legend{font-family:var(--display);font-weight:600;font-size:1.15rem;padding:6px 0 4px;display:flex;align-items:center;gap:8px;float:left;width:100%;margin-bottom:10px}
legend i{width:10px;height:10px;border-radius:50%;background:var(--tone);display:inline-block}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px 22px;clear:both}
.field{display:flex;flex-direction:column;gap:6px;min-width:0}
.field label{font-weight:500;font-size:.93rem}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
output{font-variant-numeric:tabular-nums;font-weight:600;color:var(--tone);font-size:.95rem}

select,input[type=text]{width:100%;padding:11px 13px;border:1.5px solid var(--line);border-radius:11px;background:var(--field);color:var(--ink);font:inherit;transition:border-color .15s,box-shadow .15s}
select:hover,input[type=text]:hover{border-color:#b9c7ea}
input[type=text]:focus,select:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 4px rgba(59,91,255,.15);background:#fff}
input.bad{border-color:var(--bear);background:#fff7f7}
.msg{display:none;color:var(--bear);font-size:.82rem}
.hint{color:var(--muted);font-size:.78rem}
.field.invalid .hint{display:none}
.notice{display:none;margin:0 0 16px;padding:10px 12px;border-radius:10px;background:var(--flat-bg);color:#7a4f00;font-size:.85rem}
.field.invalid .msg{display:block}

input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:7px;border-radius:7px;background:linear-gradient(to right,var(--tone) var(--fill,50%),#e3e9f7 var(--fill,50%));margin:10px 0 4px;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;border:3px solid var(--tone);box-shadow:0 2px 6px rgba(15,33,71,.25)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:3px solid var(--tone)}
input[type=range]:focus-visible{outline:3px solid rgba(59,91,255,.35);outline-offset:4px}
.ends{display:flex;justify-content:space-between;color:var(--muted);font-size:.78rem}

aside{position:sticky;top:24px}
.result{--tone:var(--brand);--tone-bg:#eef1ff;background:var(--surface);border-radius:22px;padding:26px;
  box-shadow:0 1px 2px rgba(15,33,71,.05),0 16px 40px rgba(40,70,160,.12);transition:background .3s}
.result[data-k=bull]{--tone:var(--bull);--tone-bg:var(--bull-bg)}
.result[data-k=bear]{--tone:var(--bear);--tone-bg:var(--bear-bg)}
.result[data-k=flat]{--tone:var(--flat);--tone-bg:var(--flat-bg)}
.result h2{font-family:var(--display);font-weight:600;font-size:1.05rem;margin:0 0 16px;color:var(--muted)}
.placeholder{background:var(--tone-bg);border-radius:16px;padding:26px 20px;color:var(--muted)}
.placeholder strong{display:block;font-family:var(--display);font-size:1.4rem;color:var(--ink);margin-bottom:4px}
.verdict{display:none}
.verdict.show{display:block;animation:pop .4s ease-out}
@keyframes pop{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.badge{background:var(--tone-bg);border-radius:18px;padding:20px 20px 18px}
.badge .ico{width:46px;height:46px;border-radius:14px;background:var(--tone);display:grid;place-items:center;margin-bottom:12px}
.badge .ico svg{width:26px;height:26px;stroke:#fff;fill:none;stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round}
.big{font-family:var(--display);font-weight:700;font-size:2.15rem;line-height:1.05;letter-spacing:-.02em;margin:0;color:var(--tone)}
.note{margin:8px 0 0;color:var(--ink);font-size:.95rem}
.conf{margin:14px 2px 18px;color:var(--muted);font-size:.92rem}
.bars{display:grid;gap:14px}
.bar .meta{display:flex;justify-content:space-between;font-size:.92rem;margin-bottom:5px}
.bar.top .meta{font-weight:600}
.track{height:10px;border-radius:10px;background:#e8edf9;overflow:hidden}
.fill{height:100%;width:0;border-radius:10px;transition:width .6s cubic-bezier(.2,.8,.2,1)}

.actions{display:grid;gap:10px;margin-top:22px}
button{font:inherit;font-weight:600;border:0;border-radius:13px;padding:14px 18px;cursor:pointer;transition:transform .12s,background .15s,box-shadow .15s}
button:focus-visible{outline:3px solid rgba(59,91,255,.4);outline-offset:3px}
.primary{background:var(--brand);color:#fff;box-shadow:0 8px 18px rgba(59,91,255,.3)}
.primary:hover{background:var(--brand-dark)}
.primary:active{transform:scale(.98)}
.primary[disabled]{opacity:.7;cursor:progress}
.ghost{background:transparent;color:var(--muted);border:1.5px solid var(--line)}
.ghost:hover{background:var(--field)}
.error{display:none;margin-top:14px;padding:10px 12px;border-radius:10px;background:var(--bear-bg);color:#a3262b;font-size:.9rem}
.fine{margin:14px 0 0;color:var(--muted);font-size:.78rem}

@media (max-width:920px){ .layout{grid-template-columns:1fr} aside{position:static;order:-1} }
@media (max-width:560px){ .grid{grid-template-columns:1fr} .wrap{padding:28px 16px 48px} }
@media (prefers-reduced-motion:reduce){ *{animation:none!important;transition:none!important} }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Market Outlook Predictor</h1>
    <p>Enter a company's layoff, hiring, AI and financial details. A random forest reads them and tells you whether the picture looks like a bull market, a recession or a sideways market.</p>
  </header>

{% macro text(name, title) %}
  <div class="field" data-name="{{ name }}">
    <label for="{{ name }}">{{ title }}</label>
    <input type="text" id="{{ name }}" name="{{ name }}" autocomplete="off" placeholder="{{ placeholders[name] }}" spellcheck="false">
    <span class="msg"></span>
  </div>
{% endmacro %}

{% macro num(name, title, unit='', lo_text='', hi_text='') %}
  {% set lo, hi, step, default = numeric[name] %}
  <div class="field" data-name="{{ name }}">
    <label for="{{ name }}">{{ title }}</label>
    <input type="text" inputmode="decimal" id="{{ name }}" name="{{ name }}" autocomplete="off" placeholder="e.g. {{ default }}{{ unit }}" spellcheck="false">
    <span class="hint">Typical: {{ '{:,}'.format(lo) }} to {{ '{:,}'.format(hi) }}{{ unit }}</span>
    <span class="msg"></span>
  </div>
{% endmacro %}

  <div class="layout">
    <form id="form" autocomplete="off" novalidate>
      <fieldset class="t-sky">
        <legend><i></i>Company</legend>
        <div class="grid">
          {{ text('company_name','Company name') }}
          {{ text('industry','Industry') }}
          {{ text('country','Country') }}
          {{ text('company_size','Company size') }}
        </div>
      </fieldset>

      <fieldset class="t-red">
        <legend><i></i>Layoff event</legend>
        <div class="grid">
          {{ text('month','Month') }}
          <div class="field" data-name="year">
            <label for="year">Year</label>
            <input type="text" inputmode="numeric" id="year" name="year" autocomplete="off" placeholder="e.g. 2025">
            <span class="msg"></span>
          </div>
          {{ num('layoffs_count','People laid off') }}
          {{ num('layoff_percentage','Share of workforce','%') }}
          {{ text('reason_for_layoffs','Reason for layoffs') }}
        </div>
      </fieldset>

      <fieldset class="t-violet">
        <legend><i></i>AI exposure</legend>
        <div class="grid">
          {{ num('ai_adoption_level','AI adoption level',lo_text='None',hi_text='Full') }}
          {{ num('ai_automation_impact','Automation impact') }}
          {{ num('ai_replacement_risk','Replacement risk',lo_text='Low',hi_text='High') }}
        </div>
      </fieldset>

      <fieldset class="t-green">
        <legend><i></i>Hiring</legend>
        <div class="grid">
          {{ num('open_roles','Open roles') }}
          {{ text('hiring_trend','Hiring trend') }}
          {{ text('top_hiring_role','Most-hired role') }}
          {{ num('remote_jobs_percentage','Remote jobs','%') }}
        </div>
      </fieldset>

      <fieldset class="t-amber">
        <legend><i></i>Financial health</legend>
        <div class="grid">
          {{ num('stock_growth_percent','Stock growth','%') }}
          {{ num('revenue_growth_percent','Revenue growth','%') }}
          {{ num('salary_budget_change','Salary budget change','%') }}
        </div>
      </fieldset>

      <fieldset>
        <legend><i></i>Employees</legend>
        <div class="grid">
          {{ num('employee_sentiment','Employee sentiment',lo_text='Low',hi_text='High') }}
          {{ num('job_security_score','Job security score',lo_text='Low',hi_text='High') }}
        </div>
      </fieldset>
    </form>

    <aside aria-live="polite">
      <div class="result" id="result">
        <h2>Prediction</h2>
        <div class="placeholder" id="placeholder">
          <strong>Ready when you are</strong>
          Fill in the details, then select Predict outcome.
        </div>
        <div class="verdict" id="verdict">
          <div class="badge">
            <div class="ico" id="ico"></div>
            <p class="big" id="label"></p>
            <p class="note" id="note"></p>
          </div>
          <p class="conf" id="conf"></p>
          <p class="notice" id="notice"></p>
          <div class="bars" id="bars"></div>
        </div>
        <div class="error" id="error" role="alert"></div>
        <div class="actions">
          <button class="primary" id="go" type="button">Predict outcome</button>
          <button class="ghost" id="reset" type="button">Reset inputs</button>
        </div>
        <p class="fine">A model estimate from the data it was trained on, not financial advice.</p>
      </div>
    </aside>
  </div>
</div>

<script>
const OUTCOMES = {{ outcomes|tojson }};
const TEXT_FIELDS = {{ text_fields|tojson }};
const ICONS = {
  bull:'<svg viewBox="0 0 24 24"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>',
  bear:'<svg viewBox="0 0 24 24"><path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/></svg>',
  flat:'<svg viewBox="0 0 24 24"><path d="M3 12h18"/><path d="M17 8l4 4-4 4"/></svg>'
};
const BAR_COLORS = {bull:'var(--bull)', bear:'var(--bear)', flat:'var(--flat)'};
const form = document.getElementById('form');
const defaults = {};

/* ---------- empty text inputs ---------- */
const TYPED = {{ all_fields|tojson }};
function setBad(name, message){
  const el = document.getElementById(name);
  el.classList.add('bad');
  const f = el.closest('.field'); f.classList.add('invalid');
  f.querySelector('.msg').textContent = message;
}
function clearBad(el){
  el.classList.remove('bad');
  const f = el.closest('.field'); f.classList.remove('invalid');
}
TYPED.forEach(name=>{
  const el = document.getElementById(name);
  defaults[name] = '';
  el.addEventListener('input', ()=>clearBad(el));
});

/* ---------- reset ---------- */
document.getElementById('reset').addEventListener('click', ()=>{
  Object.entries(defaults).forEach(([id,v])=>{
    const el = document.getElementById(id); el.value = v;
    clearBad(el);
  });
  document.getElementById('verdict').classList.remove('show');
  document.getElementById('placeholder').style.display = '';
  document.getElementById('error').style.display = 'none';
  document.getElementById('notice').style.display = 'none';
  document.getElementById('result').removeAttribute('data-k');
});

/* ---------- predict ---------- */
document.getElementById('go').addEventListener('click', async (e)=>{
  const btn = e.currentTarget, err = document.getElementById('error');
  err.style.display = 'none';
  const payload = Object.fromEntries(new FormData(form).entries());
  let ok = true;
  TYPED.forEach(name=>{
    clearBad(document.getElementById(name));
    if(!String(payload[name] || '').trim()){ ok = false; setBad(name, 'Required'); }
  });
  if(!ok){ err.textContent = 'Fill in the highlighted fields first.'; err.style.display = 'block'; return; }

  btn.disabled = true; btn.textContent = 'Predicting…';
  try{
    const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const out = await res.json();
    if(!res.ok){
      if(out.fields){ Object.entries(out.fields).forEach(([k,m])=>setBad(k,m)); }
      throw new Error(out.error || 'Prediction failed');
    }
    show(out);
  }catch(ex){
    err.textContent = ex.message; err.style.display = 'block';
  }finally{
    btn.disabled = false; btn.textContent = 'Predict outcome';
  }
});

function show(out){
  const top = out.probabilities[out.prediction_index];
  document.getElementById('result').dataset.k = top.key;
  document.getElementById('placeholder').style.display = 'none';
  const v = document.getElementById('verdict');
  v.classList.remove('show'); void v.offsetWidth; v.classList.add('show');
  document.getElementById('ico').innerHTML = ICONS[top.key] || '';
  document.getElementById('label').textContent = top.label;
  document.getElementById('note').textContent = top.note;
  document.getElementById('conf').textContent = Math.round(top.probability*100) + '% of the trees agree';
  const nt = document.getElementById('notice');
  if(out.approximate && out.approximate.length){
    nt.textContent = out.approximate.join(', ') + ' not found in your dataset, so an approximate code was used. Add your dataset CSV next to app.py for exact results.';
    nt.style.display = 'block';
  } else { nt.style.display = 'none'; }
  const bars = document.getElementById('bars'); bars.innerHTML = '';
  out.probabilities.forEach(p=>{
    const d = document.createElement('div');
    d.className = 'bar' + (p === top ? ' top' : '');
    d.innerHTML = '<div class="meta"><span></span><span></span></div><div class="track"><div class="fill"></div></div>';
    const s = d.querySelectorAll('span');
    s[0].textContent = p.label; s[1].textContent = (p.probability*100).toFixed(1) + '%';
    const f = d.querySelector('.fill'); f.style.background = BAR_COLORS[p.key] || 'var(--brand)';
    bars.appendChild(d);
    requestAnimationFrame(()=>requestAnimationFrame(()=>{ f.style.width = (p.probability*100) + '%'; }));
  });
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        PAGE, numeric=NUMERIC, placeholders=PLACEHOLDERS,
        outcomes=OUTCOMES, text_fields=TEXT_FIELDS, all_fields=FEATURES,
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    row, problems, approximate = {}, {}, []
    for name in FEATURES:
        raw = str(data.get(name, "")).strip()
        if not raw:
            problems[name] = "Required"
        elif name in CATEGORICAL:
            code, approx = encode(name, raw)
            row[name] = float(code)
            if approx:
                approximate.append(name.replace("_", " "))
        else:
            try:
                row[name] = float(raw.replace(",", ""))
            except ValueError:
                problems[name] = "Enter a number"
    if problems:
        return jsonify(error="Some inputs need fixing: " + ", ".join(
            k.replace("_", " ") for k in problems), fields=problems), 400

    X = pd.DataFrame([row], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    best = int(proba.argmax())

    probs = []
    for i, p in enumerate(proba):
        o = OUTCOMES[i] if i < len(OUTCOMES) else {"label": f"Class {CLASSES[i]}", "key": "", "note": ""}
        probs.append({"class": CLASSES[i], "label": o["label"], "key": o["key"],
                      "note": o["note"], "probability": round(float(p), 4)})
    return jsonify(prediction=CLASSES[best], prediction_index=best, approximate=approximate,
                   label=probs[best]["label"], probabilities=probs)


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)