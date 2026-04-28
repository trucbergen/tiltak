# ============================================================
# app.py - Helse-Monitor 2026 Chatbot
# ============================================================
import streamlit as st
import sqlite3
import pandas as pd
from groq import Groq

# ============================================================
# 1. KONFIGURASJON
# ============================================================
st.set_page_config(
    page_title="Helse-Monitor 2026",
    page_icon="🏥",
    layout="wide"
)

DB_PATH = "master_helse.db"
MODEL   = "llama-3.3-70b-versatile"

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_client()

# ============================================================
# 2. DATABASE-FUNKSJONER (sikker + smart søk)
# ============================================================

@st.cache_data
def hent_alle_kommuner():
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT DISTINCT kommune FROM tiltak ORDER BY kommune", con)
    con.close()
    return df["kommune"].tolist()

@st.cache_data
def hent_alle_kategorier():
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT DISTINCT kategori FROM tiltak ORDER BY kategori", con)
    con.close()
    return df["kategori"].tolist()

@st.cache_data
def hent_statistikk():
    con = sqlite3.connect(DB_PATH)
    stats = {
        "totalt":    pd.read_sql("SELECT count(*) as n FROM tiltak", con)["n"][0],
        "kommuner":  pd.read_sql("SELECT count(DISTINCT kommune) as n FROM tiltak", con)["n"][0],
        "kategorier":pd.read_sql("SELECT count(DISTINCT kategori) as n FROM tiltak", con)["n"][0],
    }
    con.close()
    return stats

def sok_database(sporsmal, kommune=None, kategori=None, limit=20):
    """
    Smart søk — bruker FLERE nøkkelord og filtrerer på kommune/kategori
    Mye bedre enn å bare søke på første ord!
    """
    con = sqlite3.connect(DB_PATH)

    # Bygg WHERE-klausul
    betingelser = []
    params      = []

    # Søk i tiltak_navn OG beskrivelse OG kategori
    sokeord = [o.strip() for o in sporsmal.replace(",", " ").split() if len(o.strip()) > 2]
    if sokeord:
        or_klausuler = []
        for ord in sokeord[:5]:  # Maks 5 søkeord
            or_klausuler.append(
                "(lower(tiltak_navn) LIKE ? OR lower(beskrivelse) LIKE ? OR lower(kategori) LIKE ?)"
            )
            sok = f"%{ord.lower()}%"
            params.extend([sok, sok, sok])
        betingelser.append(f"({' OR '.join(or_klausuler)})")

    # Filter kommune
    if kommune and kommune != "Alle":
        betingelser.append("lower(kommune) = ?")
        params.append(kommune.lower())

    # Filter kategori
    if kategori and kategori != "Alle":
        betingelser.append("lower(kategori) = ?")
        params.append(kategori.lower())

    where = f"WHERE {' AND '.join(betingelser)}" if betingelser else ""
    sql   = f"""
        SELECT kommune, kategori, tiltak_navn, malgruppe, beskrivelse, kilde_url
        FROM tiltak
        {where}
        ORDER BY kommune
        LIMIT {limit}
    """

    df = pd.read_sql(sql, con, params=params)
    con.close()
    return df

def gap_analyse(kategori):
    """Finn kommuner som MANGLER en spesifikk kategori"""
    con = sqlite3.connect(DB_PATH)
    alle = pd.read_sql("SELECT DISTINCT kommune FROM tiltak ORDER BY kommune", con)
    har  = pd.read_sql(
        "SELECT DISTINCT kommune FROM tiltak WHERE lower(kategori) LIKE ?",
        con, params=[f"%{kategori.lower()}%"]
    )
    con.close()
    mangler = alle[~alle["kommune"].isin(har["kommune"])]
    return mangler

# ============================================================
# 3. SYSTEM-PROMPT (mye bedre enn original!)
# ============================================================

def lag_system_prompt(kontekst_df, stats):
    return f"""Du er Helse-Monitor AI — en ekspert pa helsetiltak i norske kommuner.

Du har tilgang til en database med {stats['totalt']:,} helsetiltak fra {stats['kommuner']} norske kommuner.

RELEVANTE DATA FRA DATABASEN:
{kontekst_df.to_string(index=False) if not kontekst_df.empty else "Ingen spesifikke data funnet for dette spørsmålet."}

REGLER:
- Svar alltid pa norsk
- Vær konkret og faktabasert — bruk dataene ovenfor
- Hvis data mangler, si det ærlig
- Fremhev interessante mønstre eller mangler du ser
- Foreslå gjerne gap-analyse hvis relevant
- Hold svarene kortfattede men informative
- Bruk bullet-points for lister av tiltak"""

# ============================================================
# 4. UI — SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d9/Flag_of_Norway.svg", width=60)
    st.title("🏥 Helse-Monitor")
    st.caption("357 norske kommuner")

    st.divider()

    # Filter
    st.subheader("🔍 Filter")
    alle_kommuner  = ["Alle"] + hent_alle_kommuner()
    alle_kategorier = ["Alle"] + hent_alle_kategorier()

    valgt_kommune  = st.selectbox("Kommune", alle_kommuner)
    valgt_kategori = st.selectbox("Kategori", alle_kategorier)

    st.divider()

    # Statistikk
    stats = hent_statistikk()
    st.subheader("📊 Statistikk")
    st.metric("Totalt tiltak", f"{stats['totalt']:,}")
    st.metric("Kommuner", stats["kommuner"])
    st.metric("Kategorier", stats["kategorier"])

    st.divider()

    # Hurtigspørsmål
    st.subheader("💡 Hurtigspørsmål")
    hurtig = [
        "Hvilke kommuner har frisklivsentral?",
        "Sammenlign psykisk helse i Oslo og Bergen",
        "Hvilke kommuner mangler rustilbud?",
        "Hva er de vanligste helsetiltakene?",
        "Vis alle tiltak for eldreomsorg i Tromsø",
    ]
    for sporsmal in hurtig:
        if st.button(sporsmal, use_container_width=True):
            st.session_state.hurtig_sporsmal = sporsmal

    st.divider()
    if st.button("🗑️ Tøm chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# 5. HOVED-CHAT
# ============================================================

st.title("🏥 Helse-Monitor 2026")
st.caption("Spør meg om helsetiltak i alle 357 norske kommuner")

# Initialisér chat-historikk
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hurtig_sporsmal" not in st.session_state:
    st.session_state.hurtig_sporsmal = None

# Velkomstmelding
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(f"""
        Hei! 👋 Jeg er Helse-Monitor AI.

        Jeg har tilgang til **{stats['totalt']:,} helsetiltak** fra **{stats['kommuner']} norske kommuner**.

        Du kan spørre meg om:
        - 🔍 Hvilke tiltak finnes i en bestemt kommune?
        - 📊 Sammenligning mellom kommuner
        - ⚠️ Hvilke kommuner mangler spesifikke tilbud?
        - 📈 Statistikk og trender på tvers av landet
        """)

# Vis chat-historikk
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Håndter hurtigspørsmål
if st.session_state.hurtig_sporsmal:
    prompt = st.session_state.hurtig_sporsmal
    st.session_state.hurtig_sporsmal = None
else:
    prompt = st.chat_input("Spør om helsetiltak i norske kommuner...")

# ============================================================
# 6. PROSESSER SPØRSMÅL
# ============================================================

if prompt:
    # Vis bruker-melding
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Søk i database
    with st.spinner("Søker i database..."):
        resultater = sok_database(
            prompt,
            kommune  = valgt_kommune  if valgt_kommune  != "Alle" else None,
            kategori = valgt_kategori if valgt_kategori != "Alle" else None
        )

    # Vis data-tabell hvis funn
    if not resultater.empty:
        with st.expander(f"📋 Fant {len(resultater)} relevante tiltak i databasen", expanded=False):
            st.dataframe(
                resultater[["kommune","kategori","tiltak_navn","malgruppe"]],
                use_container_width=True,
                hide_index=True
            )

    # Groq AI-svar
    with st.chat_message("assistant"):
        with st.spinner("Analyserer..."):
            try:
                # Bygg meldingshistorikk (maks 10 siste)
                historikk = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-10:]
                    if m["role"] == "user"
                ]

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": lag_system_prompt(resultater, stats)},
                        *historikk,
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                svar = response.choices[0].message.content
                st.markdown(svar)
                st.session_state.messages.append({"role": "assistant", "content": svar})

            except Exception as e:
                feil_melding = f"Beklager, noe gikk galt: {str(e)}"
                st.error(feil_melding)
                st.session_state.messages.append({"role": "assistant", "content": feil_melding})