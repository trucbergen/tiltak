@st.cache_data(show_spinner=False, ttl=86400)
def hent_ai_svar(prompt, kommune, kategori, kontekst_csv, stats_totalt, stats_kommuner):
    kontekst_df = pd.read_json(kontekst_csv)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": lag_system_prompt(
                    kontekst_df,
                    {
                        "totalt": stats_totalt,
                        "kommuner": stats_kommuner,
                        "kategorier": 0
                    }
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content