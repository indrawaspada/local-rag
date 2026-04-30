import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Prompt evaluasi dalam Bahasa Indonesia
# ─────────────────────────────────────────────

PROMPT_FAITHFULNESS = """Anda adalah penilai kualitas sistem RAG. Tugas Anda menilai apakah JAWABAN hanya berdasarkan KONTEKS yang diberikan.

PERTANYAAN: {question}

KONTEKS:
{context}

JAWABAN: {answer}

Berikan skor antara 0.0 dan 1.0:
- 1.0 = Jawaban 100% berdasarkan konteks, tidak ada informasi yang dikarang.
- 0.5 = Sebagian jawaban berdasarkan konteks, sebagian dikarang.
- 0.0 = Jawaban sama sekali tidak berdasarkan konteks yang diberikan.

Balas HANYA dengan JSON berikut, tanpa penjelasan tambahan, tanpa markdown:
{{"score": 0.8, "alasan": "penjelasan singkat dalam bahasa indonesia"}}"""

PROMPT_RELEVANCY = """Anda adalah penilai kualitas sistem RAG. Tugas Anda menilai apakah JAWABAN benar-benar menjawab PERTANYAAN.

PERTANYAAN: {question}

JAWABAN: {answer}

Berikan skor antara 0.0 dan 1.0:
- 1.0 = Jawaban sangat relevan dan menjawab pertanyaan secara lengkap.
- 0.5 = Jawaban cukup relevan namun tidak lengkap.
- 0.0 = Jawaban tidak relevan atau tidak menjawab pertanyaan.

Balas HANYA dengan JSON berikut, tanpa penjelasan tambahan, tanpa markdown:
{{"score": 0.8, "alasan": "penjelasan singkat dalam bahasa indonesia"}}"""

PROMPT_CONTEXT_QUALITY = """Anda adalah penilai kualitas sistem RAG. Tugas Anda menilai apakah KONTEKS yang ditemukan relevan dan cukup untuk menjawab PERTANYAAN.

PERTANYAAN: {question}

KONTEKS:
{context}

Berikan skor antara 0.0 dan 1.0:
- 1.0 = Konteks sangat relevan dan berisi semua informasi yang dibutuhkan.
- 0.5 = Konteks relevan sebagian, ada informasi penting yang tidak ditemukan.
- 0.0 = Konteks tidak relevan dengan pertanyaan.

Balas HANYA dengan JSON berikut, tanpa penjelasan tambahan, tanpa markdown:
{{"score": 0.8, "alasan": "penjelasan singkat dalam bahasa indonesia"}}"""


def _call_llm_and_parse(llm, prompt: str) -> tuple[float, str]:
    """Panggil LLM dan parse skor dari respons JSON."""
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Bersihkan markdown code block jika ada (```json ... ```)
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    content = part
                    break

        # Temukan kurung kurawal pertama dan terakhir
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]

        data = json.loads(content)
        score = float(data.get("score", float("nan")))
        alasan = data.get("alasan", data.get("reason", ""))
        return score, alasan

    except Exception as e:
        return float("nan"), f"Error parsing: {e}"


def run_ragas_evaluation(questions, answers, contexts, ground_truths=None):
    """
    Evaluasi kualitas RAG menggunakan Gemini sebagai juri.
    Menggunakan Custom LLM-as-Judge (bukan library Ragas internal)
    agar tidak bergantung pada format parsing Ragas yang sering NaN.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "GANTI_DENGAN_KEY_ANDA":
        raise ValueError("GOOGLE_API_KEY belum diatur di file .env")

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Dikonfirmasi bekerja dengan API key ini
        temperature=0,
        google_api_key=api_key,
    )

    rows = []
    for i, (question, answer, context_list) in enumerate(zip(questions, answers, contexts)):
        context_str = "\n\n---\n\n".join(context_list)

        # 1. Faithfulness
        faithfulness_score, faithfulness_reason = _call_llm_and_parse(
            llm,
            PROMPT_FAITHFULNESS.format(question=question, context=context_str, answer=answer),
        )

        # 2. Answer Relevancy
        relevancy_score, relevancy_reason = _call_llm_and_parse(
            llm,
            PROMPT_RELEVANCY.format(question=question, answer=answer),
        )

        # 3. Context Quality
        context_score, context_reason = _call_llm_and_parse(
            llm,
            PROMPT_CONTEXT_QUALITY.format(question=question, context=context_str),
        )

        rows.append({
            "no":                i + 1,
            "question":          question,
            "answer":            answer[:200] + "..." if len(answer) > 200 else answer,
            "faithfulness":      faithfulness_score,
            "answer_relevancy":  relevancy_score,
            "context_quality":   context_score,
            "alasan_faithfulness": faithfulness_reason,
            "alasan_relevancy":    relevancy_reason,
            "alasan_context":      context_reason,
        })

    return pd.DataFrame(rows)
