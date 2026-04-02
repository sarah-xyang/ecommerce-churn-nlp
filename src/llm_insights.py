"""
llm_insights.py — Anthropic API calls for churn insight generation.

LLM-as-communication-layer pattern: the ML pipeline produces structured
outputs (metrics, SHAP rankings, ROI figures); Claude consumes those outputs
and produces natural language calibrated to the audience. Neither layer does
the other's job — the analytical work is done by the ML system, the
translation work is done by the LLM.
"""

import os

import anthropic
from dotenv import load_dotenv


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Initialise and return an authenticated Anthropic API client.

    LLM-as-communication-layer pattern: the client returned here is the
    entry point for translating technical ML findings into stakeholder
    language. By centralising client construction, notebooks and batch
    scoring scripts share the same auth logic — making key rotation and
    environment configuration a single-point change.

    Returns
    -------
    anthropic.Anthropic
        Authenticated client ready for messages.create() calls.

    Raises
    ------
    ValueError
        If ANTHROPIC_API_KEY is not set in the environment after loading .env.
    """
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Add it to your .env file as: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=api_key)


def build_findings_dict(
    churn_rate: float,
    model_roc_auc: float,
    model_f1: float,
    top_features: list[str],
    sentiment_finding: str,
    model_limitation: str,
) -> dict:
    """
    Construct the standardised findings dictionary used as the LLM prompt payload.

    A standardised schema matters for two reasons: (1) it makes the LLM prompt
    reproducible — the same structured input produces consistent, auditable output
    across runs, notebooks, and batch scoring jobs; (2) it enforces a clean
    separation between analysis (notebooks 01–03) and communication (notebook 04),
    so any downstream consumer of this function gets the same prompt structure
    regardless of where it is called from.

    Parameters
    ----------
    churn_rate : float
        Overall one-order churn rate as a decimal (e.g. 0.97 for 97%).
    model_roc_auc : float
        XGBoost test-set ROC-AUC score.
    model_f1 : float
        XGBoost test-set F1 score (positive class = churned).
    top_features : list[str]
        Top SHAP predictors, ordered by importance descending.
    sentiment_finding : str
        Plain-English summary of how sentiment features ranked in SHAP.
    model_limitation : str
        Honest plain-English description of what the model cannot do.

    Returns
    -------
    dict
        Standardised findings dictionary ready to pass to generate_executive_summary().
    """
    return {
        "overall_churn_rate": f"{churn_rate:.1%}",
        "model_performance": (
            f"XGBoost ROC-AUC {model_roc_auc:.3f}, "
            f"F1 {model_f1:.3f} (positive class = churned)"
        ),
        "top_predictors": top_features,
        "sentiment_finding": sentiment_finding,
        "model_limitation": model_limitation,
    }


def generate_executive_summary(
    findings: dict,
    client: anthropic.Anthropic,
) -> str:
    """
    Call the Anthropic API to generate a non-technical executive summary
    of the churn analysis findings for the Olist growth team.

    A structured findings dict is passed rather than raw data because structured
    input produces consistent, auditable output: the same schema yields the same
    prompt shape on every run, making it straightforward to diff outputs across
    model versions or time periods. Raw DataFrames or notebook variables would
    produce uncontrolled, hard-to-reproduce prompts.

    Parameters
    ----------
    findings : dict
        Standardised findings dictionary produced by build_findings_dict().
    client : anthropic.Anthropic
        Authenticated Anthropic client from get_anthropic_client().

    Returns
    -------
    str
        Plain-English executive summary text (max ~200 words).

    Raises
    ------
    RuntimeError
        On any Anthropic API failure, with a descriptive message.
    """
    # Serialise findings into a readable bullet list for the prompt
    findings_text = ""
    for key, value in findings.items():
        if isinstance(value, list):
            findings_text += f"- {key}: " + "; ".join(value) + "\n"
        else:
            findings_text += f"- {key}: {value}\n"

    prompt = (
        "You are a data analyst summarising a customer churn study for the growth team "
        "at Olist, a Brazilian e-commerce marketplace. The audience is non-technical — "
        "avoid jargon like ROC-AUC, SHAP, or F1 score.\n\n"
        "Here are the key findings from the analysis:\n\n"
        f"{findings_text}\n"
        "Write a concise executive summary (maximum 200 words) that:\n"
        "1. States what drives churn in plain business language (no jargon like "
        "ROC-AUC or SHAP)\n"
        "2. Is honest about what the model can and cannot do\n"
        "3. Lists exactly 3 specific actions the growth team should prioritise, "
        "each grounded in the findings above"
    )

    try:
        # Sending the full findings context + stakeholder framing to claude-opus-4-5.
        # System turn is omitted intentionally — all framing is in the user turn so
        # the prompt is self-contained and easy to inspect in logs.
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Anthropic API connection failed: {e}") from e
    except anthropic.AuthenticationError:
        raise RuntimeError(
            "Anthropic API authentication failed. Check your ANTHROPIC_API_KEY in .env."
        )
    except anthropic.RateLimitError:
        raise RuntimeError(
            "Anthropic API rate limit reached. Wait a moment and retry."
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


def analyze_review_themes(
    reviews: list[str],
    client: anthropic.Anthropic,
) -> str:
    """
    Call the Anthropic API to identify recurring qualitative churn themes
    from a sample of churned-customer review comments.

    SHAP explains which structured features (payment method, delivery delay,
    freight ratio) are statistically associated with churn. Review theme analysis
    explains what customers actually experienced — the specific failures, the
    emotional register, the exact moments of disappointment. The two layers are
    complementary: SHAP gives the growth team *where to focus*; review themes
    give them *what to fix*. Neither is sufficient alone.

    Claude also performs implicit translation here: the reviews are in Portuguese
    and Claude reads, translates, and clusters them into English-language themes
    and translated representative quotes in a single pass.

    Parameters
    ----------
    reviews : list[str]
        Review comment strings from churned customers (Portuguese text).
    client : anthropic.Anthropic
        Authenticated Anthropic client from get_anthropic_client().

    Returns
    -------
    str
        Numbered list of top 5 themes, each with a 1-2 sentence explanation
        and a representative translated quote.

    Raises
    ------
    RuntimeError
        On any Anthropic API failure, with a descriptive message.
    """
    # Format review list for the prompt — numbered list so Claude can
    # reference specific entries and the token budget is used efficiently.
    reviews_text = "\n".join(f"{i + 1}. {review}" for i, review in enumerate(reviews))

    # Goal: qualitative theme extraction to complement the quantitative SHAP
    # analysis. The prompt requests English output with translated quotes so
    # non-Portuguese-speaking stakeholders can read the evidence directly.
    prompt = (
        "The following are customer reviews from an e-commerce marketplace "
        "(Olist, Brazil). The reviews are written in Portuguese.\n\n"
        f"{reviews_text}\n\n"
        "Identify the top 5 recurring themes in these reviews that explain why "
        "customers did not return to make a second purchase.\n"
        "Format your response as a numbered list (1 through 5). For each theme:\n"
        "- Name the theme in bold\n"
        "- Write 1-2 sentences explaining the theme\n"
        "- Include one representative quote translated into English (label it: Quote:)\n\n"
        "Focus on themes most relevant to customer retention decisions."
    )

    try:
        # Sending raw Portuguese review comments as a numbered list.
        # Claude reads, translates, and clusters them into recurring themes
        # in a single pass — implicit translation + theme extraction.
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Anthropic API connection failed: {e}") from e
    except anthropic.AuthenticationError:
        raise RuntimeError(
            "Anthropic API authentication failed. Check your ANTHROPIC_API_KEY in .env."
        )
    except anthropic.RateLimitError:
        raise RuntimeError(
            "Anthropic API rate limit reached. Wait a moment and retry."
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}") from e


if __name__ == "__main__":
    # Actual Project 3 results from notebooks 01–03.
    findings = build_findings_dict(
        churn_rate=0.97,
        model_roc_auc=0.633,
        model_f1=0.803,
        top_features=[
            "payment_installments (rank 1)",
            "payment_value (rank 2)",
            "freight_ratio (rank 4)",
            "delivery_delay_days (rank 5)",
            "days_to_delivery (rank 7)",
        ],
        sentiment_finding=(
            "has_review_text ranks 16th of 57 features (SHAP); "
            "sentiment_polarity ranks 39th. Whether a customer wrote a review "
            "is more predictive of return behaviour than the sentiment of that review."
        ),
        model_limitation=(
            "ROC-AUC 0.633 indicates modest discrimination. Churn is "
            "near-universal (97%), so the model captures platform-wide structural "
            "patterns but cannot reliably distinguish a dissatisfied customer from "
            "one who simply had no further need to purchase."
        ),
    )

    client = get_anthropic_client()

    print("=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print("Calling Anthropic API to generate executive summary...")
    executive_summary = generate_executive_summary(findings, client)
    print(executive_summary)

    print()
    print("=" * 60)
    print("REVIEW THEME ANALYSIS")
    print("=" * 60)
    sample_reviews = [
        "Produto não chegou nunca. Péssimo serviço, não recomendo.",
        "Atrasou mais de 20 dias. Fui obrigado a cancelar.",
        "Recebi um produto completamente diferente do que pedi.",
    ]
    print(f"Sending {len(sample_reviews)} review comments to Anthropic API...")
    review_themes = analyze_review_themes(sample_reviews, client)
    print(review_themes)
