from langchain_core.tools import tool
import arxiv
import wikipedia

@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for recent research papers matching the given query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for r in client.results(search):
        results.append(
            f"Title: {r.title}\n"
            f"Authors: {', '.join([a.name for a in r.authors])}\n"
            f"Published: {r.published}\n"
            f"Summary: {r.summary}\n"
            f"PDF: {r.pdf_url}\n"
        )
    return "\n---\n".join(results) if results else "No results found on arXiv."

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for definitions and general background information on algorithms/concepts."""
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e.options[:5]}..."
    except wikipedia.exceptions.PageError:
        return "Page not found on Wikipedia."
