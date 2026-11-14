"""PolicyGuard agent format: ReAct text protocol that fits Tiny Aya's chat template.

Tiny Aya's template only allows alternating user/assistant turns (no `tool` role), so tool
results go back to the model as a user turn starting with "Observation:".

    user:      <question>
    assistant: Thought: ...\nAction: search_docs("...")
    user:      Observation:\n[chunk 12] ...
    assistant: Thought: ...\nFinal Answer: ... [chunk 12]
"""
import re

SYSTEM = """You are PolicyGuard, an assistant that answers questions about Indian health-insurance rules issued by IRDAI.

You can search IRDAI documents with the tool search_docs. Results come from several circulars. Newer circulars can replace older ones: when two documents give different rules on the same point, follow the newer one.

Answer only from what the search results say, and cite the chunks you used like [chunk 12]. If the results do not answer the question, say you could not find it in the documents. Do not guess.

Reply in exactly one of these two forms:

Thought: <short reasoning>
Action: search_docs("<search query>")

or

Thought: <short reasoning>
Final Answer: <answer with [chunk N] citations>"""

ACTION = re.compile(r'Action:\s*search_docs\(\s*["\'](.+?)["\']\s*\)', re.S)
FINAL = re.compile(r"Final Answer:\s*(.+)", re.S)
CITE = re.compile(r"\[chunk (\d+)\]")
NOT_FOUND = re.compile(r"could ?n[o']t find|not (be )?found|not covered|no information|unable to find|"
                       r"do(es)? not (answer|cover|mention|contain)|don'?t have (enough )?information", re.I)


def parse(reply):
    """-> ("search", query) | ("final", answer) | ("invalid", reply)"""
    m = ACTION.search(reply)
    f = FINAL.search(reply)
    if f and (not m or f.start() < m.start()):
        return "final", f.group(1).strip()
    if m:
        return "search", m.group(1).strip()
    return "invalid", reply.strip()


def observation(searcher, hits):
    return "Observation:\n" + searcher.render(hits)
