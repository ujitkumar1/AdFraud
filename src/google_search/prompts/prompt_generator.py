class PromptGenerator:
    @staticmethod
    def zero_shot_prompt(topic: str, num: int) -> str:
        return f"""
                    Generate {num} search queries for "{topic}".

                    Each query must:
                    1. Include exact phrases in quotes.
                    2. Target specific subtopics such as:
                       - Types of ad fraud (click fraud, install fraud, etc.)
                       - Prevention techniques
                       - Role of ad networks
                       - Legal implications
                       - Economic impact
                    3. Be concise and clear.

                    Example format:
                    "install fraud prevention techniques"
                    "legal implications of mobile ad fraud"



                    NOTE: I’m generating this Google search query for research purposes, so I need efficient search queries that cover the overall topics.
                    Generate {num} optimized Google search queries.
                    """
