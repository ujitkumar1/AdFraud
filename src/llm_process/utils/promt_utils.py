class Prompts:
    TOPIC_EXTRACTION_PROMPT: str = """
        Extract and analyze the key topics and concepts from the following content about mobile ad fraud.
        Focus on identifying:
        1. Main fraud types and techniques
        2. Detection methods and tools
        3. Prevention strategies
        4. Impact and consequences
        5. Technical details and implementation
        6. Industry standards and best practices
    
        Content: {content}
    
        Provide a detailed analysis in this format:
        - Main Topic:
            key_concepts: [list of key concepts]
            technical_details: [specific technical information]
            related_topics: [related areas]
            importance: [why this is significant]
    """

    HIERARCHY_GENERATION_PROMPT: str = """
        Create a comprehensive YAML hierarchy for mobile ad fraud based on the following analyzed content.
        Focus on creating a structure that:
        1. Has at least 3 levels of depth where appropriate
        2. Includes detailed descriptions for each topic
        3. Provides specific examples and use cases
        4. Links related concepts across different sections
        5. Includes technical specifications and implementation details
        6. References relevant standards and protocols
    
        Analyzed Content: {content}
    
        Requirements for the YAML:
        - Use descriptive section names
        - Include detailed descriptions (2-3 sentences minimum)
        - Add practical examples
        - Include technical specifications where relevant
        - Link to related topics
        - Include prevention strategies and best practices
    
        Note: Ensure maximum detail and practical applicability
              Try to make it as verbose as possible 
              The final output should be yaml only (IMPORTANT)
    
    """
