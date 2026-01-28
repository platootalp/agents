---
name: web-search
description: Enables web searches using the Tavily client to find up-to-date information from the internet. Use this skill when users need current information, research on topics, or data that may not be in Claude's training data.
---

# Web Search Skill

This skill enables Claude to perform web searches using the Tavily API to retrieve current, relevant information from the internet.

## When to Use This Skill

Use this skill when:
- Users need up-to-date information (news, recent developments, current statistics)
- Researching topics that may have changed since Claude's knowledge cutoff
- Verifying facts or finding authoritative sources
- Gathering multiple perspectives on a topic
- The requested information is likely not in Claude's training data

## How to Use This Skill

1. **Identify the search query**: Determine the most effective keywords and phrases for the search
2. **Perform the search**: Use the Tavily client to execute the search
3. **Analyze results**: Review search results for relevance and reliability
4. **Synthesize information**: Combine information from multiple sources to provide a comprehensive answer
5. **Cite sources**: Include links to original sources when appropriate

## Tavily API Integration

The Tavily API provides:
- Fast, relevant search results
- Source credibility scoring
- Content extraction from web pages
- Support for advanced search parameters

### Basic Search Parameters
- `query`: The search query string (required)
- `search_depth`: "basic" or "advanced" (default: "basic")
- `include_images`: Boolean to include images in results (default: false)
- `max_results`: Number of results to return (default: 5)

### Example Usage

```python
from tavily import TavilyClient

tavily_client = TavilyClient(api_key="your_api_key")

results = tavily_client.search(
    query="latest developments in AI",
    search_depth="advanced",
    max_results=5
)
```

## Best Practices

1. **Craft effective queries**: Use specific, targeted keywords rather than broad terms
2. **Verify information**: Cross-reference multiple sources when possible
3. **Consider source credibility**: Prioritize authoritative, reputable sources
4. **Handle conflicting information**: Acknowledge when sources disagree and explain why
5. **Respect copyright**: Summarize information rather than copying large portions of text
6. **Include source attribution**: Provide links to original sources for verification

## Error Handling

Common issues and solutions:
- **API rate limits**: Implement exponential backoff for retries
- **No results found**: Try rephrasing the query or using broader terms
- **Irrelevant results**: Refine search terms or use more specific keywords
- **Connection errors**: Check API key and network connectivity

## Related Skills

- **research-analyst**: For in-depth research projects requiring synthesis of multiple sources
- **fact-checker**: For verifying claims and assessing source credibility
- **content-summarizer**: For condensing information from multiple search results