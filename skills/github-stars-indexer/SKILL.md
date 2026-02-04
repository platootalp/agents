---
name: github-stars-indexer
description: Generate structured index documents for GitHub starred repositories. Use when the user wants to create, organize, or update a README or index document for their GitHub stars collection, organize repositories into categories with tables, create a stars catalog, or build a personal repository collection index with metadata like repo names, descriptions, URLs, and star counts.
---

# GitHub Stars Indexer

## Overview

This skill helps you create professional, well-organized index documents for GitHub starred repositories. It provides templates, structure guidelines, and best practices for organizing repository collections into categorized tables with comprehensive metadata.

## Workflow

Creating a GitHub stars index involves these steps:

1. **Fetch repository data** (optional) - Use script to automatically fetch from GitHub API
2. **Understand requirements** - Determine the scope, style, and organization approach
3. **Choose template** - Select appropriate template based on collection size
4. **Define categories** - Plan the classification structure
5. **Populate content** - Fill in repository information with proper formatting
6. **Enhance and refine** - Add metadata, icons, and polish the document

## Step 0: Fetch Repository Data (Optional)

**When to use:** If the user has a GitHub account and wants to automatically fetch their starred repositories.

**Requirements:**
- GitHub Personal Access Token
- Python 3.7+ with `requests` library

**Script location:** `scripts/fetch_github_stars.py`

**Basic usage:**

```bash
# Install dependencies
pip install requests

# Fetch and generate index in one command
python scripts/fetch_github_stars.py \
  --token YOUR_GITHUB_TOKEN \
  --generate-index my-stars-index.md

# Or fetch JSON data for manual processing
python scripts/fetch_github_stars.py \
  --token YOUR_GITHUB_TOKEN \
  --output stars.json
```

**Script features:**
- Automatically fetches all starred repositories with pagination
- Extracts complete metadata (stars, forks, language, update time, etc.)
- Can generate Markdown index directly (`--generate-index`)
- Supports filtering by minimum stars (`--min-stars`)
- Groups by language or no grouping (`--group-by`)
- Outputs JSON for custom processing

**Common options:**

```bash
# Generate index with 100+ stars only
python scripts/fetch_github_stars.py \
  --token TOKEN \
  --generate-index index.md \
  --min-stars 100

# Get another user's public stars
python scripts/fetch_github_stars.py \
  --token TOKEN \
  --username other-user \
  --output their-stars.json

# No grouping, sorted by stars
python scripts/fetch_github_stars.py \
  --token TOKEN \
  --generate-index index.md \
  --group-by none
```

**Getting a GitHub token:**
1. Visit https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Select `user:read` permission
4. Generate and copy the token

**For detailed usage**, see `scripts/README.md`.

**If automatic fetching is not needed**, skip to Step 1 to create the index manually.

---

## Step 1: Understand Requirements

Before creating the index, clarify:

**Collection size:**
- Small (< 50 repos): Use minimal template
- Medium (50-200 repos): Use standard template  
- Large (200+ repos): Use detailed template with multi-view organization

**Organization approach:**
- By technology domain (AI/ML, Web, Backend, etc.)
- By purpose (Tools, Learning, Frameworks, etc.)
- By programming language
- Mixed/custom categorization

**Detail level:**
- Minimal: Name, Description, Link
- Standard: + Stars, Language, Last Update
- Detailed: + Forks, License, Tags, Additional metadata

**Ask the user** if any of these aspects are unclear.

## Step 2: Choose and Initialize Template

Based on the collection size determined in Step 1, copy the appropriate template from `assets/index_template.md`:

**Template structure:**
```markdown
# Title
> Metadata line with update date

## About section
- Statistics
- Update frequency

## Table of Contents
- Category links

## Categories
### Subcategory
| Name | Description | Stars | Language | Last Update | Link |
|------|-------------|-------|----------|-------------|------|
| ... | ... | ... | ... | ... | ... |
```

**Customize placeholders:**
- `{DATE}`: Current date (format: YYYY-MM-DD)
- `{TOTAL_COUNT}`: Total number of repositories
- `{CATEGORY_COUNT}`: Number of categories
- `{UPDATE_FREQUENCY}`: How often the index will be updated

For detailed structure guidance and more examples, see `references/structure_guide.md` and `references/examples.md`.

## Step 3: Define Categories

Create a logical category structure based on the organization approach:

**Technology domain approach (recommended for diverse collections):**
```
AI/Machine Learning
├── LLM Tools
├── Deep Learning Frameworks
└── Computer Vision

Web Development
├── Frontend Frameworks
├── UI Components
└── Build Tools

Backend Development
├── Web Frameworks
├── API Development
└── Database Tools
```

**Purpose-based approach (good for learning collections):**
```
Learning Resources
├── Tutorials
├── Example Projects
└── Books

Development Tools
├── CLI Tools
├── Editor Plugins
└── Productivity

Frameworks & Libraries
└── ...
```

**Guideline:** Keep 2-3 hierarchy levels maximum (Category → Subcategory → Repositories).

## Step 4: Populate Repository Information

For each repository in a category, create a table row with the following fields:

### Required Fields

**Name** - Repository name (concise, recognizable)
- Use the actual repo name or a shortened version
- Example: `langchain`, `next.js`, `tensorflow`

**Description** - One-line summary (20-50 characters)
- Focus on what it does, not technical details
- Examples:
  - ✅ "React framework for production"
  - ✅ "Build LLM applications"
  - ❌ "A JavaScript library that..." (too verbose)

**Link** - GitHub repository URL
- Format: `[GitHub](https://github.com/user/repo)`
- Or with custom text: `[View](https://github.com/user/repo)`

### Recommended Fields

**Stars** - GitHub stars count with icon
- Format: `⭐ 12.5k` (use k for thousands)
- Formatting rules:
  - < 1000: Show full number (⭐ 856)
  - 1000-9999: One decimal + k (⭐ 1.2k, ⭐ 9.8k)
  - 10000+: One decimal + k (⭐ 12.5k, ⭐ 156.3k)

**Language** - Primary programming language
- Use full names: Python, JavaScript, TypeScript, Go, Rust
- For multi-language: List main language or use "Multiple"
- For documentation: Use "-" or "Markdown"

**Last Update** - Recent update timestamp
- Format options:
  - ISO: `2024-01-15`
  - Short: `2024-01`
  - Relative: `2 days ago`, `1 week ago`

### Optional Fields

- **Forks**: `🔱 2.3k`
- **License**: `MIT`, `Apache-2.0`, `GPL-3.0`
- **Tags**: `🔥 Active`, `⭐ Recommended`, `🆕 New`

### Example Table

```markdown
| Name | Description | Stars | Language | Last Update | Link |
|------|-------------|-------|----------|-------------|------|
| langchain | Build LLM applications | ⭐ 85.2k | Python | 2024-02 | [GitHub](https://github.com/langchain/langchain) |
| next.js | React framework for production | ⭐ 118.5k | TypeScript | 2024-02 | [GitHub](https://github.com/vercel/next.js) |
```

## Step 5: Enhance and Refine

After populating the basic content, enhance the document:

**Add visual elements:**
- Section emojis (📚, 💻, 🔧, etc.)
- Status indicators (🔥 Active, 🌟 Popular)
- Separators (horizontal rules between major sections)

**Improve navigation:**
- Add table of contents with anchor links
- Keep TOC in sync with section headings
- Use consistent heading levels

**Include metadata sections:**
- Statistics summary (language breakdown, stars distribution)
- Update log (recent changes)
- Usage instructions

**Ensure consistency:**
- Uniform date formats
- Consistent star count formatting
- Aligned table columns
- Same icon style throughout

## Sorting Recommendations

Within each category, sort repositories by:

1. **Stars (descending)** - Most popular first (recommended)
2. **Update time (descending)** - Most recently updated first
3. **Alphabetical** - By name
4. **Custom** - By personal preference or importance

Choose one sorting method and apply consistently within each category.

## Best Practices

**Content quality:**
- Write clear, concise descriptions (avoid filler words)
- Verify all links are correct and accessible
- Use accurate, up-to-date star counts
- Keep language names consistent

**Structure:**
- Limit categories to 8-12 for maintainability
- Keep subcategories focused (5-15 repos each)
- Use descriptive category names
- Add brief category descriptions when helpful

**Formatting:**
- Keep tables aligned and readable
- Use consistent spacing between sections
- Don't over-use emojis (2-3 types maximum)
- Maintain uniform column order

**Maintenance:**
- Include last update date prominently
- Add changelog section for tracking changes
- Consider automation for large collections
- Review and update quarterly

## Scripts and Automation

**Fetch script:** `scripts/fetch_github_stars.py`
- Automatically fetch starred repositories from GitHub API
- Generate JSON data or Markdown index directly
- Support filtering, grouping, and customization
- See `scripts/README.md` for detailed usage

**When to use the script:**
- User has a GitHub account with starred repos
- Need to fetch latest data automatically
- Want to update an existing index
- Large collection (100+ repos) that would be tedious to enter manually

**When to skip the script:**
- User doesn't have GitHub account
- Creating index for curated/selected repos only
- Small collection (<20 repos) easy to enter manually
- Need custom categorization not supported by auto-grouping

## Templates and References

**Template file:** `assets/index_template.md`
- Complete working template with all sections
- Includes placeholders for easy customization
- Copy and modify as needed

**Structure guide:** `references/structure_guide.md`
- Detailed explanation of document hierarchy
- Table field specifications and formats
- Classification strategies
- Data formatting standards

**Examples:** `references/examples.md`
- 5 complete example indexes for different scenarios
- Minimal, standard, detailed, and specialized formats
- Multi-view organization examples
- Real-world patterns and layouts

**When to read references:**
- Structure guide: When planning document organization or need formatting details
- Examples: When seeking inspiration or comparing different styles
- Script README: When user needs to fetch data from GitHub API

## Common Scenarios

**Scenario 1: Auto-generate index from GitHub stars**
1. Ask user for GitHub token (guide them to get one if needed)
2. Run `fetch_github_stars.py` with appropriate options
3. Review generated index and suggest improvements
4. Optionally reorganize categories or add custom sections
5. Add custom descriptions or commentary

**Scenario 2: Creating first index from scratch (manual)**
1. Ask user about collection size and preferred organization
2. Copy template from `assets/index_template.md`
3. Set up 4-6 main categories based on their focus areas
4. Fill in repositories with standard fields (name, description, stars, link)
5. Add TOC and finalize

**Scenario 3: Reorganizing existing list**
1. Review current organization and identify issues
2. Propose improved category structure
3. Migrate content to new structure
4. Standardize table formats
5. Add missing metadata

**Scenario 4: Updating existing index**
1. Option A: Re-run `fetch_github_stars.py` to get latest data
2. Option B: Manually refresh star counts and update dates
3. Add new repositories in appropriate categories
4. Remove archived or deprecated projects
5. Update changelog section

**Scenario 5: Converting from simple list to structured index**
1. Analyze the simple list to identify natural groupings
2. Choose template based on list size
3. Create category structure
4. Convert list items to table rows with additional metadata
5. Add TOC, statistics, and documentation sections

## Output Format

Always generate the complete markdown document, not just snippets. The output should be:

- **Valid Markdown**: Properly formatted with correct syntax
- **Ready to use**: Can be directly saved as README.md or similar
- **Well-structured**: Proper heading hierarchy, aligned tables
- **Complete**: Includes all sections (header, TOC, categories, footer)

When the document is lengthy (>500 lines), confirm with the user whether to:
- Output the full document
- Split into multiple files
- Provide section by section

## Quick Reference

**Field formats:**
- Stars: `⭐ 12.5k` (k for thousands, one decimal)
- Link: `[GitHub](https://github.com/user/repo)`
- Date: `2024-02` or `2024-02-04`
- Language: Full name (Python, JavaScript)

**Common emojis:**
- 📚 Documentation/Learning
- 💻 Development
- 🔧 Tools
- 🤖 AI/ML
- 🌐 Web
- 🔥 Active/Hot
- ⭐ Recommended
- 🆕 New

**Template locations:**
- Main template: `assets/index_template.md`
- Structure guide: `references/structure_guide.md`
- Examples: `references/examples.md`
