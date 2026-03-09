# Agent Skills

**Path**: `skills/`  
**Type**: Agent Skill Collection  
**Pattern**: SKILL.md + references/ + scripts/

## OVERVIEW

Modular agent skills with progressive context loading. Each skill provides domain-specific expertise through structured documentation, reference materials, and automation scripts.

## STRUCTURE

```
skills/
├── ai-product-manager/      # Product management skill
├── github-stars-indexer/    # GitHub stars indexing
├── java-ai-learning-planner/ # Learning path planner
└── skill-creator/           # Skill creation tools
```

## SKILL ORGANIZATION

Each skill follows this structure:
```
skill-name/
├── SKILL.md           # Main skill definition (required)
├── references/        # Domain knowledge (optional)
│   ├── *.md          # Reference documents
│   └── ...
├── scripts/           # Automation scripts (optional)
│   ├── *.py          # Python scripts
│   └── requirements.txt
└── assets/            # Output resources (optional)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Product management | `ai-product-manager/` | PRD, research, design |
| GitHub indexing | `github-stars-indexer/` | Stars organization |
| Learning paths | `java-ai-learning-planner/` | Java → AI transition |
| Skill creation | `skill-creator/` | Create new skills |
| Add skill | Create `skills/<name>/` | Follow SKILL.md pattern |

## PROGRESSIVE LOADING

Skills use 3-level context loading:

1. **Level 1**: Metadata (always loaded)
   - Name, description, trigger conditions

2. **Level 2**: SKILL.md (on trigger)
   - Workflows, guidelines, patterns

3. **Level 3**: References (on demand)
   - Detailed domain knowledge

## SKILL.MD PATTERN

```yaml
---
name: skill-name
description: When to use this skill (trigger conditions)
---

# Skill Title

## Overview
Brief description of what this skill does.

## Workflows

### Workflow 1
1. Step one
2. Step two
3. Step three

## References
- `reference-file.md` - Brief description
```

## CONVENTIONS

### File Naming
- `SKILL.md` - Required, main skill file
- `references/*.md` - Knowledge documents
- `scripts/*.py` - Executable scripts
- `assets/*` - Templates, examples

### Content Guidelines
- Keep SKILL.md under 500 lines
- Split detailed content to references/
- Include clear trigger conditions
- Provide concrete examples

### Script Requirements
- Include `requirements.txt`
- Test before committing
- Use clear argument parsing
- Return structured output

## COMMANDS

```bash
# Use skill scripts
cd skills/github-stars-indexer/scripts
python fetch_github_stars.py

# Create new skill
cd skills/skill-creator
python scripts/init_skill.py my-skill
```

## NOTES

- **Trigger-based**: Skills load based on description matching
- **Modular**: Each skill is self-contained
- **Versioned**: Skills can evolve independently
- **Reusable**: Share across projects
