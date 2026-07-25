# Submission Staging

Do not place the report PDF here.

When the team selects its final agent, use the GitHub Actions **Package agent**
workflow or run:

```bash
python scripts/package_agent.py <agent_name>
```

The generated `dist/final-project-agent-code.zip` contains the selected agent
directory only. Inspect and test that archive in a clean copy of the official
framework before uploading it to MaMPF.
