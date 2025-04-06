def test_group(name, imports):
    print(f"\nTesting {name}...")
    for module in imports:
        try:
            exec(f"import {module}")
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")

# Core dependencies
core_deps = ["langchain", "fastapi", "uvicorn", "spacy", "numpy"]
test_group("Core Dependencies", core_deps)

# Async libraries
async_deps = ["aiohttp", "asyncio", "aiosqlite", "asyncpg", "websockets"]
test_group("Async Libraries", async_deps)

# Database and ORM
db_deps = ["sqlalchemy", "alembic"]
test_group("Database and ORM", db_deps)

# Testing and Development
test_deps = ["pytest", "black", "flake8", "mypy", "pylint"]
test_group("Testing and Development Tools", test_deps)

# Documentation and Monitoring
doc_deps = ["sphinx", "structlog", "prometheus_client", "psutil"]
test_group("Documentation and Monitoring", doc_deps)

# Data Science and ML
ds_deps = ["chromadb", "sentence_transformers", "transformers", "tensorflow"]
test_group("Data Science and ML", ds_deps)

# Data Analysis and Visualization
viz_deps = ["networkx", "matplotlib", "plotly", "seaborn", "wordcloud"]
test_group("Data Analysis and Visualization", viz_deps)

# Utilities
util_deps = ["anytree", "httpx", "markdown", "scholarly", "statsmodels", "textstat", 
            "python_louvain", "leidenalg", "prophet", "feedparser", "boto3"]
test_group("Utilities", util_deps)

print("\nImport testing completed!") 