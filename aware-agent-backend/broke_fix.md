### ✅ **Agent Instruction: Python Project Refactor**

#### 🎯 **Goal:**
Reorganize the `src/` folder of the Python project for better maintainability and modularity. Ensure no import reference errors exist after refactoring.

---

### 📁 **Step 1: Analyze Project Structure**
- Scan the entire `src/` directory to identify:
  - Modules with redundant or duplicate code
  - Utility functions scattered across files
  - Unstructured scripts (e.g., directly under `src/`)
  - Business logic vs helper code
  - Cyclical or broken import references

---

### 📂 **Step 2: Recommended Folder Structure**
Restructure code into these standard folders:

```plaintext
src/
│
├── __init__.py
├── main.py                # Entry point if applicable
│
├── core/                 # Business logic
│   ├── __init__.py
│   ├── services/
│   └── models/
│
├── utils/                # Shared helpers/utilities
│   ├── __init__.py
│   └── logger.py
│   └── validators.py
│
├── config/               # Configuration files
│   ├── __init__.py
│   └── settings.py
│
├── data/                 # Data input/output handlers
│   ├── __init__.py
│   └── data_loader.py
│
├── tests/                # Test suite
│   ├── __init__.py
│   └── test_*.py
│
└── common/               # Reusable common logic
    ├── __init__.py
    └── constants.py
    └── exceptions.py
```

---

### 🧠 **Step 3: Consolidate Common Code**
- Move repeated logic (e.g., date functions, parsing logic, constants) into `common/` or `utils/`.
- Move any duplicated configuration or global variables to `config/settings.py`.

---

### 🧼 **Step 4: Fix All Imports**
- Replace relative imports with **absolute imports** rooted from `src/`, unless it's a package/module-level import.
- Update all `from .x import y` or `from ..x import y` to clean absolute references:
  
  For example:
  ```python
  # Before
  from ..utils import logger

  # After
  from src.utils import logger
  ```

---

### ✅ **Step 5: Validation**
- Check and fix all circular or broken imports
- Run a dry test:
  - `python -m compileall src/` to verify syntax
  - Run linter (e.g., `pylint`, `flake8`) and fix import issues
  - Run all unit tests under `tests/`

