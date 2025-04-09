✅ Final Refined AI Instruction Set  
**“Smart Incremental Error & Coverage Fixer”**

---

## 🧠 **Session Setup (Only Once Per Run)**

### 1. **Check & Cache Toolchain (Once Per Session)**

```bash
# Backend (Python)
which python || which python3
python --version
pip list | grep coverage
pip list | grep pytest

# Frontend (Node/JS)
which node
node -v
which npm || which yarn
npm list --depth=0 | grep jest


# Frontend
which node > .ai_log/tools_checked_frontend || echo "Node.js not found" > error.log
which npm >> .ai_log/tools_checked_frontend
npm list --depth=0 >> .ai_log/tools_checked_frontend

# Backend
which python > .ai_log/tools_checked_backend || echo "Python not found" > error.log
pip list >> .ai_log/tools_checked_backend
```

> ✅ If `.ai_log/tools_checked_*` exists → **Skip tool check**  
> ✅ If not → run and save to skip in the next iterations

---

### 2. **Initial Coverage Scan (Once Per Project)**
> Only if `.ai_log/coverage/*_todo.txt` does not exist

**Backend**
```bash
python -m coverage run -m pytest
python -m coverage json -o .ai_log/backend_coverage.json
# Parse and create .ai_log/coverage/backend_todo.txt
```

**Frontend**
```bash
npm run test -- --coverage --json --outputFile=.ai_log/frontend_coverage.json
# Parse and create .ai_log/coverage/frontend_todo.txt
```

> ✅ Save only **files < 100%**  
> ✅ Files removed once done — progress is remembered

---

## 🔁 **Iterative Phase: One File at a Time**

### 3. **Iterate Over TODO Files**

**Backend Example:**
```bash
file=$(head -n 1 .ai_log/coverage/backend_todo.txt)
python -m coverage run -m pytest "$file"
python -m coverage report --include="$file"

# If coverage good → remove from list
```

**Frontend Example:**
```bash
file=$(head -n 1 .ai_log/coverage/frontend_todo.txt)
npx jest "$file" --coverage --collectCoverageFrom="['$file']"

# If test passes and coverage ok → remove from list
```

---

## 📌 **Key Behaviors**

| Action              | When Triggered                | Notes                              |
|---------------------|-------------------------------|-------------------------------------|
| Tool Check          | First iteration only          | Uses `.ai_log/tools_checked_*`      |
| Full Coverage       | Only if no `*_todo.txt`       | One-time generation of tasks        |
| File Test           | One file per loop             | Keeps coverage focused              |
| File Removal        | Only if coverage passes       | Clean, incremental progress         |
| Final Coverage      | Only when `*_todo.txt` is empty | Final confidence sweep              |

---

## ✅ Final Coverage Recheck (Once at the End)

```bash
# Backend
python -m coverage run -m pytest
python -m coverage report --fail-under=95

# Frontend
npm run test -- --coverage
```

---

### 🧠 Agent Summary Example

```
[INIT] Loaded toolchain from .ai_log/tools_checked_backend
[SCAN] 12 backend files below threshold → saved to backend_todo.txt
[FIX] Running coverage for app/api/users.py...
[PASS] Coverage now 97%. Removing from list.
[REMAINING] 11 files to go...
```

