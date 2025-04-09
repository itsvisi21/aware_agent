### **Instruction Set: Achieve Complete Code Coverage for Frontend and Backend Projects**

#### **Scope**:
- Full-stack project with separate frontend and backend
- Goal: Reach high unit and integration test coverage (>=90% if possible)
- Environments:
  - **Frontend**: React / Angular / Vue (JavaScript or TypeScript)
  - **Backend**: FastAPI / Django / Flask (Python)

---

### **Global Steps for Both Projects**

1. **Initialize Code Coverage Tools**
   - Frontend: `Jest`, `React Testing Library`, `Cypress`, `Karma`, or similar
   - Backend: `pytest`, `coverage.py`, `unittest`, or `tox`

2. **Identify Test Gaps**
   - Analyze existing test suites.
   - Use tools to generate coverage reports:
     - Frontend: `npm run test -- --coverage`
     - Backend: `coverage run -m pytest`

3. **Log Uncovered Code**
   - Generate and save HTML reports:
     ```bash
     coverage html
     ```
     or
     ```bash
     npm run test -- --coverage && open coverage/lcov-report/index.html
     ```

---

### **Frontend Code Coverage Steps**

#### 1. **Set Up Testing Frameworks**
- If not installed:
  ```bash
  npm install --save-dev jest @testing-library/react @testing-library/jest-dom
  ```

#### 2. **Write/Improve Unit Tests**
- Create `*.test.js` or `*.spec.ts` files next to components.
- Use mocks to isolate component behavior.
- Test:
  - UI states
  - Component props
  - Event handlers (clicks, inputs)
  - Redux/state updates if used

#### 3. **Add Integration Tests**
- Use Cypress or Playwright for UI integration testing:
  ```bash
  npx cypress open
  ```

#### 4. **Run and Review Coverage**
  ```bash
  npm run test -- --coverage
  ```

#### 5. **Improve Coverage**
- Focus on files <90% coverage
- Add snapshot & render tests

---

### **Backend Code Coverage Steps**

#### 1. **Install Coverage Tools**
  ```bash
  pip install pytest coverage
  ```

#### 2. **Run Tests with Coverage**
  ```bash
  coverage run -m pytest
  coverage report
  coverage html
  ```

#### 3. **Create Tests for:**
- Routes (GET, POST, etc.)
- Services / business logic
- Model validators
- Error handling
- Security/auth flows

#### 4. **Advanced: Use Parametrize**
- Use `pytest.mark.parametrize` to test multiple inputs in one function.

#### 5. **Mock External Services**
- Use `unittest.mock` or `pytest-mock` to isolate components.

---

### **Automated Validation & CI Integration**

1. **CI Tooling (GitHub Actions / GitLab / Jenkins)**  
   Add steps to pipeline:
   ```yaml
   - name: Run Backend Tests
     run: |
       pip install -r requirements.txt
       coverage run -m pytest
       coverage report
   - name: Run Frontend Tests
     run: |
       npm ci
       npm run test -- --coverage
   ```

2. **Fail Build if Coverage Falls Below Threshold**
   ```bash
   coverage report --fail-under=90
   ```

---

### **Documentation and Reports**
- Store coverage reports in `/coverage` folder
- Commit only summaries to Git
- Add README badge via tools like Codecov

