# Incremental Code Coverage Manager

This tool implements an optimized incremental code coverage strategy for both frontend and backend code. It follows the process outlined in `fix_flow.md`.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following installed:
- Python 3.8+
- Node.js and npm
- pytest
- coverage

## Usage

Run the coverage manager:
```bash
python coverage_manager.py
```

The script will:
1. Run initial coverage scan for both frontend and backend
2. Parse coverage results and create todo lists
3. Process files one by one
4. Run final validation

## Directory Structure

- `.coverage_todo/`
  - `backend_files.txt` - List of backend files needing coverage
  - `frontend_files.txt` - List of frontend files needing coverage
  - `backend_coverage.json` - Backend coverage data
  - `frontend_coverage.json` - Frontend coverage data
  - `coverage_manager.py` - Main script
  - `requirements.txt` - Python dependencies

## How It Works

1. **Initial Scan**: Runs full coverage once for both frontend and backend
2. **Todo Lists**: Creates lists of files that need coverage
3. **Incremental Testing**: Tests files one by one
4. **Validation**: Runs final full coverage check

## Notes

- The script assumes your frontend and backend directories are named `aware-agent-frontend` and `aware-agent-backend` respectively
- Coverage data is stored in JSON format for easy parsing
- The script maintains separate todo lists for frontend and backend 