# Development Environment Setup Guide

This guide will help you set up the development environment for the Aware Agent project.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Python 3.8 or later](https://www.python.org/downloads/)
- [Node.js 18.x or later](https://nodejs.org/)
- [Git](https://git-scm.com/downloads)
- [PowerShell 7 or later](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows) (recommended)

## Initial Setup

1. **Clone the Repository**
   ```powershell
   git clone https://github.com/yourusername/aware-agent.git
   cd aware-agent
   ```

## Backend Setup

1. **Create and Activate Virtual Environment**
   ```powershell
   cd aware-agent-backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   ```powershell
   # Copy the example environment file
   Copy-Item .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize Database**
   ```powershell
   # Run database migrations
   alembic upgrade head
   ```

## Frontend Setup

1. **Install Dependencies**
   ```powershell
   cd ..\aware-agent-frontend
   npm install
   ```

2. **Set Up Environment Variables**
   ```powershell
   # Copy the example environment file
   Copy-Item .env.example .env
   # Edit .env with your configuration
   ```

## Running the Application

1. **Start Backend Server**
   ```powershell
   cd aware-agent-backend
   .\venv\Scripts\Activate.ps1
   python run.py
   ```

2. **Start Frontend Development Server**
   ```powershell
   cd aware-agent-frontend
   npm run dev
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Testing Setup

1. **Backend Tests**
   ```powershell
   cd aware-agent-backend
   .\venv\Scripts\Activate.ps1
   pytest
   ```

2. **Frontend Tests**
   ```powershell
   cd aware-agent-frontend
   # Run Jest tests
   npm test
   # Run Cypress tests
   npm run cypress:open
   ```

## Development Tools

### Code Quality Tools

1. **Backend**
   - Black (code formatting)
   - Flake8 (linting)
   - MyPy (type checking)

2. **Frontend**
   - ESLint (linting)
   - Prettier (code formatting)
   - TypeScript (type checking)

### IDE Setup

Recommended IDE: [Visual Studio Code](https://code.visualstudio.com/)

Recommended Extensions:
- Python
- ESLint
- Prettier
- GitLens
- Docker
- REST Client

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Run PowerShell as Administrator
   - Ensure proper file permissions

2. **Dependency Installation Issues**
   - Clear npm cache: `npm cache clean --force`
   - Delete node_modules and reinstall
   - Update pip: `python -m pip install --upgrade pip`

3. **Database Connection Issues**
   - Verify database credentials in .env
   - Ensure database service is running
   - Check network connectivity

## Additional Resources

- [Python Documentation](https://docs.python.org/3/)
- [Node.js Documentation](https://nodejs.org/en/docs/)
- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Support

If you encounter any issues during setup, please:
1. Check the troubleshooting section
2. Search for similar issues in the project's issue tracker
3. Create a new issue with detailed information about your problem 