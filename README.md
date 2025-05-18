# Full Stack Application

This repository contains a full-stack application with a Django backend and Next.js frontend.

## Project Structure

```
.
├── Backend/         # Django backend application
└── Frontend/        # Next.js frontend application
```

## Setup Instructions

### Backend (Django)

1. Navigate to the Backend directory:

   ```bash
   cd Backend
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # OR
   # source venv/bin/activate  # On Unix/MacOS
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run migrations:

   ```bash
   python manage.py migrate
   ```

5. Start the development server:
   ```bash
   python manage.py runserver
   ```

### Frontend (Next.js)

1. Navigate to the Frontend directory:

   ```bash
   cd Frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

## Development

- Backend API runs on: http://localhost:8000
- Frontend runs on: http://localhost:3000

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

[Add your chosen license here]
