INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'pattern_detection_app',
    'api_v1',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# CORS settings - completely permissive for demo
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_ALLOW_ALL = True
CORS_REPLACE_HTTPS_REFERER = True

# Allow all methods
CORS_ALLOW_METHODS = [
    '*',
]

# Allow all headers
CORS_ALLOW_HEADERS = [
    '*',
]

# Additional settings to ensure CORS works
CORS_EXPOSE_HEADERS = ['*']
CORS_PREFLIGHT_MAX_AGE = 86400  # 24 hours 