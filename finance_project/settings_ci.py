# finance_project/settings_ci.py

from .settings import *  # Import base settings

# Override the DATABASES setting to use SQLite for testing
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',  # Use in-memory database for tests
    }
}

# Disable any settings that rely on external services
# For example, turn off debug mode if needed
DEBUG = False

# Adjust any other settings as necessary for the test environment
