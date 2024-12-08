#!/usr/bin/env python
import os
import sys
from django.core.wsgi import get_wsgi_application

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'finance_project.settings')

# Create a top-level WSGI application object for Vercel
app = get_wsgi_application()  # Vercel looks for 'app' or 'handler'

def main():
    """Run administrative tasks."""
    # If you're executing commands like `python manage.py migrate` locally
    # this will still work because __name__=="__main__" only on direct invocation
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    main()
