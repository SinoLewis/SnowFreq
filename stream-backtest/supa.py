from supabase import create_client, Client
from dotenv import dotenv_values

env_vars = dotenv_values('.env')
url: str = env_vars["SUPABASE_URL"]
key: str = env_vars["SUPABASE_ANON_KEY"]
supaclient: Client = create_client(url, key)

# __all__ = ['supaclient']