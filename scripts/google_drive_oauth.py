"""One-time local OAuth helper for private Drive access from Kaggle.

Run on the owner's trusted computer; never commit the generated credential file.
"""
import argparse
import json
from pathlib import Path

SCOPE='https://www.googleapis.com/auth/drive'

def main():
    parser=argparse.ArgumentParser(description="Create an OAuth refresh token for this user's private Google Drive")
    parser.add_argument('--client-secrets',type=Path,required=True,help='OAuth Desktop app client_secret JSON')
    parser.add_argument('--output',type=Path,default=Path('gdrive_oauth_credentials.json'))
    args=parser.parse_args()
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow=InstalledAppFlow.from_client_secrets_file(str(args.client_secrets),scopes=[SCOPE])
    credentials=flow.run_local_server(host='localhost',port=0,access_type='offline',prompt='consent')
    args.output.write_text(credentials.to_json(),encoding='utf-8')
    args.output.chmod(0o600)
    print(f'Created private OAuth credentials: {args.output.resolve()}')
    print('Add this JSON as a private Kaggle Secret named GOOGLE_DRIVE_OAUTH_JSON. Never upload it as notebook output.')

if __name__=='__main__': main()
