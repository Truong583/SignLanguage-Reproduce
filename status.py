"""Read the most recent local run/update monitor; no process control or network."""
import datetime
import json
from pathlib import Path

if __name__=='__main__':
    path=Path(__file__).resolve().parent/'runs/status.json'
    if not path.exists():
        print('No monitored run yet. Start with python run.py.')
    else:
        value=json.loads(path.read_text())
        heartbeat=datetime.datetime.fromisoformat(value['heartbeat_utc'])
        age=(datetime.datetime.now(datetime.timezone.utc)-heartbeat).total_seconds()
        value['seconds_since_monitor_heartbeat']=round(age)
        value['monitor_may_have_stopped']=value['status']=='running' and age>120
        print(json.dumps(value,indent=2,ensure_ascii=False))
        if value['monitor_may_have_stopped']: print('Monitor heartbeat is old; inspect the server/session. This is not proof of a program error.')
