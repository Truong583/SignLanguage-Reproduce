"""Fail closed: never initialize telemetry in a missing or publicly visible project."""
import base64
import json
import urllib.request


def require_private(entity,project,key,opener=urllib.request.urlopen):
    body={'query':'query Privacy($entity: String!, $name: String!) { project(entityName: $entity, name: $name) { name access } }',
          'variables':{'entity':entity,'name':project}}
    auth=base64.b64encode(('api:'+key).encode()).decode()
    request=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps(body).encode(),
        headers={'Authorization':'Basic '+auth,'Content-Type':'application/json'})
    with opener(request,timeout=30) as response: result=json.load(response)
    value=(result.get('data') or {}).get('project')
    if result.get('errors') or not value or str(value.get('access','')).upper() not in {'PRIVATE','TEAM','RESTRICTED'}:
        raise RuntimeError('W&B project must already exist and be Private/Team/Restricted. Telemetry was not started.')
    return value
