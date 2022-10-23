import json
import requests

BASEURL = "http://172.26.38.244"
USER = "sasdemo"
PWD = "sas123"
MODNAME = "LightGBM1022v1"


def get_token(url, username, password):
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': 'Basic c2FzLmVjOg=='
    }
    payload = 'grant_type=password&username='+username+'&password='+password
    response = requests.post(url+'/SASLogon/oauth/token',
                             data=payload, headers=headers)
    auth_json = json.loads(response.content.decode())
    return auth_json['access_token']


token = get_token(BASEURL, USER, PWD)

headers = {
    'Authorization': 'bearer ' + token,
    'Content-Type': 'application/json ; application/vnd.sas.microanalytic.module.step.input+json',
    'Access-Control-Allow-Origin': "*"
}

URL = ("%s/microanalyticScore/modules/%s/steps/score") % (BASEURL, MODNAME)

parms = {
    "version": 1,
    "inputs": [
        {
            "name": "A",
            "value": 1
        },
        {
            "name": "B",
            "value": 2
        },
        {
            "name": "C",
            "value": 3
        },
    ]
}

res = requests.post(URL, headers=headers, data=json.dumps(parms)).json()
print(res)
