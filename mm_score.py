import json
import requests

BASEURL = "http://172.26.38.244"
USER = "sasdemo"
PWD = "sas123"
MODNAME = "HMEQ_LOGI_0225"


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
            "name": "LOAN",
            "value": 1700
        },
        {
            "name": "MORTDUE",
            "value": 97800
        },
        {
            "name": "VALUE",
            "value": 112000
        },
        {
            "name": "REASON",
            "value": "HomeImp"
        },
        {
            "name": "JOB",
            "value": "Office"
        },
        {
            "name": "YOJ",
            "value": 3
        },
        {
            "name": "DEROG",
            "value": 0
        },
        {
            "name": "DELINQ",
            "value": 0
        },
        {
            "name": "CLAGE",
            "value": 93.33333
        },
        {
            "name": "NINQ",
            "value": 0
        },
        {
            "name": "CLNO",
            "value": 14
        },
        {
            "name": "DEBTINC",
            "value": 0
        }
    ]
}

res = requests.post(URL, headers=headers, data=json.dumps(parms)).json()
print(res)
