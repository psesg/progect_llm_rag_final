# -*- coding: utf-8 -*-

import os
import platform
import socket as sckt
import logging
import requests
import json

logger = logging.getLogger(__name__)

def get_giga_credentials() -> str:
    """
        get GigaChat credentials from environment variable `GIGACHAT_CREDENTIALS`
    """
    credentials = ''
    try:
        credentials = os.environ['GIGACHAT_CREDENTIALS']
    except KeyError:
        logger.critical('OS variable: GIGACHAT_CREDENTIALS not set')
    else:
        logger.info(f'OS variable: GIGACHAT_CREDENTIALS is set') # to: <{credentials}>
    finally:
        return credentials


def get_giga_url_access_mode() -> tuple[str, dict]:
    """
        get GigaChat url access point and dict of platform, type host, hostname  and type_giga_access
    """
    type_giga_access = ''
    type_host = ''
    url_oauth = ''
    plat = platform.system()
    hostname = sckt.gethostname()

    if plat == 'Windows': # Windows
        try:
            userdomain = os.environ['USERDOMAIN']
        except KeyError:
            pass
        else:
            if userdomain.upper() == 'SIGMA':
                type_giga_access = 'SIGMA'
                type_host = 'corporate'
            else:
                type_giga_access = 'ENABLER'
                type_host = 'private'

    if plat == 'Darwin':  #
        if hostname.startswith('cab-ws'):
            type_giga_access = 'SIGMA'
            type_host = 'corporate'
        else:
            type_giga_access = 'ENABLER'
            type_host = 'private'

    if plat == 'Linux':  #
       type_giga_access = 'ENABLER'
       type_host = 'private'

    if type_giga_access == 'SIGMA':
        url_oauth = 'https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth'
    if type_giga_access == 'ENABLER':
        url_oauth = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'


    access_mode = {
        "type_host": type_host,
        "hostname": hostname,
        "type_giga_access": type_giga_access,
        "plat": plat
    }
    logger.info(f'host type: "{type_host}"; host name: "{hostname}"; OS: "{plat}"; type_giga_access: "{type_giga_access}"; url_oauth: "{url_oauth}"')
    return url_oauth, access_mode


def get_giga_token_access(url_oauth :str, credentials :str) -> tuple[bool, str]:
    """
        get GigaChat url access point and dict of platform, type host, hostname  and type_giga_access
    """
    rc = False
    # url for get list models
    urlm = "https://gigachat.devices.sberbank.ru/api/v1/models"

    # try to get token from file
    if os.path.exists('tk.txt'):
        ftk = open('tk.txt', 't+r', encoding='utf-8')
        tk = ftk.readline().strip('\n')
        ftk.close()
    else:
        tk = ''
    # test that token is valid
    payload = {}
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {tk}'
    }
    response = requests.request("GET", urlm, headers=headers, data=payload,
                                verify=False, cert=False)
    if (response.status_code == 401 and
            (json.loads(response.text).get('message') == 'Token has expired') or
            (json.loads(response.text).get('message') == 'Unauthorized')):
        logger.info('token expired - will get new one...')
        payload_tok = {
            'scope': 'GIGACHAT_API_CORP'
        }
        headers_tok = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': '1aa07e53-26a5-46e5-bdc5-34238c732bda',
            'Authorization': f'Basic {credentials}'
        }
        response = requests.request("POST", url_oauth, headers=headers_tok, data=payload_tok, verify=False, cert=False)
        tk = json.loads(response.text).get('access_token')
        logger.info(f'got new access token') # :{tk}
        ftk = open('tk.txt', 't+w', encoding='utf-8')
        if tk is not None:
            ftk.writelines([tk])
            rc = True
        else:
            rc = False
        ftk.close()
    else:
        rc = True
    return rc, tk