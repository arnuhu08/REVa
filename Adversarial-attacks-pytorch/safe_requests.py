import requests
from urllib.parse import urlparse

DEFAULT_TIMEOUT = 15

class SafeSession(requests.Session):
    # resemble requests.Session but with safe defaults
    def __init__(self, trust_env=False):
        super().__init__()
        self.trust_env = trust_env

    def request(self, method, url, *args, **kwargs):
        # verify defaults to True
        if 'verify' not in kwargs:
            kwargs['verify'] = True
        return super().request(method, url, *args, **kwargs)

def _host_from_url(url):
    try:
        return urlparse(url).hostname
    except Exception:
        return None

def safe_request(method, url, *, auth=None, allowed_auth_hosts=None,
                verify=True, allow_redirects=True, timeout=DEFAULT_TIMEOUT,
                trust_env=False, **kwargs):

    session = SafeSession(trust_env=trust_env)
    try:
        if auth and allowed_auth_hosts:
            host = _host_from_url(url)
            if host not in set(allowed_auth_hosts):
                # if host is not allowed
                auth_to_use = None
            else:
                auth_to_use = auth
        else:
            auth_to_use = auth

        resp = session.request(
            method, url,
            auth=auth_to_use,
            verify=verify,
            allow_redirects=allow_redirects,
            timeout=timeout,
            **kwargs
        )
        return resp
    finally:
        session.close()

# wrappers:
def safe_get(url, **kwargs):
    return safe_request('GET', url, **kwargs)

def safe_post(url, data=None, json=None, **kwargs):
    return safe_request('POST', url, data=data, json=json, **kwargs)