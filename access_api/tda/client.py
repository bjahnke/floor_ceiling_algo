"""

"""

import tda
import selenium.webdriver
from access_api.creds import tda_creds
from access_api.factory import (
    AbstractClient,
    AbstractFactory,
    Dict,
)


class TdaClient(AbstractClient):
    def __init__(self, tda_client, account_id):
        self._client = tda_client
        self._account_id = account_id
        self._name = 'TD Ameritrade'

    @property
    def name(self):
        return self._name

    def products(self) -> Dict:
        pass

    def order(self, data: dict) -> dict:
        pass

    




class TdaFactory(AbstractFactory):
    def get_client(
            self,
            api_key: str,
            redirect_uri: str,
            token_path: str,
            account_id: int
    ) -> AbstractClient:
        client = tda.auth.easy_client(
            webdriver_func=selenium.webdriver.Firefox,
            api_key=tda_creds.api_key,
            token_path=tda_creds.token_path,
            redirect_uri=tda_creds.redirect_uri
        )
        return TdaClient
