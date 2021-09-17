# Ledger - A web application to track cryptocurrency investments
# Copyright (C) 2021 teleprint.me
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import abc
import requests
import typing

__agent__: str = 'teleprint.me'
__source__: str = 'https://github.com/teleprint-me/ledger'
__version__: str = '0.1.3'

__offset__: int = 50
__limit__: int = 250
__timeout__: float = 0.275

List = typing.TypeVar('List', list, list[dict])
Dict = typing.TypeVar('DList', dict, List)
Response = typing.TypeVar('Response', requests.Response, List[requests.Response])


class AbstractAuth(abc.ABC):
    @abc.abstractmethod
    def __init__(self, key: str, secret: str):
        pass


class AbstractQuery(abc.ABC):
    @property
    @abc.abstractmethod
    def endpoint(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def data(self) -> dict:
        pass


class AbstractAPI(abc.ABC):
    @property
    @abc.abstractmethod
    def version(self) -> int:
        """return the current rest api version"""
        pass

    @property
    @abc.abstractmethod
    def url(self) -> str:
        """return the rest api url"""
        pass

    @abc.abstractmethod
    def endpoint(self, value: str) -> str:
        """return a endpoint based on `value`"""
        pass

    @abc.abstractmethod
    def path(self, value: str) -> str:
        """return the full path based on url and endpoint `value`"""
        pass


class AbstractMessenger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, auth: AbstractAuth):
        pass

    @property
    @abc.abstractmethod
    def api(self) -> AbstractAPI:
        pass

    @property
    @abc.abstractmethod
    def auth(self) -> AbstractAuth:
        pass

    @property
    @abc.abstractmethod
    def session(self) -> requests.Session:
        pass

    @property
    @abc.abstractmethod
    def timeout(self) -> int:
        pass

    @abc.abstractmethod
    def get(self, query: AbstractQuery) -> Response:
        pass

    @abc.abstractmethod
    def post(self, query: AbstractQuery) -> Response:
        pass

    @abc.abstractmethod
    def page(self, query: AbstractQuery) -> Response:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class AbstractContext(abc.ABC):
    @property
    @abc.abstractmethod
    def response(self) -> Response:
        pass

    @property
    @abc.abstractmethod
    def id(self) -> str:
        pass

    @abc.abstractmethod
    def data(self) -> Dict:
        pass

    @abc.abstractmethod
    def error(self) -> bool:
        pass


class AbstractClient(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def messenger(self) -> AbstractMessenger:
        pass

    @abc.abstractmethod
    def products(self) -> Dict:
        pass

    @abc.abstractmethod
    def accounts(self) -> Dict:
        pass

    @abc.abstractmethod
    def history(self, product_id: str) -> Dict:
        pass

    @abc.abstractmethod
    def deposits(self, product_id: str) -> Dict:
        pass

    @abc.abstractmethod
    def withdrawals(self, product_id: str) -> Dict:
        pass

    @abc.abstractmethod
    def price(self, product_id: str) -> dict:
        pass

    @abc.abstractmethod
    def order(self, data: dict) -> dict:
        pass


class AbstractFactory(abc.ABC):
    @abc.abstractmethod
    def get_client(self, *args, **kwargs) -> AbstractClient:
        pass
