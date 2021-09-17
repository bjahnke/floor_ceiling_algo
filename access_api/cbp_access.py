"""
Interface for coinbase pro library
"""
import coinbase_pro as cbpro
import access_api.creds.cbpro_creds as cbpro_creds

client = cbpro.CoinbaseProFactory().get_client(
    key=cbpro_creds.key,
    secret=cbpro_creds.b64secret,
    passphrase=cbpro_creds.passphrase
)


class CbproLocalClient:
    def __init__(self):
        super().__init__()