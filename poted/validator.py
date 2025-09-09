class ProtocolValidator:
    @classmethod
    def validate(cls, tokens):
        from .control import ControlToken
        from main import Reporter
        if not tokens:
            raise ValueError('Token stream empty')
        if tokens[0] != int(ControlToken.BOS):
            raise ValueError('Missing BOS')
        if tokens[-1] != int(ControlToken.EOS):
            raise ValueError('Missing EOS')
        if len(tokens) < 3 or tokens[1] != int(ControlToken.RST):
            raise ValueError('Missing RST after BOS')
        if tokens[2] != int(ControlToken.SYNC):
            raise ValueError('Missing SYNC after RST')
        Reporter.report('validated_tokens', 'Number of tokens validated', len(tokens))
        return True
