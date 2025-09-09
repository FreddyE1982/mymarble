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
        if len(tokens) < 3:
            raise ValueError('Token stream too short')
        index = 1
        if tokens[1] == int(ControlToken.RST):
            index = 2
        if tokens[index] != int(ControlToken.SYNC):
            raise ValueError('Missing SYNC after BOS')
        Reporter.report('validated_tokens', 'Number of tokens validated', len(tokens))
        return True
