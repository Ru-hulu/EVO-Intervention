import logging
from pathlib import Path
from mme_vla_suite.training import config
from mme_vla_suite.policies import policy_config
from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer
logging.basicConfig(level=logging.INFO)
p=policy_config.create_trained_policy(config.get_config('mme_vla_suite'),Path('/cache/robot-baselines/models/RoboMME/perceptual-framesamp-modul/79999'),seed=7)
WebsocketPolicyServer(policy=p,host='127.0.0.1',port=8015,metadata=p.metadata).serve_forever()
