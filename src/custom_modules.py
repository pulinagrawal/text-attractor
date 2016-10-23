from time import sleep

from simple_youbot_nav.msg import WheelCommand
from numpy import average
from smach import State, StateMachine

from lidapy.framework.msg import built_in_topics, FrameworkTopic
from lidapy.framework.shared import CognitiveContent
from lidapy.framework.module import FrameworkModule
from lidapy.module.action_selection import ActionSelection
from lidapy.module.current_situational_model import CurrentSituationalModel
from lidapy.module.global_workspace import GlobalWorkspace
from lidapy.module.perceptual_associative_memory import PerceptualAssociativeMemory
from lidapy.module.procedural_memory import ProceduralMemory
from lidapy.module.sensory_memory import SensoryMemory
from lidapy.module.sensory_motor_memory import SensoryMotorMemory
from lidapy.module.workspace import Workspace
from lidapy.framework.strategy import LinearExciteStrategy
from lidapy.util import logger
from std_msgs.msg import String

from sensor_msgs.msg import LaserScan

# Topic definitions
TEXTSCAN_TOPIC = FrameworkTopic("/text_attractor/env", String)
DORSAL_STREAM_TOPIC = built_in_topics["dorsal_stream"]
VENTRAL_STREAM_TOPIC = built_in_topics["ventral_stream"]
DETECTED_FEATURES_TOPIC = built_in_topics["detected_features"]
PERCEPTS_TOPIC = built_in_topics["percepts"]
WORKSPACE_COALITIONS_TOPIC = built_in_topics["workspace_coalitions"]
GLOBAL_BROADCAST_TOPIC = built_in_topics["global_broadcast"]
CANDIDATE_BEHAVIORS_TOPIC = built_in_topics["candidate_behaviors"]
SELECTED_BEHAVIORS_TOPIC = built_in_topics["selected_behaviors"]

ENVIRONMENT_MODULE = "environment"

class TextAttractorEnvironment(FrameworkModule):
    def __init__(self, **kwargs):
        super(FrameworkModule, self).__init__(ENVIRONMENT_MODULE,**kwargs)

    def add_publisher(self, topic):
        super(FrameworkModule, self).add_publisher(TEXTSCAN_TOPIC)

    def call(self):
        self.publish(TEXTSCAN_TOPIC, "attract text")


class BasicSensoryMemory(SensoryMemory):
    def __init__(self, **kwargs):
        super(BasicSensoryMemory, self).__init__(**kwargs)

    def add_subscribers(self):
        super(BasicSensoryMemory, self).add_subscribers()
        super(BasicSensoryMemory, self).add_subscriber(TEXTSCAN_TOPIC)

    def get_next_msg(self, topic):
        return super(BasicSensoryMemory, self).get_next_msg(topic)

    def publish(self, topic, msg):
        super(BasicSensoryMemory, self).publish(topic, msg)

    def call(self):
        text = self.get_next_msg(TEXTSCAN_TOPIC)

        for x in text:
            self.publish(DETECTED_FEATURES_TOPIC, CognitiveContent(x))

class TextAttractorCSM(CurrentSituationalModel):
    nodes = {}

    def receive_add_csm_content_request(self, request):
        response = super(TextAttractorCSM, self).receive_add_csm_content_request(request)

        les = LinearExciteStrategy(.5)
        if self.nodes[request] is KeyError:
            self.nodes[request] = .1
        les.apply(self.nodes[request], 1)

        return response

    def call(self):
        self.publish(GLOBAL_BROADCAST_TOPIC, max(self.nodes))

class TextAttractorProceduralMemory(ProceduralMemory):
    SCHEMES = {"a": "an amazing adventure",
               "b": "because birthdays blast",
               "c": "can cats crawl",
               "d": "do dogs dump",
               "e": "every eel etches",
               "f": "forever fumbling farts",
               "g": "got good grammar",
               "h": "how he hops",
               "i": "i in ink",
               "j": "jockies just jam",
               "k": "kittens kill kites",
               "l": "llama legs laugh",
               "m": "moth mates moth",
               "n": "nobody never nags",
               "o": "onward on oocyte",
               "p": "please push pulin",
               "q": "qualms quarry quest",
               "r": "real roasting rant",
               "s": "say something savvy",
               "t": "todd told tales",
               "u": "until uranus usurped",
               "v": "victory vouches valor",
               "w": "wondering wind wallops",
               "x": "x xanthanic xenophobe",
               "y": "yelled yolo yesterday",
               "z": "zen's zero zone"
               }

    def call(self):
        letter = self.get_next_msg(GLOBAL_BROADCAST_TOPIC)
        self.publish(CANDIDATE_BEHAVIORS_TOPIC, self.SCHEMES[letter])


class BasicSensoryMotorMemory(SensoryMotorMemory):
    def __init__(self, **kwargs):
        super(BasicSensoryMotorMemory, self).__init__(**kwargs)

        self.stateMachine = self.create_state_machine()

    def add_publishers(self):
        super(BasicSensoryMotorMemory, self).add_publisher(TEXTSCAN_TOPIC)

    def get_next_msg(self, topic):
        return super(SensoryMotorMemory, self).get_next_msg(topic)

    def publish(self, topic, msg):
        super(SensoryMotorMemory, self).publish(topic, msg)

    def call(self):
        behavior = self.get_next_msg(SELECTED_BEHAVIORS_TOPIC)
        self.publish(TEXTSCAN_TOPIC, behavior)
