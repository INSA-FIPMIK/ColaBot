from new_objects import main_new_objects
from test_pixel import main_test_pixel
import json
from pyniryo import *
import math

def import_dict():
    # import the dictionnaries for color and shape from .json file
    tf = open("Dictionnary database.json", "r")
    dict_color, dict_shape = json.load(tf)
    
    return dict_color, dict_shape

def data():
    robot_ip_address = "192.168.245.199"  # Replace by robot ip address
    workspace = "workspace_picking"  # Name of your workspace

    observation_pose = PoseObject(  # position for the robot to watch the workspace
        x=0.20, y=0, z=0.36,
        roll=0.0, pitch=math.pi / 2, yaw=0.0,
    )

    drop_pose = PoseObject(  # position for the robot to drop the object
        x=0.20, y=0.20, z=0.10,
        roll=0.0, pitch=math.pi / 2, yaw=0.0,
    )
    
    return robot_ip_address, workspace, observation_pose, drop_pose

def connect(robot_ip_address, observation_pose):
    # Connecting to robot
    client = NiryoRobot(robot_ip_address)
    client.calibrate(CalibrateMode.AUTO)
    client.update_tool()
    tool_id = client.get_current_tool_id()
    if tool_id == 11 or tool_id == 12 or tool_id == 13:  # If it is a gripper
        client.close_gripper()  # close gripper so that workspace is more visible
    client.move_pose(*observation_pose.to_list())
    return client
    
def main():
    robot_ip_address, workspace, observation_pose, drop_pose = data()
    client = connect(robot_ip_address, observation_pose)
    
    dict_color, dict_shape = import_dict()
    
    print("\nBienvenue dans l'interface de commande SmartRobot pour robot Ned2 ! :) \n")
    print("Prise d'un objet appris dans le workplace --> Entrer '1' ")
    print("   Les couleurs déjà enregistrées sont les suivantes : ")
    for a_color in dict_color:
        print("     -", a_color)
    print("   Les formes déjà enregistrées sont les suivantes : ")
    for a_shape in dict_shape:
        print("     -", a_shape)
    print("\nCe que vous souhaitez attraper n'est pas encore enregistré (couleur ou forme) ? ")
    print("Apprentissage d'un nouvel objet dans la base de donnée --> Entrer '2' \n")
    print("Quitter l'interface de commande --> Entrer '3'")
    q1 = input("\nVotre réponse : ")

    if q1 == "1":
        main_test_pixel(dict_color, dict_shape, client, workspace, drop_pose, observation_pose)
    
    if q1 == "2":
        main_new_objects(dict_color, dict_shape, client)
    
    return q1

q1 = "0"
while (q1 != "3"):
    q1 = main()