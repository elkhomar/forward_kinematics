import kinetix_scenegraph as ks
from kinetix_scenegraph.conversational.fbx_to_scene import fbx_to_scenegraph

scene = fbx_to_scenegraph("/home/omar/Downloads/test.fbx")
scene.render()