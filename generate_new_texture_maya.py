import maya.standalone
maya.standalone.initialize(name='python')

import maya.cmds as cmds
import maya.mel as mel

def import_obj(file_path):
    # 导入OBJ文件
    cmds.file(file_path, i=True, type="OBJ", ignoreVersion=True, ra=True, mergeNamespacesOnClash=False, namespace=":", options="mo=1", pr=True, importTimeRange="combine")

def auto_uv_unwrap(obj_name):
    # 选择对象
    cmds.select(obj_name)

    # 创建UV集
    cmds.polyUVSet(create=True, uvSet='myUVSet')

    # 设置当前UV集
    cmds.polyUVSet(currentUVSet=True, uvSet='myUVSet')

    # 执行自动UV展开
    mel.eval('polyAutoProjection -lm 0 -pb 0 -ib 4 -cm 1 -l 2 -sc 1 -o 1 -p 6 -ps 0.2 -ws 0;')

    # 显示UV编辑器
    mel.eval('TextureViewWindow;')

# 示例：导入并对其进行UV展开
file_path = "path_to_your_obj_file.obj"
import_obj(file_path)

# 获取导入的模型名称
imported_objects = cmds.ls(assemblies=True)
if imported_objects:
    obj_name = imported_objects[0]
    auto_uv_unwrap(obj_name)
else:
    print("没有找到导入的对象")